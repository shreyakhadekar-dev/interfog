"""FastAPI application: REST API, live WebSocket feed and the dashboard, all from one process.

Run locally:  uvicorn app.main:app --reload      then open http://localhost:8000
API docs:     http://localhost:8000/docs
"""
import asyncio
import math
import os
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import __version__
from app.core import (AvailabilityService, Decision, NodeState, NodeStore, NodeTelemetry, OutcomeReport,
                      ReassessRequest, Scheduler, Settings, TaskRequest, calibration_summary)
from app.ml import ModelHub, auto_forecast
from app.simulate import SCENARIOS, compare, list_scenarios, scenario_nodes

STATIC = Path(__file__).parent / "static"


# ----------------------------------------------------------------------------- request bodies
class TrainRequest(BaseModel):
    n_samples: int = Field(1500, ge=200, le=6000)
    seed: int = 42
    activate: Optional[str] = Field("best", description="'best', a model name, or null to keep the current model")


class ActivateRequest(BaseModel):
    name: str


class CalibrateRequest(BaseModel):
    method: str = "platt"
    min_outcomes: int = Field(30, ge=5)


class SimRequest(BaseModel):
    scenario: str = "S1"
    n_tasks: int = Field(1000, ge=100, le=3000)
    seed: int = 7


# ----------------------------------------------------------------------------- live feed
class Hub:
    """Fan-out of events to every connected dashboard."""

    def __init__(self) -> None:
        self.clients: set = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    async def broadcast(self, event: str, data) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json({"event": event, "data": data})
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def publish(self, event: str, data) -> None:
        """Safe to call from sync endpoints running in the thread pool."""
        if self.loop and self.clients:
            asyncio.run_coroutine_threadsafe(self.broadcast(event, data), self.loop)


def _nodes_json(store: NodeStore) -> list:
    return [n.model_dump(mode="json") for n in store.all()]


# ----------------------------------------------------------------------------- app
def create_app() -> FastAPI:
    cfg = Settings()
    store = NodeStore()
    avail = AvailabilityService()
    models = ModelHub(avail)
    hub = Hub()
    decisions: deque = deque(maxlen=300)
    by_task: Dict[str, Decision] = {}
    outcomes: List[tuple] = []  # (predicted_p, completed)
    demo: Dict[str, dict] = {}  # node_id -> {"base": harvest watts, "phase": radians}
    tick_speed = float(os.getenv("TICK_SPEED", "0.5"))

    def load_nodes(nodes: List[dict]) -> List[NodeState]:
        store.clear()
        demo.clear()
        out = []
        for i, n in enumerate(nodes):
            out.append(store.upsert(NodeTelemetry(**n)))
            demo[n["node_id"]] = {"base": n["harvest_power_w"], "phase": i * 1.7}
        return out

    async def ticker() -> None:
        """Gently move the demo nodes so the dashboard is alive: harvest wobbles, batteries drain and recharge."""
        t = 0.0
        while True:
            await asyncio.sleep(2.0)
            t += 2.0
            for nid, d in list(demo.items()):
                s = store.get(nid)
                if s is None:
                    continue
                harvest = max(0.0, d["base"] * (1 + 0.3 * math.sin(t / 25 + d["phase"])) + np.random.normal(0, 0.02))
                soc = s.battery_soc + (harvest - s.consumption_power_w) * 2.0 * tick_speed / s.capacity_j
                cycles = s.power_cycles
                if soc <= 0.02:  # node browned out and rebooted
                    soc, cycles = 0.2, cycles + 1
                store.upsert(NodeTelemetry(**{**s.model_dump(exclude={"timestamp"}), "battery_soc": min(1.0, soc),
                                              "harvest_power_w": round(harvest, 4), "power_cycles": cycles}))
            await hub.broadcast("nodes", _nodes_json(store))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        np.random.seed(cfg.seed)
        hub.loop = asyncio.get_running_loop()
        task = None
        if cfg.enable_demo:
            load_nodes(scenario_nodes("S1"))
            task = asyncio.create_task(ticker())
        try:
            yield
        finally:
            if task:
                task.cancel()

    app = FastAPI(title="INTERFOG", version=__version__, lifespan=lifespan,
                  description="Probability-of-completion orchestration for intermittently powered fog nodes.")
    app.add_middleware(CORSMiddleware, allow_origins=cfg.cors_origins, allow_methods=["*"], allow_headers=["*"])
    api = APIRouter(prefix="/api/v1")
    scheduler = Scheduler(avail, store, cfg)

    # ---- system
    @app.get("/health", tags=["system"])
    def health():
        return {"status": "ok", "version": __version__, "nodes": len(store), "model": avail.name}

    # ---- nodes
    @api.post("/telemetry", response_model=NodeState, status_code=202, tags=["nodes"])
    def post_telemetry(t: NodeTelemetry):
        demo.pop(t.node_id, None)  # real data replaces the animated demo node
        state = store.upsert(t)
        hub.publish("nodes", _nodes_json(store))
        return state

    @api.get("/nodes", response_model=List[NodeState], tags=["nodes"])
    def list_nodes():
        return store.all()

    @api.get("/nodes/{node_id}", response_model=NodeState, tags=["nodes"])
    def get_node(node_id: str):
        n = store.get(node_id)
        if n is None:
            raise HTTPException(404, f"Unknown node '{node_id}'")
        return n

    @api.get("/availability", tags=["nodes"])
    def availability(tau_s: float = Query(8.0, gt=0, le=86400)):
        out = []
        for s in store.all():
            life = avail.lifetime(s)
            out.append({"node_id": s.node_id, "tau_s": tau_s, "probability": round(avail.probability(s, tau_s), 6),
                        "expected_lifetime_s": None if math.isinf(life) else round(life, 2), "model": avail.name})
        return out

    @api.get("/survival-curves", tags=["models"])
    def survival_curves(max_tau_s: float = Query(40.0, gt=0, le=3600), points: int = Query(41, ge=5, le=200)):
        ts = np.linspace(0, max_tau_s, points)
        return {"model": avail.name, "ts": [round(float(t), 3) for t in ts],
                "curves": {s.node_id: [round(avail.probability(s, float(t)), 5) for t in ts] for s in store.all()}}

    @api.get("/nodes/{node_id}/forecast", tags=["models"])
    def forecast(node_id: str, horizon_s: float = Query(60.0, gt=0, le=3600)):
        hist = store.history(node_id)
        if not hist:
            raise HTTPException(404, f"No telemetry history for '{node_id}'")
        series = [h.harvest_power_w for h in hist]
        ts = [h.timestamp.timestamp() for h in hist]
        step = max(float(np.median(np.diff(ts))) if len(ts) > 1 else 2.0, 0.5)
        steps = min(60, max(1, int(math.ceil(horizon_s / step))))
        f = auto_forecast(series, steps)
        return {"node_id": node_id, "step_s": round(step, 2), "method": f["method"], "history_w": series[-120:],
                "forecast_w": [round(v, 4) for v in f["path"]], "validation_mae": f["mae"]}

    # ---- models
    @api.get("/models", tags=["models"])
    def models_status():
        return models.status()

    @api.post("/models/train", tags=["models"])
    def models_train(body: TrainRequest):
        try:
            models.train(body.n_samples, body.seed, body.activate)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return models.status()

    @api.post("/models/activate", tags=["models"])
    def models_activate(body: ActivateRequest):
        try:
            models.activate(body.name)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return models.status()

    @api.post("/models/calibrate", tags=["models"])
    def models_calibrate(body: CalibrateRequest):
        try:
            return models.calibrate([o[0] for o in outcomes], [1.0 if o[1] else 0.0 for o in outcomes],
                                    body.method, body.min_outcomes)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @api.delete("/models/calibrate", tags=["models"])
    def models_clear_calibration():
        models.clear_calibration()
        return models.status()

    # ---- scheduling
    def _record(d: Decision) -> None:
        decisions.append(d)
        by_task[d.task_id] = d
        if len(by_task) > 2000:
            by_task.pop(next(iter(by_task)))
        hub.publish("decision", d.model_dump(mode="json"))

    @api.post("/tasks", response_model=Decision, status_code=201, tags=["scheduling"])
    def schedule_task(task: TaskRequest):
        d = scheduler.decide(task)
        _record(d)
        return d

    @api.post("/tasks/reassess", response_model=Decision, tags=["scheduling"])
    def reassess_task(body: ReassessRequest):
        if store.get(body.node_id) is None:
            raise HTTPException(404, f"Unknown node '{body.node_id}'")
        d = scheduler.reassess(body.task, body.node_id, body.elapsed_s)
        _record(d)
        return d

    @api.get("/decisions", response_model=List[Decision], tags=["scheduling"])
    def list_decisions(limit: int = Query(30, ge=1, le=300)):
        return list(decisions)[-limit:][::-1]

    @api.post("/outcomes", status_code=201, tags=["scheduling"])
    def report_outcome(o: OutcomeReport):
        d = by_task.get(o.task_id)
        if d is None:
            raise HTTPException(404, f"No decision recorded for task '{o.task_id}'")
        outcomes.append((d.p_complete, o.completed))
        return {"task_id": o.task_id, "completed": o.completed, "predicted_p": d.p_complete}

    @api.get("/calibration", tags=["scheduling"])
    def calibration(bins: int = Query(10, ge=2, le=50)):
        return calibration_summary([o[0] for o in outcomes], [1.0 if o[1] else 0.0 for o in outcomes], bins)

    # ---- scenarios and simulation
    @api.get("/scenarios", tags=["simulation"])
    def scenarios():
        return list_scenarios()

    @api.post("/scenarios/{scenario_id}/load", response_model=List[NodeState], tags=["simulation"])
    def load_scenario(scenario_id: str):
        if scenario_id not in SCENARIOS:
            raise HTTPException(404, f"Unknown scenario '{scenario_id}'")
        nodes = load_nodes(scenario_nodes(scenario_id))
        hub.publish("nodes", _nodes_json(store))
        return nodes

    @api.post("/simulate", tags=["simulation"])
    def simulate(body: SimRequest):
        try:
            return compare(avail, cfg, body.scenario, body.n_tasks, body.seed)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    app.include_router(api)

    # ---- live updates
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        hub.clients.add(ws)
        try:
            await ws.send_json({"event": "nodes", "data": _nodes_json(store)})
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(ws)

    # ---- dashboard
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()

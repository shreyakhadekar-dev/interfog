"""Run with:  pip install pytest httpx  &&  python -m pytest -q"""
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core import (Action, AvailabilityService, Criticality, EnergyModel, NodeStore, NodeTelemetry, Scheduler,
                      Settings, TaskRequest, calibration_summary)
from app.main import create_app
from app.ml import (CoxPH, DeepSurv, ModelHub, PlattScaler, WeibullAFT, auto_forecast, concordance_index,
                    make_calibrator, synthetic_lifetimes)
from app.simulate import SCENARIOS, compare


def node(nid, soc=0.5, cap=40, harvest=0.0, cons=1.0, lat=20, queue=0):
    return NodeTelemetry(node_id=nid, battery_soc=soc, capacity_j=cap, harvest_power_w=harvest,
                         consumption_power_w=cons, latency_ms=lat, queue_length=queue)


def sched(*nodes):
    store = NodeStore()
    for n in nodes:
        store.upsert(n)
    return Scheduler(AvailabilityService(), store, Settings())


# ----------------------------------------------------------------------------- availability
def test_energy_model_matches_hand_calculation():
    s = NodeStore().upsert(node("a", soc=0.30, cap=40, harvest=0.2, cons=1.4))
    m = EnergyModel()
    assert m.expected_lifetime_s(s) == pytest.approx((0.30 - 0.05) * 40 / 1.2)
    assert m.survival_probability(s, 8.0) == pytest.approx(math.exp(-((8 / (10 / 1.2)) ** 2)) * math.exp(-1e-4 * 8), rel=1e-6)


def test_self_sustaining_node_never_depletes():
    s = NodeStore().upsert(node("a", soc=0.5, harvest=2.0, cons=1.0))
    assert math.isinf(EnergyModel().expected_lifetime_s(s))
    assert EnergyModel().survival_probability(s, 3600) > 0.69  # only random outages remain


# ----------------------------------------------------------------------------- the six actions
def test_execute_picks_the_safe_node():
    d = sched(node("weak", soc=0.2), node("strong", soc=0.9, harvest=1.5, lat=50)).decide(
        TaskRequest(exec_time_s=8, deadline_s=30, criticality=Criticality.deadline_critical))
    assert d.action == Action.execute and d.target_node_id == "strong"


def test_replicate_when_no_single_node_is_safe_enough():
    a, b = node("a", soc=0.425, cons=1.0), node("b", soc=0.425, cons=1.0, lat=25)
    d = sched(a, b).decide(TaskRequest(exec_time_s=8, deadline_s=30, criticality=Criticality.deadline_critical))
    assert d.action == Action.replicate and set(d.replica_node_ids) == {"a", "b"} and d.p_complete > 0.85


def test_delay_best_effort_work_waits_for_harvest():
    d = sched(node("a", soc=0.30, cap=20, harvest=0.05, cons=1.0)).decide(
        TaskRequest(exec_time_s=8, deadline_s=80, criticality=Criticality.best_effort))
    assert d.action == Action.delay and d.retry_after_s == 30


def test_cloud_fallback_when_fog_is_too_weak():
    d = sched(node("a", soc=0.2, cap=20)).decide(
        TaskRequest(exec_time_s=8, deadline_s=20, criticality=Criticality.deadline_critical))
    assert d.action == Action.cloud_fallback and d.target_node_id == "cloud"


def test_offload_to_peer_when_favourite_is_full():
    full = node("full", soc=1.0, cap=100, harvest=2.0, lat=5, queue=8)
    peer = node("peer", soc=0.9, cap=100, harvest=2.0, lat=40)
    d = sched(full, peer).decide(TaskRequest(exec_time_s=8, deadline_s=30, criticality=Criticality.best_effort))
    assert d.action == Action.offload_peer and d.target_node_id == "peer"


def test_checkpoint_and_migrate_before_a_node_runs_dry():
    s = sched(node("weak", soc=0.15, cap=40, cons=1.2), node("strong", soc=0.9, cap=100, harvest=2.0, lat=40))
    d = s.reassess(TaskRequest(exec_time_s=20, deadline_s=60, criticality=Criticality.deadline_critical), "weak", 6)
    assert d.action == Action.checkpoint_migrate and d.target_node_id == "strong"


# ----------------------------------------------------------------------------- ML
@pytest.fixture(scope="module")
def data():
    return synthetic_lifetimes(1200, seed=3)


def test_survival_models_rank_better_than_chance(data):
    X, d, e = data
    for cls in (WeibullAFT, CoxPH, DeepSurv):
        m = cls().fit(X, d, e)
        assert concordance_index(d, m.risk_score(X), e) > 0.7, cls.__name__
        s = m.survival_matrix(X[:5], 8.0)
        assert np.all((s >= 0) & (s <= 1))


def test_survival_is_monotone_in_time(data):
    X, d, e = data
    m = WeibullAFT().fit(X, d, e)
    vals = [float(m.survival_matrix(X[:1], t)[0]) for t in (1, 5, 20, 60)]
    assert vals == sorted(vals, reverse=True)


def test_calibration_reduces_overconfidence():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(size=4000) < true_p).astype(float)
    over = np.clip(true_p ** 0.4, 0, 1)  # systematically overconfident
    before = calibration_summary(over, y)["ece"]
    for method in ("platt", "temperature", "isotonic"):
        cal = make_calibrator(method).fit(over, y)
        after = calibration_summary(cal.transform(over), y)["ece"]
        # temperature scaling has one parameter, so it improves but cannot fix a skewed bias completely
        assert after < (before if method == "temperature" else before / 2), method


def test_forecast_follows_a_steady_signal():
    out = auto_forecast(np.full(40, 0.8) + np.random.default_rng(1).normal(0, 0.01, 40), 5)
    assert all(abs(v - 0.8) < 0.1 for v in out["path"])


def test_model_hub_trains_and_switches():
    hub = ModelHub(AvailabilityService())
    hub.train(600, seed=1, activate="best")
    assert hub.report["best"] in hub.trained and hub.avail.model is hub.trained[hub.report["best"]]
    hub.activate("heuristic")
    assert hub.avail.model.name == "energy-heuristic"
    with pytest.raises(ValueError):
        hub.activate("nope")


# ----------------------------------------------------------------------------- simulation
def test_interfog_beats_naive_schedulers_when_energy_is_scarce():
    r = compare(AvailabilityService(), Settings(), "S3", 600, seed=5)["policies"]
    assert r["interfog"]["completion_rate"] > r["latency"]["completion_rate"] + 0.3
    assert r["interfog"]["completion_rate"] > r["energy"]["completion_rate"]


def test_calibration_helps_when_the_model_is_overconfident():
    c = compare(AvailabilityService(), Settings(), "S4", 800, seed=5)["calibration"]
    assert c["after"]["ece"] < c["before"]["ece"]


def test_every_scenario_runs():
    for sc in SCENARIOS:
        out = compare(AvailabilityService(), Settings(), sc, 100, seed=1)
        assert set(out["policies"]) >= {"latency", "interfog"}


# ----------------------------------------------------------------------------- API
@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c


def test_dashboard_and_health(client):
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200 and "INTERFOG" in page.text


def test_demo_nodes_are_seeded(client):
    nodes = client.get("/api/v1/nodes").json()
    assert {n["node_id"] for n in nodes} == {"fog-1", "fog-2", "fog-3", "fog-4"}


def test_schedule_reassess_and_outcomes(client):
    d = client.post("/api/v1/tasks", json={"task_id": "t1", "exec_time_s": 8, "deadline_s": 25,
                                            "criticality": "deadline_critical"})
    assert d.status_code == 201 and d.json()["action"] in {a.value for a in Action}
    r = client.post("/api/v1/tasks/reassess", json={"task": {"task_id": "t1", "exec_time_s": 8, "deadline_s": 25},
                                                     "node_id": "fog-1", "elapsed_s": 4})
    assert r.status_code == 200
    assert client.post("/api/v1/outcomes", json={"task_id": "t1", "completed": True}).status_code == 201
    assert client.post("/api/v1/outcomes", json={"task_id": "ghost", "completed": True}).status_code == 404
    assert client.get("/api/v1/calibration").json()["n"] == 1
    assert client.post("/api/v1/models/calibrate", json={}).status_code == 422  # too few outcomes


def test_bad_input_is_rejected(client):
    assert client.post("/api/v1/tasks", json={"exec_time_s": -1, "deadline_s": 5}).status_code == 422
    assert client.get("/api/v1/nodes/nope").status_code == 404
    assert client.post("/api/v1/scenarios/S9/load").status_code == 404
    assert client.post("/api/v1/simulate", json={"scenario": "S9"}).status_code == 422


def test_models_endpoints(client):
    assert client.post("/api/v1/models/activate", json={"name": "deepsurv"}).status_code == 422
    st = client.post("/api/v1/models/train", json={"n_samples": 500}).json()
    assert st["trained"] and st["active"] in st["available"]
    assert client.post("/api/v1/models/activate", json={"name": "heuristic"}).json()["active"] == "energy-heuristic"
    curves = client.get("/api/v1/survival-curves?max_tau_s=30&points=11").json()
    assert len(curves["ts"]) == 11 and set(curves["curves"]) == {"fog-1", "fog-2", "fog-3", "fog-4"}


def test_scenario_load_and_simulate(client):
    nodes = client.post("/api/v1/scenarios/S5/load").json()
    assert len(nodes) == 6
    out = client.post("/api/v1/simulate", json={"scenario": "S5", "n_tasks": 100}).json()
    assert out["n_tasks"] == 100 and "interfog" in out["policies"]


def test_telemetry_and_forecast(client):
    for i in range(20):
        client.post("/api/v1/telemetry", json={"node_id": "x1", "battery_soc": 0.5, "capacity_j": 30,
                                                "harvest_power_w": 0.5 + 0.01 * i, "consumption_power_w": 0.4})
    assert client.get("/api/v1/nodes/x1/forecast").json()["method"] in {"ewma", "holt", "ar"}


def test_websocket_sends_a_snapshot(client):
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["event"] == "nodes" and len(msg["data"]) >= 4

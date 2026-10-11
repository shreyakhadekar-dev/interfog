"""Core building blocks: settings, data models, node store, availability, scheduler and metrics.

Everything the scheduler needs lives in this one file so the whole decision path can be read top to bottom.
"""
import math
import os
import threading
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Deque, Dict, List, Optional, Sequence
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, computed_field


# ----------------------------------------------------------------------------- settings
def _env(name: str, default, cast=float):
    raw = os.getenv(name)
    if raw is None:
        return default
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return cast(raw)


class Settings:
    """All tunables, overridable with environment variables of the same name in upper case."""

    def __init__(self) -> None:
        # objective: maximise alpha*P - beta*L - gamma*E - delta*M - epsilon*C - zeta*V
        self.alpha = _env("ALPHA", 1.0)
        self.beta = _env("BETA", 0.25)
        self.gamma = _env("GAMMA", 0.15)
        self.delta = _env("DELTA", 0.20)
        self.epsilon = _env("EPSILON", 0.10)
        self.zeta = _env("ZETA", 1.0)
        # minimum completion probability per task class
        self.p_min_life_critical = _env("P_MIN_LIFE_CRITICAL", 0.95)
        self.p_min_deadline_critical = _env("P_MIN_DEADLINE_CRITICAL", 0.85)
        self.p_min_best_effort = _env("P_MIN_BEST_EFFORT", 0.60)
        # scheduler behaviour
        self.max_queue_length = _env("MAX_QUEUE_LENGTH", 8, int)
        self.delay_window_s = _env("DELAY_WINDOW_S", 30.0)
        self.cloud_enabled = _env("CLOUD_ENABLED", True, bool)
        self.cloud_latency_ms = _env("CLOUD_LATENCY_MS", 120.0)
        self.cloud_availability = _env("CLOUD_AVAILABILITY", 0.999)
        self.migration_s_per_mb = _env("MIGRATION_S_PER_MB", 0.5)
        self.checkpoint_overhead_s = _env("CHECKPOINT_OVERHEAD_S", 0.4)
        self.migration_margin = _env("MIGRATION_MARGIN", 0.05)
        # service
        self.enable_demo = _env("ENABLE_DEMO", True, bool)  # seed nodes and animate them
        self.cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
        self.seed = _env("SEED", 42, int)


# ----------------------------------------------------------------------------- data models
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


class Criticality(str, Enum):
    life_critical = "life_critical"
    deadline_critical = "deadline_critical"
    best_effort = "best_effort"


class Action(str, Enum):
    """The six scheduling actions."""

    execute = "execute"
    delay = "delay"
    checkpoint_migrate = "checkpoint_migrate"
    replicate = "replicate"
    offload_peer = "offload_peer"
    cloud_fallback = "cloud_fallback"


class NodeTelemetry(BaseModel):
    node_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    timestamp: datetime = Field(default_factory=utcnow)
    battery_soc: float = Field(ge=0.0, le=1.0, description="State of charge, 0 to 1")
    capacity_j: float = Field(gt=0, description="Energy storage capacity in joules")
    harvest_power_w: float = Field(default=0.0, ge=0)
    consumption_power_w: float = Field(default=0.0, ge=0)
    cpu_util: float = Field(default=0.0, ge=0.0, le=1.0)
    queue_length: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    source: str = Field(default="solar", max_length=32)
    power_cycles: int = Field(default=0, ge=0)


class NodeState(NodeTelemetry):
    @computed_field
    @property
    def stored_energy_j(self) -> float:
        return self.battery_soc * self.capacity_j

    @computed_field
    @property
    def net_power_w(self) -> float:
        """Harvest minus consumption. Negative means the node is draining."""
        return self.harvest_power_w - self.consumption_power_w


class TaskRequest(BaseModel):
    task_id: str = Field(default_factory=lambda: new_id("task"), max_length=128)
    exec_time_s: float = Field(gt=0, le=86400)
    energy_j: Optional[float] = Field(default=None, ge=0, description="Energy the task needs; estimated if omitted")
    deadline_s: float = Field(gt=0)
    criticality: Criticality = Criticality.best_effort
    state_size_mb: float = Field(default=1.0, ge=0, description="Checkpoint size, drives migration cost")


class CandidateScore(BaseModel):
    node_id: str
    p_complete: float
    latency_ms: float
    energy_j: float
    saturated: bool = False
    violates_deadline: bool = False
    score: float


class Decision(BaseModel):
    decision_id: str = Field(default_factory=lambda: new_id("dec"))
    task_id: str
    action: Action
    target_node_id: Optional[str] = None
    replica_node_ids: List[str] = Field(default_factory=list)
    retry_after_s: Optional[float] = None
    p_complete: float
    candidates: List[CandidateScore] = Field(default_factory=list)
    reason: str
    created_at: datetime = Field(default_factory=utcnow)


class ReassessRequest(BaseModel):
    task: TaskRequest
    node_id: str
    elapsed_s: float = Field(ge=0)


class OutcomeReport(BaseModel):
    task_id: str
    completed: bool


# ----------------------------------------------------------------------------- metrics
def brier_score(p: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if not len(p):
        return None
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def calibration_bins(p: Sequence[float], y: Sequence[float], n_bins: int = 10) -> List[dict]:
    """Reliability-diagram data: predicted vs observed completion rate per probability bin."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1) if len(p) else np.array([], int)
    out = []
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        out.append({"lower": b / n_bins, "upper": (b + 1) / n_bins, "count": c,
                    "mean_predicted": float(p[m].mean()) if c else None,
                    "observed_rate": float(y[m].mean()) if c else None})
    return out


def expected_calibration_error(p: Sequence[float], y: Sequence[float], n_bins: int = 10) -> Optional[float]:
    if not len(p):
        return None
    n = len(p)
    return float(sum(b["count"] / n * abs(b["observed_rate"] - b["mean_predicted"])
                     for b in calibration_bins(p, y, n_bins) if b["count"]))


def calibration_summary(p: Sequence[float], y: Sequence[float], n_bins: int = 10) -> dict:
    return {"n": len(p), "completion_rate": float(np.mean(y)) if len(y) else None,
            "brier_score": brier_score(p, y), "ece": expected_calibration_error(p, y, n_bins),
            "bins": calibration_bins(p, y, n_bins)}


# ----------------------------------------------------------------------------- node store
class NodeStore:
    """Latest state of every node plus a short telemetry history (used for forecasting)."""

    def __init__(self, history: int = 240) -> None:
        self._nodes: Dict[str, NodeState] = {}
        self._hist: Dict[str, Deque[NodeTelemetry]] = {}
        self._max = history
        self._lock = threading.Lock()

    def upsert(self, t: NodeTelemetry) -> NodeState:
        state = NodeState(**t.model_dump())
        with self._lock:
            self._nodes[t.node_id] = state
            self._hist.setdefault(t.node_id, deque(maxlen=self._max)).append(t)
        return state

    def get(self, node_id: str) -> Optional[NodeState]:
        return self._nodes.get(node_id)

    def all(self) -> List[NodeState]:
        return sorted(self._nodes.values(), key=lambda s: s.node_id)

    def history(self, node_id: str) -> List[NodeTelemetry]:
        return list(self._hist.get(node_id, []))

    def clear(self) -> None:
        with self._lock:
            self._nodes.clear()
            self._hist.clear()

    def __len__(self) -> int:
        return len(self._nodes)


# ----------------------------------------------------------------------------- availability
class EnergyModel:
    """Transparent baseline: energy-balance lifetime with a Weibull-shaped depletion curve.

    lifetime = usable_energy / net_drain, P(survive tau) = exp(-(tau/lifetime)^k) * exp(-outage_rate * tau).
    A node whose harvest covers its consumption has unbounded lifetime; only random outages apply.
    """

    name = "energy-heuristic"

    def __init__(self, shape: float = 2.0, outage_rate_per_s: float = 1e-4, reserve_soc: float = 0.05) -> None:
        self.shape, self.outage_rate_per_s, self.reserve_soc = shape, outage_rate_per_s, reserve_soc

    def expected_lifetime_s(self, s: NodeState) -> float:
        usable = max(0.0, s.battery_soc - self.reserve_soc) * s.capacity_j
        if usable <= 0.0:
            return 0.0
        drain = s.consumption_power_w - s.harvest_power_w
        return math.inf if drain <= 1e-9 else usable / drain

    def survival_probability(self, s: NodeState, tau_s: float) -> float:
        if tau_s <= 0:
            return 1.0
        life = self.expected_lifetime_s(s)
        if life <= 0.0:
            return 0.0
        depletion = 1.0 if math.isinf(life) else math.exp(-((tau_s / life) ** self.shape))
        return max(0.0, min(1.0, depletion * math.exp(-self.outage_rate_per_s * tau_s)))


class AvailabilityService:
    """Answers P(node stays online for tau seconds). Holds a base model and an optional calibrator layer."""

    def __init__(self, model=None) -> None:
        self.model = model or EnergyModel()
        self.calibrator = None  # callable p -> p', fitted from observed outcomes

    @property
    def name(self) -> str:
        return self.model.name + (f"+{self.calibrator.name}" if self.calibrator else "")

    def probability(self, s: NodeState, tau_s: float) -> float:
        p = self.model.survival_probability(s, tau_s)
        if self.calibrator is not None and tau_s > 0:
            p = max(0.0, min(1.0, self.calibrator(p)))
        return p

    def lifetime(self, s: NodeState) -> float:
        return self.model.expected_lifetime_s(s)


# ----------------------------------------------------------------------------- scheduler
CLOUD = "cloud"


class Scheduler:
    """Scores every candidate with the proposal's objective and picks one of the six actions."""

    def __init__(self, availability: AvailabilityService, store: NodeStore, cfg: Settings) -> None:
        self.avail, self.store, self.cfg = availability, store, cfg

    def _p_min(self, c: Criticality) -> float:
        return {Criticality.life_critical: self.cfg.p_min_life_critical,
                Criticality.deadline_critical: self.cfg.p_min_deadline_critical,
                Criticality.best_effort: self.cfg.p_min_best_effort}[c]

    def _score(self, p, latency_ms, energy_frac, violates, deadline_s, migration_s=0.0, checkpoint_s=0.0) -> float:
        c = self.cfg
        return (c.alpha * p
                - c.beta * min(latency_ms / (deadline_s * 1000.0), 1.0)
                - c.gamma * min(energy_frac, 1.0)
                - c.delta * min(migration_s / deadline_s, 1.0)
                - c.epsilon * min(checkpoint_s / deadline_s, 1.0)
                - c.zeta * (1.0 if violates else 0.0))

    def _candidate(self, s: NodeState, task: TaskRequest) -> CandidateScore:
        p = self.avail.probability(s, task.exec_time_s)
        energy = task.energy_j if task.energy_j is not None else s.consumption_power_w * task.exec_time_s
        violates = s.latency_ms / 1000.0 + task.exec_time_s > task.deadline_s
        return CandidateScore(node_id=s.node_id, p_complete=round(p, 6), latency_ms=s.latency_ms,
                              energy_j=round(energy, 4), saturated=s.queue_length >= self.cfg.max_queue_length,
                              violates_deadline=violates,
                              score=round(self._score(p, s.latency_ms, energy / s.capacity_j, violates, task.deadline_s), 6))

    def _cloud_fits(self, task: TaskRequest, extra_s: float = 0.0) -> bool:
        return self.cfg.cloud_enabled and extra_s + self.cfg.cloud_latency_ms / 1000.0 + task.exec_time_s <= task.deadline_s

    def _cloud_candidate(self, task: TaskRequest) -> CandidateScore:
        p, lat = self.cfg.cloud_availability, self.cfg.cloud_latency_ms
        viol = lat / 1000.0 + task.exec_time_s > task.deadline_s
        return CandidateScore(node_id=CLOUD, p_complete=p, latency_ms=lat, energy_j=0.0, violates_deadline=viol,
                              score=round(self._score(p, lat, 0.0, False, task.deadline_s), 6))

    def _decision(self, task, action, cands, p, reason, target=None, replicas=None, retry_after_s=None) -> Decision:
        shown = list(cands) + ([self._cloud_candidate(task)] if self.cfg.cloud_enabled else [])
        return Decision(task_id=task.task_id, action=action, target_node_id=target, replica_node_ids=replicas or [],
                        retry_after_s=retry_after_s, p_complete=round(p, 6), candidates=shown, reason=reason)

    # ---- placement of a new task
    def decide(self, task: TaskRequest) -> Decision:
        cfg = self.cfg
        states = self.store.all()
        cands = [self._candidate(s, task) for s in states]
        p_min = self._p_min(task.criticality)
        usable = sorted((c for c in cands if not c.saturated and not c.violates_deadline),
                        key=lambda c: c.score, reverse=True)

        # 1. one node is likely enough: execute, or offload when the favourite is full
        if usable and usable[0].p_complete >= p_min:
            best = usable[0]
            fav = max(cands, key=lambda c: c.score)
            if fav.saturated and fav.node_id != best.node_id:
                return self._decision(task, Action.offload_peer, cands, best.p_complete, target=best.node_id,
                                      reason=f"{fav.node_id} scores highest but its queue is full, so the task goes to "
                                             f"{best.node_id} (P={best.p_complete:.2f}, needs {p_min:.2f}).")
            return self._decision(task, Action.execute, cands, best.p_complete, target=best.node_id,
                                  reason=f"{best.node_id} has the best score and P(complete)={best.p_complete:.2f} "
                                         f"clears the {p_min:.2f} bar.")

        # 2. critical work: replicate until the combined probability is high enough
        if task.criticality != Criticality.best_effort and len(usable) >= 2:
            chosen, fail = [], 1.0
            for c in sorted(usable, key=lambda c: c.p_complete, reverse=True)[:3]:
                chosen.append(c)
                fail *= 1.0 - c.p_complete
                if len(chosen) >= 2 and 1.0 - fail >= p_min:
                    break
            combined = 1.0 - fail
            if len(chosen) >= 2 and combined >= p_min:
                ids = [c.node_id for c in chosen]
                return self._decision(task, Action.replicate, cands, combined, target=ids[0], replicas=ids,
                                      reason=f"No single node reaches {p_min:.2f}. {len(ids)} replicas together give "
                                             f"P={combined:.2f}.")

        # 3. best-effort work can wait for harvested energy to recover
        slack = task.deadline_s - task.exec_time_s - min((s.latency_ms for s in states), default=0.0) / 1000.0
        if (task.criticality == Criticality.best_effort and usable
                and any(s.harvest_power_w > 0 for s in states) and slack >= cfg.delay_window_s):
            return self._decision(task, Action.delay, cands, usable[0].p_complete, retry_after_s=cfg.delay_window_s,
                                  reason=f"Best P={usable[0].p_complete:.2f} is under {p_min:.2f}, but the deadline "
                                         f"leaves room to wait {cfg.delay_window_s:.0f}s for energy to recover.")

        # 4. the cloud has the availability the fog cannot offer
        if self._cloud_fits(task):
            return self._decision(task, Action.cloud_fallback, cands, cfg.cloud_availability, target=CLOUD,
                                  reason="No fog strategy reaches the required probability, so the task goes to the cloud."
                                  if states else "No fog nodes are reporting, so the task goes to the cloud.")

        # 5. last resort: best node available, below the bar
        pool = usable or [c for c in cands if not c.saturated] or cands
        if pool:
            best = max(pool, key=lambda c: c.p_complete)
            return self._decision(task, Action.execute, cands, best.p_complete, target=best.node_id,
                                  reason=f"Nothing reaches {p_min:.2f} and the cloud cannot meet the deadline. "
                                         f"{best.node_id} is the best available (P={best.p_complete:.2f}).")
        return self._decision(task, Action.delay, cands, 0.0, retry_after_s=cfg.delay_window_s,
                              reason="No nodes are reporting and the cloud is off. Retry later.")

    # ---- check on a task that is already running
    def reassess(self, task: TaskRequest, node_id: str, elapsed_s: float) -> Decision:
        cfg = self.cfg
        remaining = max(task.exec_time_s - elapsed_s, 0.1)
        p_min = self._p_min(task.criticality)
        cur = self.store.get(node_id)
        p_stay = self.avail.probability(cur, remaining) if cur else 0.0
        if p_stay >= p_min:
            return self._decision(task, Action.execute, [], p_stay, target=node_id,
                                  reason=f"Stay on {node_id}: P(survive the remaining {remaining:.1f}s)={p_stay:.2f} "
                                         f"clears {p_min:.2f}.")
        ckpt, mig = cfg.checkpoint_overhead_s, task.state_size_mb * cfg.migration_s_per_mb
        cands: List[CandidateScore] = []
        for s in self.store.all():
            if s.node_id == node_id or s.queue_length >= cfg.max_queue_length:
                continue
            p = self.avail.probability(s, remaining + mig)
            violates = elapsed_s + ckpt + mig + s.latency_ms / 1000.0 + remaining > task.deadline_s
            en = s.consumption_power_w * remaining
            cands.append(CandidateScore(node_id=s.node_id, p_complete=round(p, 6), latency_ms=s.latency_ms,
                                        energy_j=round(en, 4), violates_deadline=violates,
                                        score=round(self._score(p, s.latency_ms, en / s.capacity_j, violates,
                                                                task.deadline_s, mig, ckpt), 6)))
        viable = sorted((c for c in cands if not c.violates_deadline), key=lambda c: c.score, reverse=True)
        if viable and viable[0].p_complete > p_stay + cfg.migration_margin:
            b = viable[0]
            return self._decision(task, Action.checkpoint_migrate, cands, b.p_complete, target=b.node_id,
                                  reason=f"{node_id} is likely to run dry (P={p_stay:.2f} over {remaining:.1f}s). "
                                         f"Checkpoint now and move to {b.node_id} (P={b.p_complete:.2f}).")
        if self._cloud_fits(task, extra_s=elapsed_s + ckpt + mig + remaining - task.exec_time_s):
            return self._decision(task, Action.cloud_fallback, cands, cfg.cloud_availability, target=CLOUD,
                                  reason=f"{node_id} is at risk (P={p_stay:.2f}) and no peer is safer, so the task "
                                         f"resumes from its checkpoint in the cloud.")
        return self._decision(task, Action.execute, cands, p_stay, target=node_id,
                              reason=f"No safer option, so the task stays on {node_id} (P={p_stay:.2f}).")

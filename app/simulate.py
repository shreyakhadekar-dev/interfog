"""Scenarios S1-S7 and the Monte Carlo comparison of INTERFOG against simple schedulers.

Every policy sees the same tasks and the same hidden node lifetimes, so the comparison is paired. A node's true
lifetime comes from its physical energy balance, scaled by the scenario's noise and optimism, plus random outages.
The scheduler never sees the truth: it only sees the telemetry and its own availability model.
"""
import math
from typing import Dict, List, Optional

import numpy as np

from app.core import (Action, AvailabilityService, Criticality, EnergyModel, NodeState, NodeStore, NodeTelemetry,
                      Scheduler, Settings, TaskRequest, calibration_summary)
from app.ml import PlattScaler

DEFAULT_NODES: List[dict] = [
    # fog-1 is the fastest but nearly empty; fog-4 holds the most energy but burns it quickly;
    # fog-2 stores little yet harvests as much as it spends.
    dict(node_id="fog-1", battery_soc=0.30, capacity_j=40, harvest_power_w=0.2, consumption_power_w=1.4, latency_ms=20, source="solar", queue_length=1),
    dict(node_id="fog-2", battery_soc=0.55, capacity_j=60, harvest_power_w=0.8, consumption_power_w=0.75, latency_ms=31, source="solar", queue_length=0),
    dict(node_id="fog-3", battery_soc=0.60, capacity_j=50, harvest_power_w=0.2, consumption_power_w=1.1, latency_ms=27, source="rf", queue_length=2),
    dict(node_id="fog-4", battery_soc=0.95, capacity_j=100, harvest_power_w=0.3, consumption_power_w=2.2, latency_ms=45, source="vibration", queue_length=0),
]

_HETERO = [
    dict(node_id="fog-1", battery_soc=0.25, capacity_j=30, harvest_power_w=0.1, consumption_power_w=1.6, latency_ms=12, source="rf", queue_length=1),
    dict(node_id="fog-2", battery_soc=0.90, capacity_j=120, harvest_power_w=0.4, consumption_power_w=0.7, latency_ms=38, source="solar", queue_length=0),
    dict(node_id="fog-3", battery_soc=0.50, capacity_j=45, harvest_power_w=0.9, consumption_power_w=1.0, latency_ms=22, source="vibration", queue_length=3),
    dict(node_id="fog-4", battery_soc=0.70, capacity_j=90, harvest_power_w=1.8, consumption_power_w=1.1, latency_ms=55, source="solar", queue_length=0),
    dict(node_id="fog-5", battery_soc=0.15, capacity_j=25, harvest_power_w=0.05, consumption_power_w=1.3, latency_ms=9, source="rf", queue_length=2),
    dict(node_id="fog-6", battery_soc=0.60, capacity_j=70, harvest_power_w=0.6, consumption_power_w=0.9, latency_ms=33, source="solar", queue_length=1),
]

_BASE = dict(harvest_x=1.0, cons_x=1.0, soc_x=1.0, queue_add=0, exec=(3, 28), slack=(1.4, 3.0),
             crit=(0.10, 0.30, 0.60), sigma=0.2, optimism=1.0, outage=1e-4, nodes=None)

SCENARIOS: Dict[str, dict] = {
    "S1": dict(_BASE, name="Stable harvesting", blurb="Steady sunlight and vibration. Nodes slowly drain and recover."),
    "S2": dict(_BASE, name="Energy drought", blurb="Harvest drops to a sliver. Stored energy is all that is left.",
               harvest_x=0.15, soc_x=0.7),
    "S3": dict(_BASE, name="Workload spike", blurb="Longer tasks, busier CPUs and fuller queues.",
               cons_x=1.45, queue_add=3, exec=(6, 40)),
    "S4": dict(_BASE, name="Repeated interruption", blurb="Nodes drop out at random, far more often than the model assumes.",
               outage=0.02, soc_x=0.85),
    "S5": dict(_BASE, name="Heterogeneous fog", blurb="Six very different nodes, from tiny and fast to big and slow.",
               nodes=_HETERO),
    "S6": dict(_BASE, name="Prediction uncertainty", blurb="The model is overconfident. Calibration is what saves tasks here.",
               sigma=0.5, optimism=0.6, harvest_x=0.7),
    "S7": dict(_BASE, name="Deadline-critical mix", blurb="Mostly critical tasks with tight deadlines.",
               crit=(0.25, 0.55, 0.20), slack=(1.15, 1.7)),
}


def scenario_nodes(sc_id: str) -> List[dict]:
    """Node telemetry for a scenario, with the scenario's multipliers applied."""
    sc = SCENARIOS[sc_id]
    out = []
    for n in sc["nodes"] or DEFAULT_NODES:
        n = dict(n)
        n["harvest_power_w"] = round(n["harvest_power_w"] * sc["harvest_x"], 4)
        n["consumption_power_w"] = round(n["consumption_power_w"] * sc["cons_x"], 4)
        n["battery_soc"] = round(min(1.0, n["battery_soc"] * sc["soc_x"]), 4)
        n["queue_length"] = n["queue_length"] + sc["queue_add"]
        out.append(n)
    return out


def list_scenarios() -> List[dict]:
    return [{"id": k, "name": v["name"], "blurb": v["blurb"], "nodes": len(v["nodes"] or DEFAULT_NODES)}
            for k, v in SCENARIOS.items()]


# ----------------------------------------------------------------------------- policy evaluation
POLICIES = ["latency", "energy", "random", "interfog", "interfog_calibrated"]
TRUTH = EnergyModel()


def _lifetimes(rng, states: List[NodeState], sc: dict) -> Dict[str, float]:
    """Hidden true lifetime of each node from now: noisy energy balance, capped by random outages."""
    out = {}
    for s in states:
        life = TRUTH.expected_lifetime_s(s)
        outage = rng.exponential(1.0 / sc["outage"])
        if life <= 0:
            out[s.node_id] = 0.0
        elif math.isinf(life):
            out[s.node_id] = outage
        else:
            scale = life * sc["optimism"] * rng.lognormal(0.0, sc["sigma"])
            out[s.node_id] = min(scale * rng.weibull(2.0), outage)
    return out


def _make_task(rng, sc: dict, i: int) -> TaskRequest:
    exec_s = float(rng.uniform(*sc["exec"]))
    crit = [Criticality.life_critical, Criticality.deadline_critical, Criticality.best_effort][int(rng.choice(3, p=sc["crit"]))]
    deadline = exec_s * float(rng.uniform(*sc["slack"])) + 0.2
    if crit == Criticality.best_effort and rng.random() < 0.25:
        deadline += 40.0  # some background jobs are happy to wait
    return TaskRequest(task_id=f"t{i}", exec_time_s=round(exec_s, 2), deadline_s=round(deadline, 2),
                       criticality=crit, state_size_mb=float(rng.choice([0.5, 1.0, 2.0])))


class _Run:
    """Outcome of one task under one policy."""

    __slots__ = ("ok", "wasted_j", "latency_ms", "action", "p")

    def __init__(self, ok, wasted_j, latency_ms, action, p=None):
        self.ok, self.wasted_j, self.latency_ms, self.action, self.p = ok, wasted_j, latency_ms, action, p


def _simple(node: NodeState, T: float, task: TaskRequest, action: str) -> _Run:
    ok = T > task.exec_time_s and node.latency_ms / 1000.0 + task.exec_time_s <= task.deadline_s
    wasted = 0.0 if T > task.exec_time_s else node.consumption_power_w * max(T, 0.0)
    return _Run(ok, wasted, node.latency_ms, action)


def _interfog(rng, sched: Scheduler, store: NodeStore, task: TaskRequest, T: Dict[str, float], sc: dict,
              cfg: Settings, avail: AvailabilityService) -> _Run:
    d = sched.decide(task)
    st = {s.node_id: s for s in store.all()}
    action = d.action.value

    if d.action == Action.cloud_fallback:
        return _Run(rng.random() < cfg.cloud_availability, 0.0, cfg.cloud_latency_ms, action, d.p_complete)

    if d.action == Action.delay:  # wait for harvest, then place once more on the recovered nodes
        later = NodeStore()
        for s in st.values():
            soc = min(1.0, s.battery_soc + s.harvest_power_w * d.retry_after_s / s.capacity_j)
            later.upsert(NodeTelemetry(**{**s.model_dump(), "battery_soc": soc}))
        t2 = task.model_copy(update={"deadline_s": task.deadline_s - d.retry_after_s})
        d2 = Scheduler(avail, later, cfg).decide(t2)
        if d2.action in (Action.delay, Action.cloud_fallback) or d2.target_node_id not in st:
            return _Run(rng.random() < cfg.cloud_availability if d2.action == Action.cloud_fallback else False,
                        0.0, cfg.cloud_latency_ms, action, d.p_complete)
        Tn = _lifetimes(rng, later.all(), sc)
        r = _simple(later.get(d2.target_node_id), Tn[d2.target_node_id], t2, action)
        r.p = d.p_complete
        return r

    if d.action == Action.replicate:
        ids = d.replica_node_ids
        ok = any(T[i] > task.exec_time_s for i in ids) and \
            min(st[i].latency_ms for i in ids) / 1000.0 + task.exec_time_s <= task.deadline_s
        wasted = sum(st[i].consumption_power_w * min(T[i], task.exec_time_s) for i in ids if T[i] <= task.exec_time_s)
        return _Run(ok, wasted, min(st[i].latency_ms for i in ids), action, d.p_complete)

    # execute / offload_peer: run, then check on the task halfway through
    x = d.target_node_id
    node = st[x]
    e = task.exec_time_s * 0.5
    if T[x] <= e:
        return _Run(False, node.consumption_power_w * T[x], node.latency_ms, action, d.p_complete)
    aged = NodeStore()
    for s in st.values():
        soc = max(0.0, s.battery_soc + (s.harvest_power_w - s.consumption_power_w) * e / s.capacity_j) if s.node_id == x else s.battery_soc
        aged.upsert(NodeTelemetry(**{**s.model_dump(), "battery_soc": soc}))
    r2 = Scheduler(avail, aged, cfg).reassess(task, x, e)
    remaining = task.exec_time_s - e
    if r2.action == Action.checkpoint_migrate:
        mig = task.state_size_mb * cfg.migration_s_per_mb
        y = r2.target_node_id
        if T[x] < e + cfg.checkpoint_overhead_s:
            return _Run(False, node.consumption_power_w * T[x], node.latency_ms, "checkpoint_migrate", d.p_complete)
        total = node.latency_ms / 1000.0 + e + cfg.checkpoint_overhead_s + mig + remaining
        ok = T[y] > remaining + mig and total <= task.deadline_s
        wasted = 0.0 if ok else st[y].consumption_power_w * min(T[y], remaining)
        return _Run(ok, wasted, node.latency_ms, "checkpoint_migrate", d.p_complete)
    if r2.action == Action.cloud_fallback:
        ok = T[x] > e + cfg.checkpoint_overhead_s and rng.random() < cfg.cloud_availability
        return _Run(ok, 0.0, node.latency_ms, "cloud_fallback", d.p_complete)
    r = _simple(node, T[x], task, action)
    r.p = d.p_complete
    return r


def _run_policies(avail: AvailabilityService, cfg: Settings, sc_id: str, n: int, seed: int,
                  policies: List[str], cal_avail: Optional[AvailabilityService] = None) -> Dict[str, List[_Run]]:
    sc = SCENARIOS[sc_id]
    rng = np.random.default_rng(seed)
    store = NodeStore()
    for nd in scenario_nodes(sc_id):
        store.upsert(NodeTelemetry(**nd))
    states = store.all()
    sched = Scheduler(avail, store, cfg)
    sched_cal = Scheduler(cal_avail, store, cfg) if cal_avail else None
    res: Dict[str, List[_Run]] = {p: [] for p in policies}
    for i in range(n):
        task = _make_task(rng, sc, i)
        T = _lifetimes(rng, states, sc)
        free = [s for s in states if s.queue_length < cfg.max_queue_length] or states
        for p in policies:
            if p == "latency":
                node = min(free, key=lambda s: s.latency_ms)
                res[p].append(_simple(node, T[node.node_id], task, "execute"))
            elif p == "energy":
                node = max(free, key=lambda s: s.stored_energy_j)
                res[p].append(_simple(node, T[node.node_id], task, "execute"))
            elif p == "random":
                node = free[int(rng.integers(len(free)))]
                res[p].append(_simple(node, T[node.node_id], task, "execute"))
            elif p == "interfog":
                res[p].append(_interfog(rng, sched, store, task, T, sc, cfg, avail))
            elif p == "interfog_calibrated" and sched_cal:
                res[p].append(_interfog(rng, sched_cal, store, task, T, sc, cfg, cal_avail))
    return res


def _summary(runs: List[_Run]) -> dict:
    n = len(runs)
    actions: Dict[str, int] = {}
    for r in runs:
        actions[r.action] = actions.get(r.action, 0) + 1
    return {"completion_rate": round(sum(r.ok for r in runs) / n, 4),
            "wasted_energy_j": round(sum(r.wasted_j for r in runs), 1),
            "mean_latency_ms": round(sum(r.latency_ms for r in runs) / n, 1),
            "actions": actions}


def compare(avail: AvailabilityService, cfg: Settings, sc_id: str = "S1", n: int = 1000, seed: int = 7) -> dict:
    """Run every policy on the same tasks. Also fits a Platt calibrator on a separate training run and shows the
    reliability of INTERFOG's predicted probabilities before and after it."""
    if sc_id not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{sc_id}'. Choose from {sorted(SCENARIOS)}")
    # 1. learn a calibrator from a separate batch of simulated outcomes
    train = _run_policies(avail, cfg, sc_id, 600, seed + 1000, ["interfog"])["interfog"]
    pairs = [(r.p, 1.0 if r.ok else 0.0) for r in train if r.p is not None and r.action != "cloud_fallback"]
    cal = PlattScaler().fit([p for p, _ in pairs], [y for _, y in pairs]) if len(pairs) >= 30 else None
    cal_avail = AvailabilityService(avail.model)
    cal_avail.calibrator = cal
    # 2. fresh evaluation batch
    res = _run_policies(avail, cfg, sc_id, n, seed, POLICIES, cal_avail if cal else None)
    out = {"scenario": {"id": sc_id, "name": SCENARIOS[sc_id]["name"]}, "n_tasks": n, "model": avail.name,
           "policies": {p: _summary(r) for p, r in res.items() if r}}

    def rel(runs):
        pr = [(r.p, 1.0 if r.ok else 0.0) for r in runs if r.p is not None and r.action != "cloud_fallback"]
        return calibration_summary([p for p, _ in pr], [y for _, y in pr])

    out["calibration"] = {"before": rel(res["interfog"]),
                          "after": rel(res["interfog_calibrated"]) if cal else None,
                          "method": "platt" if cal else None}
    return out

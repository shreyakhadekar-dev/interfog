"""Machine learning: survival models, probability calibration, harvest forecasting and the model hub.

Survival models (Weibull AFT, Cox PH, DeepSurv) predict P(node still powered after tau seconds) from node features.
All run on numpy/scipy only, so the deployed image stays small. Calibration then corrects over- or under-confidence,
which matters because the scheduler compares probabilities against hard thresholds.
"""
import math
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from app.core import (AvailabilityService, NodeState, brier_score, expected_calibration_error)

# ============================================================================ features and synthetic data
FEATURES = ["soc", "stored_energy_j", "harvest_w", "consumption_w", "net_drain_w", "cpu_util", "queue_length"]


def node_features(s: NodeState) -> np.ndarray:
    return np.array([s.battery_soc, s.stored_energy_j, s.harvest_power_w, s.consumption_power_w,
                     s.consumption_power_w - s.harvest_power_w, s.cpu_util, float(s.queue_length)], float)


def synthetic_lifetimes(n: int = 1500, seed: int = 42, censor_frac: float = 0.25):
    """Simulate (X, durations, events). Lifetimes are Weibull with a scale that grows with usable energy and shrinks
    with net drain, plus random outages. Some rows are right-censored (the node was still up when observed)."""
    rng = np.random.default_rng(seed)
    cap, soc = rng.uniform(30, 90, n), rng.uniform(0.05, 1.0, n)
    harvest, cons = rng.uniform(0.0, 1.6, n), rng.uniform(0.5, 1.8, n)
    cpu, queue = rng.uniform(0, 1, n), rng.integers(0, 6, n).astype(float)
    stored, drain = soc * cap, cons - harvest
    scale = np.clip(np.where(drain > 0.05, stored / np.maximum(drain, 0.05), 400.0), 2.0, 400.0)
    t = np.minimum(scale * rng.weibull(2.0, n), rng.exponential(1.0 / 2e-3, n))
    cens = rng.uniform(0, 1, n) < censor_frac
    durations = np.where(cens, rng.uniform(0.3, 1.0, n) * t, t)
    X = np.column_stack([soc, stored, harvest, cons, drain, cpu, queue])
    return X, np.maximum(durations, 1e-3), (~cens).astype(float)


def concordance_index(d, risk, e) -> float:
    """Harrell's C: share of comparable pairs where the higher-risk node fails first."""
    d, r, e = map(np.asarray, (d, risk, e))
    num = den = 0.0
    for i in np.where(e == 1)[0]:
        later = d > d[i]
        den += later.sum()
        num += (r[i] > r[later]).sum() + 0.5 * (r[i] == r[later]).sum()
    return float(num / den) if den else float("nan")


# ============================================================================ survival models
class SurvivalModel:
    """Base class. Subclasses satisfy the same interface as app.core.EnergyModel."""

    name = "survival"

    def __init__(self) -> None:
        self.mu = self.sd = None
        self.metrics: dict = {}

    def _fit_scaler(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        return self._z(X)

    def _z(self, X):
        return (X - self.mu) / self.sd

    def survival_matrix(self, X, tau: float):
        raise NotImplementedError

    def risk_score(self, X):
        raise NotImplementedError

    def survival_probability(self, s: NodeState, tau_s: float) -> float:
        if tau_s <= 0:
            return 1.0
        if s.battery_soc <= 0.0:
            return 0.0
        return max(0.0, min(1.0, float(self.survival_matrix(node_features(s)[None, :], tau_s)[0])))

    def expected_lifetime_s(self, s: NodeState) -> float:
        """Median survival time by bisection on S(t)=0.5; inf if S stays above 0.5 for a day."""
        x, hi, lo = node_features(s)[None, :], 86400.0, 1e-3
        if self.survival_matrix(x, hi)[0] > 0.5:
            return math.inf
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self.survival_matrix(x, mid)[0] > 0.5 else (lo, mid)
        return 0.5 * (lo + hi)


class WeibullAFT(SurvivalModel):
    """log T = x.beta + b + sigma*W (W Gumbel-min), fitted by maximum likelihood with right-censoring."""

    name = "weibull-aft"

    def __init__(self, l2: float = 1e-3) -> None:
        super().__init__()
        self.l2, self.beta, self.b, self.log_k = l2, np.zeros(0), 0.0, 0.0

    def _nll(self, th, Z, d, e):
        beta, b, log_k = th[:-2], th[-2], th[-1]
        w = np.exp(log_k) * (np.log(d) - (Z @ beta + b))
        nll = -np.sum(e * (log_k - np.log(d) + w) - np.exp(np.clip(w, -50, 50)))
        return nll / len(d) + self.l2 * np.sum(beta ** 2)

    def fit(self, X, d, e):
        Z = self._fit_scaler(np.asarray(X, float))
        th0 = np.zeros(Z.shape[1] + 2)
        th0[-2] = np.log(np.median(d))
        res = minimize(self._nll, th0, args=(Z, d, e), method="L-BFGS-B")
        self.beta, self.b, self.log_k = res.x[:-2], res.x[-2], res.x[-1]
        self.metrics = {"shape_k": float(np.exp(self.log_k))}
        return self

    def survival_matrix(self, X, tau):
        scale = np.exp(self._z(np.atleast_2d(X)) @ self.beta + self.b)
        return np.exp(-((tau / scale) ** np.exp(self.log_k)))

    def risk_score(self, X):
        return -(self._z(np.atleast_2d(X)) @ self.beta)


def _breslow(risk, d, e):
    """Cumulative baseline hazard at each unique event time."""
    times = np.unique(d[e == 1])
    H, cum = [], 0.0
    for t in times:
        cum += ((d == t) & (e == 1)).sum() / risk[d >= t].sum()
        H.append(cum)
    return times, np.array(H)


def _cum_hazard(times, H, tau):
    if len(times) == 0 or tau < times[0]:
        return 0.0
    h = H[np.searchsorted(times, tau, side="right") - 1]
    if tau > times[-1]:  # extrapolate with the tail hazard rate
        h = H[-1] + H[-1] / times[-1] * (tau - times[-1])
    return float(h)


class CoxPH(SurvivalModel):
    """Cox proportional hazards (Breslow baseline), Newton-Raphson with a ridge penalty."""

    name = "cox-ph"

    def __init__(self, l2: float = 1e-2) -> None:
        super().__init__()
        self.l2, self.beta, self.t, self.H0 = l2, np.zeros(0), np.zeros(0), np.zeros(0)

    @staticmethod
    def _grad_hess(Z, d, e, beta, l2):
        order = np.argsort(-d)
        Zs, es = Z[order], e[order]
        eta = Zs @ beta
        w = np.exp(eta - eta.max())
        S0, S1 = np.cumsum(w), np.cumsum(w[:, None] * Zs, axis=0)
        S2 = np.cumsum(w[:, None, None] * (Zs[:, :, None] * Zs[:, None, :]), axis=0)
        ev = es == 1
        mean = S1[ev] / S0[ev, None]
        grad = (Zs[ev] - mean).sum(0) - l2 * len(d) * beta
        hess = -(S2[ev] / S0[ev, None, None] - mean[:, :, None] * mean[:, None, :]).sum(0) - l2 * len(d) * np.eye(Z.shape[1])
        return grad, hess

    def fit(self, X, d, e):
        Z = self._fit_scaler(np.asarray(X, float))
        beta = np.zeros(Z.shape[1])
        for _ in range(50):
            g, H = self._grad_hess(Z, d, e, beta, self.l2)
            step = np.linalg.solve(H - 1e-9 * np.eye(len(beta)), g)
            beta = beta - step
            if np.max(np.abs(step)) < 1e-7:
                break
        self.beta = beta
        self.t, self.H0 = _breslow(np.exp(Z @ beta), d, e)
        return self

    def survival_matrix(self, X, tau):
        return np.exp(-_cum_hazard(self.t, self.H0, tau) * np.exp(self._z(np.atleast_2d(X)) @ self.beta))

    def risk_score(self, X):
        return self._z(np.atleast_2d(X)) @ self.beta


class DeepSurv(SurvivalModel):
    """Neural Cox model in plain numpy: one tanh hidden layer, manual backprop, Adam. Learns non-linear energy effects."""

    name = "deepsurv"

    def __init__(self, hidden: int = 16, epochs: int = 200, lr: float = 0.01, l2: float = 1e-3, seed: int = 42) -> None:
        super().__init__()
        self.hidden, self.epochs, self.lr, self.l2, self.seed = hidden, epochs, lr, l2, seed
        self.t, self.H0, self.shift = np.zeros(0), np.zeros(0), 0.0

    def _forward(self, Z):
        A = np.tanh(Z @ self.W1 + self.b1)
        return A, (A @ self.W2 + self.b2).ravel()

    def fit(self, X, d, e):
        Z = self._fit_scaler(np.asarray(X, float))
        order = np.argsort(-d)
        Z, d, e = Z[order], d[order], e[order]
        n, p = Z.shape
        rng = np.random.default_rng(self.seed)
        self.W1, self.b1 = rng.normal(0, 1 / np.sqrt(p), (p, self.hidden)), np.zeros(self.hidden)
        self.W2, self.b2 = rng.normal(0, 1 / np.sqrt(self.hidden), (self.hidden, 1)), np.zeros(1)
        params = [self.W1, self.b1, self.W2, self.b2]
        m, v = [np.zeros_like(q) for q in params], [np.zeros_like(q) for q in params]
        n_ev = max(e.sum(), 1.0)
        for ep in range(1, self.epochs + 1):
            A, h = self._forward(Z)
            w = np.exp(h - h.max())
            S0 = np.cumsum(w)
            gh = (-e + w * np.cumsum((e / S0)[::-1])[::-1]) / n_ev
            dA = gh[:, None] @ self.W2.T * (1 - A ** 2)
            grads = [Z.T @ dA + self.l2 * self.W1, dA.sum(0), A.T @ gh[:, None] + self.l2 * self.W2, np.array([gh.sum()])]
            for i, (q, g) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                q -= self.lr * (m[i] / (1 - 0.9 ** ep)) / (np.sqrt(v[i] / (1 - 0.999 ** ep)) + 1e-8)
        _, h = self._forward(Z)
        self.shift = float(h.max())
        self.t, self.H0 = _breslow(np.exp(h - self.shift), d, e)
        return self

    def risk_score(self, X):
        return self._forward(self._z(np.atleast_2d(X)))[1]

    def survival_matrix(self, X, tau):
        return np.exp(-_cum_hazard(self.t, self.H0, tau) * np.exp(self.risk_score(X) - self.shift))


MODEL_CLASSES = {"weibull-aft": WeibullAFT, "cox-ph": CoxPH, "deepsurv": DeepSurv}
EVAL_TAUS = (4.0, 8.0, 16.0, 32.0)


def evaluate(model: SurvivalModel, X, d, e) -> dict:
    """Held-out C-index plus complete-case Brier score and ECE of S(tau) at several horizons."""
    out: dict = {"c_index": round(concordance_index(d, model.risk_score(X), e), 4), "horizons": {}}
    briers = []
    for tau in EVAL_TAUS:
        known = (d > tau) | (e == 1)  # status at tau is known
        if known.sum() < 10:
            continue
        p, y = model.survival_matrix(X[known], tau), (d[known] > tau).astype(float)
        b = brier_score(p, y)
        briers.append(b)
        out["horizons"][str(int(tau))] = {"brier": round(b, 4), "ece": round(expected_calibration_error(p, y), 4)}
    out["mean_brier"] = round(float(np.mean(briers)), 4) if briers else None
    return out


def train_all(n_samples: int = 1500, seed: int = 42) -> Tuple[Dict[str, SurvivalModel], dict]:
    """Fit each model on a train split, report held-out metrics, then refit on everything for serving."""
    X, d, e = synthetic_lifetimes(n_samples, seed=seed)
    idx = np.random.default_rng(seed).permutation(len(d))
    cut = int(len(d) * 0.75)
    a, b = idx[:cut], idx[cut:]
    report = {"n_train": int(cut), "n_val": int(len(d) - cut), "censored_frac": round(float(1 - e.mean()), 3), "models": {}}
    fitted: Dict[str, SurvivalModel] = {}
    for name, cls in MODEL_CLASSES.items():
        kw = {"seed": seed} if name == "deepsurv" else {}
        report["models"][name] = evaluate(cls(**kw).fit(X[a], d[a], e[a]), X[b], d[b], e[b])
        fitted[name] = cls(**kw).fit(X, d, e)
    scores = {n: m["mean_brier"] for n, m in report["models"].items() if m["mean_brier"] is not None}
    report["best"] = min(scores, key=scores.get) if scores else None
    return fitted, report


# ============================================================================ calibration
EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def _nll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class Calibrator:
    name = "identity"

    def fit(self, p, y):
        return self

    def transform(self, p):
        return np.asarray(p, float)

    def __call__(self, p: float) -> float:
        return float(self.transform(np.array([p]))[0])


class PlattScaler(Calibrator):
    """p' = sigmoid(a*logit(p) + b)."""

    name = "platt"

    def __init__(self):
        self.a, self.b = 1.0, 0.0

    def fit(self, p, y):
        z, y = _logit(p), np.asarray(y, float)
        r = minimize(lambda t: _nll(_sigmoid(t[0] * z + t[1]), y) + 1e-4 * (t[0] - 1) ** 2, [1.0, 0.0], method="L-BFGS-B")
        self.a, self.b = float(r.x[0]), float(r.x[1])
        return self

    def transform(self, p):
        return _sigmoid(self.a * _logit(p) + self.b)


class TemperatureScaler(Calibrator):
    """p' = sigmoid(logit(p)/T). One parameter, safe with few outcomes."""

    name = "temperature"

    def __init__(self):
        self.T = 1.0

    def fit(self, p, y):
        z, y = _logit(p), np.asarray(y, float)
        r = minimize_scalar(lambda lt: _nll(_sigmoid(z / np.exp(lt)), y), bounds=(-3, 3), method="bounded")
        self.T = float(np.exp(r.x))
        return self

    def transform(self, p):
        return _sigmoid(_logit(p) / self.T)


class IsotonicCalibrator(Calibrator):
    """Monotone non-parametric calibration (pool adjacent violators). Needs a few hundred outcomes."""

    name = "isotonic"

    def __init__(self):
        self.x, self.y = np.array([0.0, 1.0]), np.array([0.0, 1.0])

    def fit(self, p, y):
        p, y = np.asarray(p, float), np.asarray(y, float)
        order = np.argsort(p, kind="stable")
        blocks: List[list] = []
        for xv, yv in zip(p[order], y[order]):
            blocks.append([yv, 1.0, xv, xv])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
                s, w, _, hi = blocks.pop()
                blocks[-1][0] += s
                blocks[-1][1] += w
                blocks[-1][3] = hi
        self.x = np.array([(b[2] + b[3]) / 2 for b in blocks])
        self.y = np.array([b[0] / b[1] for b in blocks])
        return self

    def transform(self, p):
        return np.interp(np.asarray(p, float), self.x, self.y)


CALIBRATORS = {"platt": PlattScaler, "isotonic": IsotonicCalibrator, "temperature": TemperatureScaler}


def make_calibrator(method: str) -> Calibrator:
    if method not in CALIBRATORS:
        raise ValueError(f"Unknown calibration method '{method}'. Choose from {sorted(CALIBRATORS)}")
    return CALIBRATORS[method]()


# ============================================================================ forecasting
class _Ewma:
    name = "ewma"

    def fit(self, y):
        level = float(y[0])
        for v in y[1:]:
            level = 0.3 * float(v) + 0.7 * level
        self.level = level
        return self

    def predict(self, k):
        return np.full(k, max(self.level, 0.0))


class _Holt:
    """Double exponential smoothing with a damped trend."""

    name = "holt"

    def fit(self, y):
        a, b, self.phi = 0.4, 0.15, 0.9
        level, trend = float(y[0]), (float(y[1] - y[0]) if len(y) > 1 else 0.0)
        for v in y[1:]:
            prev = level
            level = a * float(v) + (1 - a) * (prev + self.phi * trend)
            trend = b * (level - prev) + (1 - b) * self.phi * trend
        self.level, self.trend = level, trend
        return self

    def predict(self, k):
        return np.maximum(self.level + np.cumsum(self.phi ** np.arange(1, k + 1)) * self.trend, 0.0)


class _AR:
    """Autoregressive model of order p by ridge least squares; captures periodic structure."""

    name = "ar"

    def fit(self, y):
        y = np.asarray(y, float)
        p = min(6, max(1, len(y) // 3))
        self.p = p
        if len(y) <= p + 2:
            self.coef, self.bias, self.tail = np.zeros(p), float(y.mean()), np.resize(y, p)
            return self
        X = np.column_stack([y[i:len(y) - p + i] for i in range(p)] + [np.ones(len(y) - p)])
        w = np.linalg.solve(X.T @ X + 1e-3 * np.eye(p + 1), X.T @ y[p:])
        self.coef, self.bias, self.tail = w[:p], float(w[p]), y[-p:].copy()
        return self

    def predict(self, k):
        hist, out = list(self.tail), []
        for _ in range(k):
            nxt = max(float(np.dot(self.coef, hist[-self.p:]) + self.bias), 0.0)
            out.append(nxt)
            hist.append(nxt)
        return np.array(out)


_FORECASTERS = {"ewma": _Ewma, "holt": _Holt, "ar": _AR}


def auto_forecast(series: Sequence[float], steps: int, horizon: int = 5, folds: int = 4) -> dict:
    """Pick the forecaster with the lowest rolling-origin MAE on this node's own series, refit it, predict `steps`."""
    y, n = np.asarray(series, float), len(series)
    if n < 12:
        return {"method": "ewma", "path": _Ewma().fit(y).predict(steps).tolist(), "mae": {}}
    origins = np.linspace(max(8, n // 2), n - horizon, folds).astype(int)
    scores = {k: float(np.mean([np.mean(np.abs(c().fit(y[:o]).predict(horizon) - y[o:o + horizon])) for o in origins]))
              for k, c in _FORECASTERS.items()}
    best = min(scores, key=scores.get)
    return {"method": best, "path": _FORECASTERS[best]().fit(y).predict(steps).tolist(), "mae": scores}


# ============================================================================ model hub
class ModelHub:
    """Owns the trained models and decides which one the availability service uses."""

    def __init__(self, avail: AvailabilityService) -> None:
        self.avail = avail
        self.heuristic = avail.model
        self.trained: Dict[str, SurvivalModel] = {}
        self.report: Optional[dict] = None
        self.calibration: Optional[dict] = None
        self._lock = threading.Lock()

    def train(self, n_samples: int = 1500, seed: int = 42, activate: Optional[str] = "best") -> None:
        fitted, report = train_all(n_samples, seed)
        with self._lock:
            self.trained, self.report = fitted, report
        if activate:
            self.activate(report["best"] if activate == "best" else activate)

    def activate(self, name: str) -> None:
        if name in ("heuristic", self.heuristic.name):
            self.avail.model = self.heuristic
        elif name in self.trained:
            self.avail.model = self.trained[name]
        else:
            raise ValueError(f"Model '{name}' is not trained yet. Train first or choose 'heuristic'.")
        self.avail.calibrator = None
        self.calibration = None

    def calibrate(self, p: Sequence[float], y: Sequence[float], method: str = "platt", min_outcomes: int = 30) -> dict:
        if len(p) < min_outcomes:
            raise ValueError(f"Need at least {min_outcomes} recorded outcomes to calibrate, have {len(p)}.")
        cal = make_calibrator(method).fit(p, y)
        before = expected_calibration_error(p, y)
        after = expected_calibration_error(cal.transform(p), y)
        self.avail.calibrator = cal
        self.calibration = {"method": method, "n": len(p), "ece_before": before, "ece_after": after}
        return self.calibration

    def clear_calibration(self) -> None:
        self.avail.calibrator = None
        self.calibration = None

    def status(self) -> dict:
        return {"active": self.avail.name, "available": ["heuristic"] + sorted(self.trained),
                "trained": bool(self.trained), "report": self.report, "calibration": self.calibration}

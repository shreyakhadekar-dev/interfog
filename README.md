<h1 align="center">INTERFOG</h1>
<p align="center"><b>Will the task finish before the battery runs dry?</b><br>
Probability-of-completion scheduling for fog nodes powered by harvested energy.</p>

![INTERFOG dashboard](docs/dashboard.png)

Fog nodes that run on sunlight, vibration or radio waves lose power without warning. INTERFOG estimates the chance that
each node stays on long enough to finish a task, then picks one of six actions: **execute**, **delay**,
**checkpoint and migrate**, **replicate**, **offload to a peer** or **fall back to the cloud**.

One small Python service gives you the REST API, a live WebSocket feed and the dashboard shown above.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open <http://localhost:8000> for the dashboard and <http://localhost:8000/docs> for the interactive API.
Four demo nodes start already running, so there is nothing to set up.

## Deploy on Render

1. Push this folder to a GitHub repository (the files go at the repo root).
2. In Render choose **New → Blueprint**, select the repository and click **Apply**. `render.yaml` does the rest.
3. Wait for the build, then open the `onrender.com` link. Health check: `/health`.

No Blueprint? Create a **Web Service** with build command `pip install -r requirements.txt` and start command
`uvicorn app.main:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips '*'`.
State is kept in memory, so it resets when the free instance sleeps. That is fine for a demo.

## What is inside

| Part | What it does |
|---|---|
| Dashboard (`app/static/index.html`) | Live battery tubes, survival curves, scenario simulator, model switcher, decision feed |
| Availability | Energy-balance heuristic, or Weibull AFT, Cox PH and DeepSurv trained with one click |
| Scheduler | Scores nodes with `α·P − β·L − γ·E − δ·M − ε·C − ζ·V` and applies the six actions |
| Calibration | Platt, temperature and isotonic scaling so "80%" really means 80% |
| Forecasting | EWMA, damped Holt or autoregressive harvest forecasts, chosen per node |
| Simulator | Scenarios S1 to S7, five strategies, same tasks and hidden battery luck for each |

## API at a glance

| Call | Purpose |
|---|---|
| `POST /api/v1/telemetry` | Report a node's battery, harvest and load |
| `GET /api/v1/availability?tau_s=8` | Chance each node survives 8 seconds |
| `POST /api/v1/tasks` | Place a task, get an action and the reasoning |
| `POST /api/v1/tasks/reassess` | Check a running task and decide whether to checkpoint and move |
| `POST /api/v1/outcomes` | Report whether a task finished, for calibration |
| `POST /api/v1/models/train` · `/activate` · `/calibrate` | Manage the survival models |
| `POST /api/v1/simulate` | Compare strategies on scenario S1 to S7 |
| `WS /ws` | Live node updates and decisions |

Example:

```bash
curl -X POST localhost:8000/api/v1/tasks -H 'content-type: application/json' \
  -d '{"exec_time_s": 8, "deadline_s": 25, "criticality": "deadline_critical"}'
```

## Settings

Set as environment variables (all optional): `ENABLE_DEMO` (default `true`), `CORS_ORIGINS`, the objective weights
`ALPHA BETA GAMMA DELTA EPSILON ZETA`, the probability bars `P_MIN_LIFE_CRITICAL`, `P_MIN_DEADLINE_CRITICAL`,
`P_MIN_BEST_EFFORT`, plus `CLOUD_ENABLED`, `CLOUD_LATENCY_MS`, `DELAY_WINDOW_S`. See `app/core.py` for the full list.

## Tests

```bash
pip install pytest httpx
python -m pytest -q
```

## Layout

```
app/main.py        API, WebSocket, dashboard hosting
app/core.py        data models, node store, availability, scheduler, metrics
app/ml.py          survival models, calibration, forecasting, model hub
app/simulate.py    scenarios S1-S7 and the strategy comparison
app/static/        the dashboard (one HTML file, no build step)
tests/             24 tests
docs/              architecture notes, research proposal, review deck
```

More detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Released under the MIT licence.

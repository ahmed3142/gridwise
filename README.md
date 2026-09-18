# GridWise LLM: LLM-assisted campus energy scheduling

This is our entry for the **BUP CSE Fest 2026 Hackathon online preliminary** ("Smart Campus Energy
Optimization Challenge: LLM-Assisted Operator Directive Interpretation").

The service reads a 24-hour campus energy scenario and 1 to 3 natural-language operator notes. An
**LLM** turns each note into a machine-checkable directive. **Deterministic guardrails** validate it.
An **exact linear-programming optimizer** then returns the minimum-cost 24-hour grid, solar and
battery schedule that obeys every directive and every GridWise energy rule.

```
 request ──► schema checks ──► LLM interpreter ──► normalizer + guardrails ──► LP optimizer ──► replay validator ──► response
             (400 / 422)       (OpenAI, one call    (units → numbers,           (HiGHS; cost,    (judge-style
                                per note, parallel)  hours, section 08 rules)    then peak,       self-check of
                                                                                 then throughput) every response)
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Readiness. Returns `{"status":"ok"}` without waiting for the LLM provider. |
| `POST /optimize-energy` | The main endpoint. Implements the request and response contract of Problem Statement sections 07 and 10. |
| `GET /version` | Build, prompt version, the resolved models, and live LLM health counters. It never shows secrets. |
| `GET /docs` | Interactive OpenAPI documentation. |

---

## 1. Quickstart (local, about 3 minutes)

Requires Python 3.12.

**Linux / macOS (bash or zsh):**

```bash
git clone <this-repo-url> gridwise && cd gridwise
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
cp .env.example .env                  # then edit .env and set OPENAI_API_KEY=sk-...
.venv/bin/python -m app               # serves on http://127.0.0.1:8080
```

In a second terminal:

```bash
curl http://127.0.0.1:8080/health     # {"status":"ok"}
.venv/bin/python scripts/make_request.py SAMPLE-06 > sample.json
curl -X POST http://127.0.0.1:8080/optimize-energy -H "Content-Type: application/json" --data-binary @sample.json
```

**Windows (PowerShell):**

```powershell
git clone <this-repo-url> gridwise; cd gridwise
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env           # then edit .env and set OPENAI_API_KEY=sk-...
.venv\Scripts\python -m app
```

In a second PowerShell window:

```powershell
curl.exe http://127.0.0.1:8080/health
.venv\Scripts\python scripts\make_request.py SAMPLE-06 | Out-File -Encoding ascii sample.json
curl.exe -X POST http://127.0.0.1:8080/optimize-energy -H "Content-Type: application/json" --data-binary "@sample.json"
```

Every `python scripts/...` command below means the venv's interpreter
(`.venv/bin/python` or `.venv\Scripts\python`).

### Run the public sample pack like the judge does

```bash
python scripts/judge.py --url http://127.0.0.1:8080
```

**Expected result:** `GET /health -> 200 {'status': 'ok'}`, then `cases passed: 10/10`, `non-200: 0`
and `optimization quality (mean ratio): 1.0000`. For every case, relevance, type, hours and values
must all match, and `valid` must be `True`.

The script replays each `hourly_plan` against the **organizer ground-truth directives** and every
GridWise rule. It also compares cost with the reference optimum and prints p50/p95 latency. Add
`--repeat 3 --concurrency 4` for a stability and latency run. Point `--url` at the deployed service
to test it from outside.

> On Windows, prefer `127.0.0.1` over `localhost`. Windows tries IPv6 first for `localhost`, which
> adds about 2 s to every client request.

### Offline test suite (no API key needed)

```bash
.venv/bin/python -m pytest -q
```

The suite has 160+ tests. They cover the 10 public cases end to end through a mock LLM: exact
interpretation, a valid plan under the ground truth, and the reference optimal cost. They also
cover every 400/422 path, extra-field tolerance, provider failures of every kind, the re-ask
behaviour, the cache, infeasibility repair, 40 randomized scenarios, and each validator rule.

### Measured results (2026-09-18, real OpenAI key)

| Check | Result |
|---|---|
| Public sample pack against the live Railway URL (`scripts/judge.py`) | **10/10 cases pass**, p95 2.2 s: every interpretation field correct, plans valid under the ground truth, cost quality 1.0000 |
| Same, with the Docker Hub image pulled by digest | 10/10, `/health` ready in 5 s |
| 65 paraphrased notes (`scripts/eval_interpretation.py`) | **65/65 correct** with the primary `gpt-5.4-mini` and with the fallback `gpt-4.1` (also `gpt-5.6-luna`, `gpt-4.1-mini` and `gpt-5.6-terra`) |
| 24 unique 3-note requests, 8 concurrent | p50 1.9–2.1 s, p95 2.1–2.8 s, 0 non-200 |
| Provider failures (invalid key, missing model, timeouts) | Controlled responses in 1.2–3.0 s. A missing primary switches to the fallback. The key is redacted in logs. |

### Live LLM checks (needs `OPENAI_API_KEY`)

```bash
python scripts/probe_llm.py                 # which models the key can use, latency and accuracy per model
python scripts/eval_interpretation.py       # 65 paraphrased notes: per-field accuracy (relevance/type/hours/values)
```

---

## 2. Docker fallback image

```bash
docker pull ahmed3142/gridwise-llm:1.0.0
# the same image pinned by digest: ahmed3142/gridwise-llm@sha256:ec5bc94754adc496e302ce32688d756161c208ca50c964ef897bbc093d9d62bf
docker run --rm -p 8080:8080 -e OPENAI_API_KEY=sk-... ahmed3142/gridwise-llm:1.0.0
curl http://127.0.0.1:8080/health
```

* The image binds `0.0.0.0:$PORT` (default `8080`), runs as a non-root user, has a `HEALTHCHECK`,
  and contains **no secrets**. `.env` is excluded by `.dockerignore`.
* Without `OPENAI_API_KEY` the container still starts and `/health` returns `ok`. Notes are then
  answered as controlled `no_op` entries and the response carries the header
  `X-GridWise-Degraded: true`.
* Build it yourself: `docker build -t gridwise-llm:1.0.0 .`. For registries use
  `docker buildx build --platform linux/amd64 -t ahmed3142/gridwise-llm:1.0.0 --push .`.

## 3. Deployment

**Fly.io** (always on; `fly.toml` included):

```bash
fly launch --no-deploy --copy-config        # choose a unique app name
fly secrets set OPENAI_API_KEY=sk-...
fly deploy
python scripts/judge.py --url https://<app>.fly.dev
```

**Railway** (`railway.json` included, this is where the live service runs:
`https://gridwise-api-production.up.railway.app`). Deploy with the Railway CLI:

```bash
railway login
railway init --name gridwise
railway add --service gridwise-api --variables "OPENAI_MODEL=gpt-5.4-mini" --variables "OPENAI_FALLBACK_MODEL=gpt-4.1"
railway variable set OPENAI_API_KEY --stdin --service gridwise-api   # paste the key, never commit it
railway up --service gridwise-api --detach
railway domain --service gridwise-api
```

Railway injects `PORT`, and the health check path is `/health`. Railway redirects `http://` to
`https://` with a 301, so always use the https URL.

Both configurations keep one instance always running. That matters because the judge needs
`/health` to answer within 60 s and the rubric scores p95 latency, so cold starts must be avoided.

---

## 4. Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | *(none)* | **Required for LLM interpretation.** Keep it in `.env` or platform secrets only. |
| `OPENAI_MODEL` | `auto` | `auto` picks the first model in `OPENAI_MODEL_PREFERENCE` that the key can access (measured best: `gpt-5.4-mini`). Otherwise it is an explicit model id. |
| `OPENAI_FALLBACK_MODEL` | *(next accessible)* | Used when the primary fails or times out. |
| `OPENAI_MODEL_PREFERENCE` | `gpt-5.4-mini,gpt-4.1,gpt-5.6-luna,gpt-4.1-mini,gpt-5-mini,gpt-4o-mini` | Order used by `auto`. |
| `OPENAI_REASONING_EFFORT` | `low` | Applies to reasoning models only. It is dropped automatically when a model rejects it. |
| `OPENAI_BASE_URL` | *(api.openai.com)* | Any OpenAI-compatible endpoint, for example a backup or local model. |
| `LLM_TIMEOUT_SECONDS` | `9` | Cap for a single LLM attempt. |
| `REQUEST_DEADLINE_SECONDS` | `24` | End-to-end budget per request. The judge limit is 30 s. |
| `LLM_MAX_CONCURRENCY` | `48` | Maximum simultaneous LLM calls, which protects rate limits. |
| `INTERPRETATION_CACHE_SIZE` | `2048` | Number of cached LLM answers. Repeated notes cost no LLM call. |
| `LLM_MAX_OUTPUT_TOKENS` | `2500` | Output cap per LLM call. |
| `MAX_BODY_BYTES` | `1000000` | Largest accepted request body. Larger bodies get a 400 and are never buffered. |
| `WEB_CONCURRENCY` | `1` | Uvicorn worker processes. |
| `PORT` / `HOST` | `8080` / `0.0.0.0` | Bind address. |
| `LOG_LEVEL` | `INFO` | Log verbosity. |

---

## 5. How it works

### 5.1 The LLM's role (mandatory interpretation path)

For every note, the service makes one **OpenAI Structured Outputs** call
(`chat.completions.parse(response_format=SemanticInterpretation)`). The calls for different notes
run in parallel. The model sees all notes of the request for context, so "during the same window"
references resolve, and it returns:

```json
{"time_evidence": "from noon until 2 PM", "quantity_evidence": "roughly 25% of the forecast",
 "applies_to_schedule": true, "directive_type": "solar_reduction",
 "time_windows": [{"start_hour": 12, "start_minute": 0, "end_hour": 14, "end_minute": 0}],
 "quantity_value": 25, "quantity_unit": "percent_remaining", "explanation": "..."}
```

Constrained decoding against this JSON schema means the model **cannot** emit an unsupported
directive type or a malformed object. The LLM decides **relevance, directive type, time windows,
and the quantity with its meaning**. Deterministic code then applies the fixed conventions:

* the start-inclusive, end-exclusive hour list, including windows that wrap past midnight;
* `percent_reduction` 80 becomes factor 0.2, and `percent_of_capacity` 50 on a 200 kWh battery becomes 100 kWh;
* MWh becomes kWh.

This is the "LLM extracts, code computes" pattern. It removes the classic LLM off-by-one and
percentage errors without taking any interpretation away from the model. The prompt contains 16
original examples. None of the public sample notes is copied.

### 5.2 Guardrails (Problem Statement section 08)

Before anything reaches the optimizer, the service enforces these checks:

* The type is one of the six allowed values.
* There is exactly one entry per note, in `note_index` order.
* Hours are unique, ascending integers from 0 to 23, and the list is not empty.
* The factor is within [0, 1].
* The reserve is finite, not negative, and not above capacity.
* The grid cap is finite and not negative.
* `no_op` implies `applies=false` and `structured_adjustment=null`, and every other type implies `applies=true`.
* Nothing is invented: base demand, solar, tariff and battery data are never changed.

**Deterministic cross-check.** A second channel compares the model's answer with the literal note.
It flags quoted evidence that is not in the note, numbers that appear nowhere in the note, and
window boundaries that match no written time, such as the typical "end hour minus one" slip. Any
finding triggers **one re-ask** of the LLM, with the finding as feedback. The LLM's second answer
is final. The checker never interprets a note itself.

**Safe failure.** The service handles each failure mode in a controlled way:

* **Timeout or 5xx:** it retries once, then switches to the fallback model.
* **Auth, quota, refusal or unknown model:** it switches to the fallback model.
* **Every attempt fails:** the note becomes a controlled `no_op`. The response still carries a
  valid plan and the header `X-GridWise-Degraded: true`.

A global deadline keeps every request under 30 s. Unhandled errors return a JSON 500 with no stack
trace.

### 5.3 Optimizer (Problem Statement sections 05.3 and 09)

This is an exact LP over 24 × (grid, solar used, charge, discharge, battery energy), solved with
`scipy.optimize.linprog(method="highs")` in about 10 ms. It enforces:

* energy balance;
* the battery transition and end-of-day neutrality (`E_23 = initial`);
* `max(base minimum, reserve)` ≤ E ≤ capacity;
* the charge and discharge rate limits, set to 0 inside no-charge and no-discharge windows;
* solar used ≤ solar × factor;
* grid ≤ cap inside max-grid windows.

The objective is **lexicographic**:

1. **Minimum grid cost.** This is the scored objective, and the public cases match the reference optimum exactly.
2. **Minimum peak import.**
3. **Minimum battery throughput.** As a result, the battery never charges and discharges in the same hour.

**Conservative choices keep plans valid under any reading of the ground truth.** When two
directives of the same type overlap, the service multiplies the solar factors, takes the larger
reserve and the smaller cap, and takes the union of the windows. For non-round values such as "a
third" (0.3333), the optimizer rounds the factor and cap *down* and the reserve *up* to 2 decimals.

**Infeasibility repair.** Judge scenarios are guaranteed feasible, so an infeasible directive set
means a misread note. The service finds the note or notes whose removal restores feasibility and
re-asks the LLM about exactly those notes, citing the reason. Only if that also fails does it return
a least-violation plan, flagged in `plan_summary`.

### 5.4 Output and self-validation

Battery flows are netted and rounded to 1e-6, then repaired so that the day ends *exactly* at the
initial energy. `grid_kwh` is recomputed from the balance equation after rounding, and totals are
computed from the rounded rows. Every response is then replayed by `app/validator.py`, an
independent judge-style checker, before it is returned. `scripts/judge.py` uses the same validator.

### 5.5 HTTP status codes

| Code | When |
|---|---|
| 200 | The request succeeded. This includes provider failures, which are degraded but controlled. |
| 400 | Malformed JSON, a body that is not an object, missing or ill-typed fields, not exactly 24 unique hours, 0 or more than 3 notes, blank notes, negative, NaN or infinite numbers, or numeric strings. |
| 422 | The request is well-formed but physically inconsistent: minimum above capacity, or initial energy outside [minimum, capacity]. |
| 500 | A controlled internal error. The body is JSON and contains no trace. |

Unknown extra JSON fields are **ignored**, never rejected. Hours may arrive in any order.

---

## 6. Repository layout

```
app/
  main.py               FastAPI app, HTTP policy, middleware, error handling
  pipeline.py           interpreter → guardrails → optimizer → validator, infeasibility repair
  schemas.py            request/response contracts (sections 07/10)
  directives.py         Directive model + section-08 guardrails
  validator.py          independent judge-style replay (also used by tests and scripts)
  interpreter/
    semantic.py         LLM output schema (Structured Outputs)
    prompt.py           system prompt + original few-shot examples
    llm.py              OpenAI client: parameter adaptation, error mapping, model auto-selection
    normalize.py        clock windows → hours, quantity units → factor/kWh
    crosscheck.py       deterministic disagreement detector (triggers one re-ask)
    service.py          parallel per-note calls, deadline, re-ask, fallback, cache, single-flight
  optimizer/
    lp.py               LP model, lexicographic solve, relaxed fallback, culprit finder
    plan.py             rounding, neutrality repair, totals, plan_summary
scripts/                judge.py, probe_llm.py, eval_interpretation.py, make_request.py
tests/                  pytest suite + data (public pack, mock answers, 65 paraphrases)
Dockerfile, fly.toml, railway.json, requirements*.txt, requirements.lock, .env.example
```

## 7. Security and secrets

* No keys are stored in the repository or the image. `.env` is ignored by both git and Docker, and
  `.env.example` contains only empty values.
* Logs contain the method, path, status, timing, request id, error types and short provider error messages, with the configured key and anything key-shaped redacted. They never
  contain request bodies, prompts, keys or stack traces. Provider error text is scrubbed of
  anything that looks like a key.
* Before making the repository public, run
  `git log -p --all | grep -nE "sk-[A-Za-z0-9_-]{20,}"`. It must print nothing. Rotate the key after the
  evaluation window.

## 8. Known limitations

* Interpretation quality depends on the LLM. Paraphrase accuracy is measured with
  `scripts/eval_interpretation.py`, but hidden wordings can still surprise it.
* A note is mapped to exactly one directive type, as the spec requires. A note that states two
  constraints yields its main one.
* Half-hour windows are widened to whole hours: 13:30 to 15:00 becomes [13, 14], which is the
  conservative choice.
* If the provider is unreachable for every attempt, affected notes degrade to `no_op`. The service
  stays up and flags the response.
* Single-instance deployment. The in-memory cache is per instance and is lost on restart.

## 9. Credits

* **Team work:** architecture, prompt design, optimizer and guardrails.
* **AI coding assistant:** Claude Code by Anthropic assisted with implementation, tests and
  documentation, as permitted by the official rulebook.
* **LLM provider:** OpenAI API, for operator-note interpretation at runtime.
* **Libraries:** FastAPI, Starlette, Uvicorn, Pydantic, OpenAI Python SDK, NumPy, SciPy (with the
  HiGHS solver), python-dotenv and pytest.
* **Data:** only the synthetic challenge data supplied by the organizers.

# Hardening notes

This file lists every significant change made while hardening the service, with its reason and the
evidence that it works, so the team can explain and defend it. The architecture was not replaced.
The flow is still LLM interpreter, then guardrails, then exact LP optimizer, then replay validator,
and the LLM remains the only interpreter of operator notes.

## 1. Gap report (rubric line → status → evidence)

| Rubric line | Pts | Status | Evidence (reproducible) |
|---|---|---|---|
| Relevance / no_op | 5 | pass | Trap catalogue (`scripts/eval_notes.py`, 53 notes: 10 no_op-discipline cases plus prompt injection) scored 53/53 in 3 of 3 runs, with identical answers. The 5 hidden simulations scored 100%. |
| directive_type | 5 | pass | Same runs. 65/65 paraphrases (`scripts/eval_interpretation.py`) on both the primary and the fallback model. |
| Hours | 5 | pass | Same runs. They cover wraparound (22:00–02:00 gives [0,1,22,23]), "until midnight", "during the 2 PM hour", "after 8 PM", "before 6 AM", "1300 to 1500 hrs", and "from one until three" for solar. |
| Numeric values and shape | 5 | pass | Same runs. "Drop to" vs "drop by", "four-fifths", "a third" (0.3333), "offline" (0), 50% of capacity (100 kWh), 0.15 MWh (150). Exact adjustment keys are checked by `app/validator.py`. |
| Paraphrase robustness | 5 | pass | About 490 labelled notes in total: 18 public, 65 paraphrases, 53 trap, and about 427 hidden-simulation notes over 5 seeds (`scripts/hidden_sim.py`). All 100% on the live URL. |
| Ground-truth application | 10 | pass | The judge-style replay against organizer ground truth passes the 10 public cases and 190/190 hidden-simulation plans on the live URL. The one HTTP 502 was an edge error, not a plan error (section 4). |
| Balance and effective solar | 5 | pass | About 10,000 fuzzed scenarios gave 0 invalid plans (independent review). There are also 40 randomized pytest scenarios. |
| Battery transitions, bounds, rates | 5 | pass | Same. The energy trajectory is rounded per hour, so rounding error never accumulates, and reported energy is never below 0 or above capacity. |
| Action consistency, neutrality, non-negatives | 5 | pass | Netting guarantees a single action per hour; `|net| < 1e-6` is idle with 0 kWh. The final energy is exactly the initial energy. Covered by regression tests. |
| Optimization quality | 10 | pass | Exact lexicographic LP (HiGHS). Public costs equal the references to within 1e-6. Quality is 1.0000 on every hidden-simulation run. |
| Endpoints and status | 2 | pass | Live: `/health` 200; 404 and 405 are JSON. |
| Request validation | 2 | pass | Live malformed-input matrix returns 400 for all of it (empty, invalid JSON, array, null, missing fields, NaN). 26 bad-input variants and a physical-inconsistency test (422) in pytest. |
| Interpretation schema, order, types | 3 | pass | `interpretation_violations()` runs in every response and in the tests. |
| hourly_plan, top level, scenario_id echo | 3 | pass | Scrambled unicode `scenario_id` plus shuffled `hours`: 10/10 on the live URL, with the id echoed byte for byte. |
| Health readiness | 2 | pass | `/health` never waits for the LLM and answers about 4–5 s after a container starts (Docker test). |
| p95 latency | 3 | pass | Live: public pack p95 2.1–2.8 s; hidden simulation at 6–8 concurrent requests p95 2.5–3.7 s; cached repeats p95 0.7 s. A 4-minute keep-warm call avoids slow first requests after idle. |
| Stability | 3 | pass (see risk 3) | About 1,000 live LLM calls with 0 failures (`/version` counters). One HTTP 502 was seen 10 s after a redeploy; it never reached the app. |
| Controlled failures and secrets | 2 | pass | Chaos tests with the real SDK: an invalid key, a missing model and forced timeouts each return a controlled 200 in 1.2–3.0 s. The key is redacted in logs and never in `repr`. The history scan finds 0 key-shaped strings. |
| Live reachability | 3 | pass | Railway, always on. |
| Pullable image reaching /health | 4 | risk until the image is public | `docker.io/ahmed3142/gridwise-llm:1.0.0` pulled by digest and scored 10/10. The image is multi-arch (amd64 and arm64). **It must be switched to public after the deadline.** |
| Clean startup | 2 | pass | The container starts with no key; `/health` answers and notes degrade to no_op. |
| No judge debugging | 1 | pass | The same documented command works on the image and the live service. |
| Quickstart | 3 | pass | README gives bash and PowerShell blocks. |
| Env, config, model docs | 2 | pass | README section 4, pinned models, `.env.example`. |
| Public-sample procedure and expected result | 2 | pass | `scripts/judge.py`, with its expected output in the README. |
| Architecture, Docker, deps, limitations, secrets | 3 | pass | README sections 2, 5, 7, 8, 9 and a Mermaid diagram. |

## 2. Changes, with the reason for each

| Change | Reason | Evidence |
|---|---|---|
| Parameter adaptation made safe under concurrency, plus known model capabilities | Parallel first calls to a new model all failed when it rejected `temperature` or `reasoning_effort`, and notes silently became no_op. Paraphrase accuracy was 59–62/65. | 65/65 afterwards; regression test `test_param_adaptation_is_concurrency_safe`. |
| Energy-trajectory rounding in `plan.py` | Rounding each flow separately let about 1e-6 of drift accumulate: negative battery energy in about 0.3% of scenarios, and an int64 overflow with huge rates. | About 10k fuzzed plans, 0 invalid. |
| Relaxed plans built against the relaxed limits | A least-violation plan could break physical rules. | 327 relaxed plans, 0 physical violations. |
| Hard wall-clock cap on queueing and on each LLM call; timeout goes straight to the fallback model | A slow provider plus queueing could push requests past 30 s. | `test_slow_llm_is_cut_by_the_deadline`. |
| Single-flight followers survive a cancelled or failed leader | One cancelled request could hand no_op to identical concurrent requests. | Reviewer's single-flight script. |
| Infeasibility re-ask shows the previous answer and is not cached across scenarios | A scenario-specific re-ask could poison the shared cache. | `test_reinterpretation_is_not_cached_across_scenarios`. |
| Cross-check false positives removed ("afternoon" contains "noon", named periods, "fully charged", minutes, "3/4", number words); Bangla digits normalized | False flags cost an extra LLM round trip each. The fallback model's Bangla 4 PM → 8 PM slip went unnoticed. | Parametrized cross-check tests. |
| Prompt v6: no_op discipline (tomorrow, past, cancelled, already in forecast, unsupported effects); "hold its charge" means no_discharge; "don't top up" means no_charge | Trap-catalogue misses. | 53/53 three times. |
| Conservative repairs are stated in `explanation` | The team's rule: never clamp or invent silently. Repairs only run after the LLM retry budget is used up, and each picks the reading that stays valid under any ground truth. | `test_lenient_repairs_do_not_invent_constraints`. |
| Safe-baseline fallback when our own replay fails | Final validator in the request path. | `test_self_check_failure_falls_back_to_safe_baseline`. |
| Streamed body cap, deep nesting returns 400, 27 s safety net, OpenAPI fix, `LOG_LEVEL` and `.env` handling | API audit found a memory-exhaustion crash, a 500 on deep nesting, and more. | Live malformed matrix; regression tests. |
| Keep-warm background call every 4 minutes | The first burst after idle reached a p95 of about 5.4 s. | Warm p95 2.5–3.7 s. |
| Multi-arch image (amd64 and arm64) | Judges on Apple Silicon or ARM servers. | `docker buildx imagetools inspect` shows both platforms. |

## 2b. Final verification round (23:20–23:30, frozen build)

| Check (live URL unless noted) | Result |
|---|---|
| Independent hold-out: 60 notes written by a separate author who never saw our prompt or tests (`tests/data/independent_holdout.jsonl`) | **60/60** on every field; a transient OpenAI connection error was recovered by the retry |
| The Problem Statement's own example notes (sections 4.2 and 11.4) | 7/7 |
| 12 edge scenarios: fractional data, zero solar, solar above demand, flat and zero tariffs, zero demand, unusable, zero-rate and zero-capacity batteries, 100x scale | 12/12 HTTP 200, 0 rule violations |
| Burst of 30 concurrent requests | 0 errors, p95 0.9 s |
| 20 valid + 20 malformed requests interleaved | 20/20 valid, 20/20 JSON 400; healthy afterwards |
| Memory over the last hour | about 149 MB on average, 215 MB peak (limit 8 GB), so no leak |
| Live logs | 0 secret-shaped strings, 0 errors |

Known observability gap, not fixed because of the freeze: keep-warm calls bypass the `/version` success counter
and log only on failure. The code path was verified separately with the real OpenAI client.

## 3. What was deliberately not changed

- **Per-note parallel LLM calls rather than one call for all notes.** They're already parallel, and a single call would lengthen output and couple the notes' failures.
- **The LLM does not see capacity.** Percent-of-capacity is converted in code, which is safer than asking the LLM to do arithmetic.
- **Missing or odd `Content-Type` is accepted.** Section 6.1 only requires 400 for malformed or structurally invalid bodies. Rejecting valid JSON over a header would risk a valid scored case.
- **No demo UI.** It is worth 0 base points, and the video is produced separately.

## 4. Remaining risks, ranked by points at stake

1. **Repository or image still private at evaluation time** (docs 10, Docker 4, the LLM-in-path check). Make both public right after the deadline.
2. **Unseen phrasings in hidden notes** (each miss costs interpretation, application and optimization points for that case). Mitigations: 490 labelled notes at 100%, cross-check re-asks, and a fallback model from a different family.
3. **Platform-level 502 during a redeploy** (seen once, 10 s after a deploy, and never reached the app). Do not redeploy while judging runs.
4. **OpenAI key revoked or out of credit during judging** (all interpretation points). Keep the account funded. The key was shared in a chat, so rotate it only after results are published.

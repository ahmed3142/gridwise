# 3-minute solution video: script and shot list

Target length is 2:45, which leaves margin under the 3:00 limit. Record the screen with a voice-over.
Show the repository, a terminal and the deployed URL.

| Time | Show | Say |
|---|---|---|
| 0:00–0:20 | Problem statement, section 03 diagram | "GridWise must understand free-text operator notes, turn them into safe, machine-checkable directives, and produce the cheapest valid 24-hour grid, solar and battery plan." |
| 0:20–0:55 | README architecture diagram, `app/interpreter/semantic.py` | "One OpenAI Structured Outputs call per note, in parallel. The model extracts the directive type, the clock windows and the quantity with its meaning. Deterministic code converts those to hours and numbers, so there are no off-by-one or percentage-math errors. The schema-constrained output means the model cannot invent a directive type." |
| 0:55–1:20 | `app/directives.py` guardrails, `crosscheck.py` | "The section 08 guardrails validate every directive. A deterministic cross-check compares the answer with the literal note, and any disagreement triggers one re-ask. Failures degrade to a controlled no_op within a 24-second budget, never a crash." |
| 1:20–1:50 | `app/optimizer/lp.py` docstring | "An exact linear program solved with HiGHS in about 10 ms. The objective is cost, then peak, then throughput. Overlapping or non-round directives are handled conservatively. An infeasible directive set means a misread note, so we re-ask about exactly that note." |
| 1:50–2:20 | Terminal: `python scripts/judge.py --url https://<app>.fly.dev` | "Our judge harness replays every plan against the organizers' ground truth: 10 out of 10, optimal cost matched, p95 latency printed." |
| 2:20–2:40 | `pytest -q` output, Docker run and `/health` | "170 automated tests, a Docker fallback image with no secrets, and it starts even without a key." |
| 2:40–2:50 | README credits | "Thanks. The details are in the README." |

Checklist: the video is 3:00 or shorter, it plays in an incognito window, and it says nothing about
secrets. Upload it early, not in the last 5 minutes.

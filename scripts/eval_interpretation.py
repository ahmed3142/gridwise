#!/usr/bin/env python3
"""Measure LLM interpretation accuracy on paraphrased notes (rubric category 1, per field).

Runs every case in tests/data/paraphrases.json through the REAL interpreter (LLM + normalizer +
cross-check + re-ask) and reports accuracy for relevance, directive_type, hours and values, plus
every miss. Needs OPENAI_API_KEY (environment or .env).

    python scripts/eval_interpretation.py
    python scripts/eval_interpretation.py --model gpt-5-mini --repeat 2
    python scripts/eval_interpretation.py --url https://your-app.fly.dev     # through the deployed API
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.validator import compare_interpretation  # noqa: E402

DATA = json.loads((ROOT / "tests" / "data" / "paraphrases.json").read_text(encoding="utf-8"))
BASE = json.loads((ROOT / "tests" / "data" / "public_sample_cases.json").read_text(encoding="utf-8"))["cases"][8]["input"]


def expected_entry(case: dict, index: int) -> dict:
    exp = dict(case["expected"])
    exp["applies"] = exp["directive_type"] != "no_op"
    exp["note_index"] = index
    return exp


async def run_direct(model: str | None, repeat: int) -> list[tuple[dict, dict, float]]:
    import os

    if model:
        os.environ["OPENAI_MODEL"] = model
    from app.config import load_settings
    from app.interpreter.llm import OpenAIInterpreterClient
    from app.interpreter.service import InterpretationService
    from app.schemas import Battery

    settings = load_settings()
    if not settings.openai_api_key:
        print("OPENAI_API_KEY is not set (environment or .env).")
        sys.exit(2)
    client = OpenAIInterpreterClient(settings)
    await client.resolve_models()
    print(f"model: {client.primary} (fallback {client.fallback})")
    battery = Battery.model_validate(DATA["battery"])
    service = InterpretationService(client, settings)
    out = []

    async def one(case, rep):
        notes = case.get("notes") or [case["note"]]
        target = case.get("target", 0)
        # vary nothing but bypass the cache between repeats by adding a trailing space per repeat
        notes = [n + " " * rep for n in notes]
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        results = await service.interpret(notes, battery, loop.time() + 25)
        ms = (time.perf_counter() - started) * 1000
        return case, results[target].directive.to_interpretation(), ms

    sem = asyncio.Semaphore(8)

    async def guarded(case, rep):
        async with sem:
            return await one(case, rep)

    for rep in range(repeat):
        out += await asyncio.gather(*(guarded(c, rep) for c in DATA["cases"]))
    await client.aclose()
    return out


def run_http(url: str, repeat: int) -> list[tuple[dict, dict, float]]:
    out = []
    for rep in range(repeat):
        for n, case in enumerate(DATA["cases"]):
            notes = case.get("notes") or [case["note"]]
            target = case.get("target", 0)
            payload = dict(BASE, scenario_id=f"PARA-{n}-{rep}", operator_notes=notes, battery=DATA["battery"])
            req = urllib.request.Request(f"{url.rstrip('/')}/optimize-energy", data=json.dumps(payload).encode(),
                                         method="POST", headers={"Content-Type": "application/json"})
            started = time.perf_counter()
            with urllib.request.urlopen(req, timeout=35) as resp:
                body = json.loads(resp.read())
            ms = (time.perf_counter() - started) * 1000
            out.append((case, body["directive_interpretation"][target], ms))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="override OPENAI_MODEL for the direct run")
    ap.add_argument("--url", help="evaluate through a running API instead of calling the interpreter directly")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    results = run_http(args.url, args.repeat) if args.url else asyncio.run(run_direct(args.model, args.repeat))
    totals = {"relevance": 0, "directive_type": 0, "hours": 0, "values": 0}
    all_ok = 0
    for case, got, ms in results:
        exp = expected_entry(case, case.get("target", 0))
        cmp = compare_interpretation(got, exp)
        for k, ok in cmp.items():
            totals[k] += int(ok)
        if all(cmp.values()):
            all_ok += 1
        else:
            note = case.get("note") or case["notes"][case.get("target", 0)]
            print(f"MISS ({ms:.0f} ms) {note!r}\n    expected {exp['directive_type']} {exp['structured_adjustment']}\n"
                  f"    got      {got.get('directive_type')} {got.get('structured_adjustment')}")
    n = len(results)
    lat = sorted(ms for _, _, ms in results)
    print()
    for k, v in totals.items():
        print(f"{k:<15} {v}/{n}  ({100 * v / n:.1f}%)")
    print(f"{'fully correct':<15} {all_ok}/{n}  ({100 * all_ok / n:.1f}%)")
    print(f"latency p50 {lat[len(lat) // 2]:.0f} ms  p95 {lat[int(0.95 * (len(lat) - 1))]:.0f} ms")
    return 0 if all_ok == n else 1


if __name__ == "__main__":
    sys.exit(main())

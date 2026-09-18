#!/usr/bin/env python3
"""Trap-catalogue evaluation through the REAL interpretation path (LLM + normalizer + guardrails).

Each line of tests/data/eval_notes.jsonl holds one note with its ground truth (labels written first,
wording second). Battery used for percent conversions: capacity 200 kWh, minimum 40 kWh.

    python scripts/eval_notes.py                 # one run
    python scripts/eval_notes.py --repeat 3      # determinism: the same answers three times
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from app.interpreter.llm import OpenAIInterpreterClient  # noqa: E402
from app.interpreter.service import InterpretationService  # noqa: E402
from app.schemas import Battery  # noqa: E402

BATTERY = Battery(capacity_kwh=200, initial_energy_kwh=120, minimum_energy_kwh=40,
                  max_charge_kwh_per_hour=50, max_discharge_kwh_per_hour=50)
VALUE_KEY = {"solar_reduction": "factor", "minimum_battery_reserve": "minimum_energy_kwh", "max_grid_window": "max_grid_kwh"}


def check(case: dict, directive) -> dict:
    exp_type = case["type"]
    got = directive.to_interpretation()
    adj = got["structured_adjustment"] or {}
    res = {
        "applies": got["applies"] == (exp_type != "no_op"),
        "type": got["directive_type"] == exp_type,
        "hours": (adj.get("hours") == case["hours"]) if exp_type != "no_op" else adj == {},
        "value": True,
    }
    if exp_type in VALUE_KEY:
        v = adj.get(VALUE_KEY[exp_type])
        res["value"] = isinstance(v, (int, float)) and abs(v - case["value"]) <= 0.01
    return res


async def run_once(service, cases, fresh: bool, tag: int):
    sem = asyncio.Semaphore(8)

    async def one(case):
        note = case["note"] + (" " * tag if fresh else "")  # bypass the cache between repeats
        async with sem:
            loop = asyncio.get_running_loop()
            out = await service.interpret([note], BATTERY, loop.time() + 25)
        return out[0].directive

    return await asyncio.gather(*(one(c) for c in cases))


async def main_async(repeat: int):
    settings = load_settings()
    if not settings.openai_api_key:
        print("OPENAI_API_KEY is not set (.env)")
        sys.exit(2)
    cases = [json.loads(line) for line in (ROOT / "tests" / "data" / "eval_notes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    client = OpenAIInterpreterClient(settings)
    service = InterpretationService(client, settings)
    await client.resolve_models()
    print(f"model {client.primary} (fallback {client.fallback}); {len(cases)} labelled notes; {repeat} run(s)")
    answers = []
    for r in range(repeat):
        directives = await run_once(service, cases, fresh=True, tag=r)
        answers.append([json.dumps(d.to_interpretation()["structured_adjustment"]) + d.directive_type for d in directives])
        totals = {"applies": 0, "type": 0, "hours": 0, "value": 0}
        full = 0
        for case, d in zip(cases, directives):
            res = check(case, d)
            for k, ok in res.items():
                totals[k] += int(ok)
            if all(res.values()):
                full += 1
            else:
                got = d.to_interpretation()
                print(f"  MISS run {r + 1}: {case['note'][:90]!r}\n       expected {case['type']} {case.get('hours')} {case.get('value')}"
                      f"\n       got      {got['directive_type']} {got['structured_adjustment']}")
        n = len(cases)
        print(f"run {r + 1}: " + "  ".join(f"{k} {v}/{n}" for k, v in totals.items()) + f"  | fully correct {full}/{n}")
    if repeat > 1:
        stable = sum(len({a[i] for a in answers}) == 1 for i in range(len(cases)))
        print(f"determinism: {stable}/{len(cases)} notes gave identical results in all {repeat} runs")
    await client.aclose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    asyncio.run(main_async(ap.parse_args().repeat))

#!/usr/bin/env python3
"""First-minutes probe: verify the OpenAI key, which models it can use, and their latency/accuracy.

Reads OPENAI_API_KEY from the environment or .env. For each candidate model it interprets a few
public-sample notes through the real interpreter client and prints latency and correctness, so the
team can pin OPENAI_MODEL / OPENAI_FALLBACK_MODEL to measured choices before deploying.

    python scripts/probe_llm.py
    python scripts/probe_llm.py --models gpt-4.1-mini,gpt-5-mini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402
from app.interpreter.llm import LLMCallError, OpenAIInterpreterClient  # noqa: E402
from app.interpreter.normalize import normalize  # noqa: E402
from app.interpreter.prompt import SYSTEM_PROMPT, user_message  # noqa: E402
from app.schemas import Battery  # noqa: E402

CASES = json.loads((ROOT / "tests" / "data" / "public_sample_cases.json").read_text(encoding="utf-8"))["cases"]


async def run(models: list[str], per_model: int) -> None:
    settings = load_settings()
    if not settings.openai_api_key:
        print("OPENAI_API_KEY is not set (environment or .env).")
        sys.exit(2)
    client = OpenAIInterpreterClient(settings)
    await client.resolve_models()
    print(f"auto-selected primary={client.primary} fallback={client.fallback}")
    if client.available is not None:
        pref = [m for m in settings.model_preference if m in client.available]
        print(f"preference models available to this key: {pref}")
    samples = []
    for case in CASES:
        for i, note in enumerate(case["input"]["operator_notes"]):
            samples.append((case, i, note))
    samples = samples[:per_model]

    for model in models:
        ok = 0
        times = []
        for case, i, note in samples:
            notes = case["input"]["operator_notes"]
            battery = Battery.model_validate(case["input"]["battery"])
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message(notes, i)}]
            started = time.perf_counter()
            try:
                sem = await client.interpret(model, messages, timeout=20)
            except LLMCallError as exc:
                print(f"  {model}: {exc}")
                break
            times.append((time.perf_counter() - started) * 1000)
            d, _ = normalize(sem, battery, i, note, strict=False)
            exp = case["expected_output"]["directive_interpretation"][i]
            good = d.directive_type == exp["directive_type"] and d.structured_adjustment() == exp["structured_adjustment"]
            ok += int(good)
            if not good:
                print(f"  {model} MISS: {note[:70]!r} -> {d.directive_type} {d.structured_adjustment()}")
        if times:
            times.sort()
            print(f"{model:<16} correct {ok}/{len(times)}  median {times[len(times)//2]:.0f} ms  max {times[-1]:.0f} ms")
    await client.aclose()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="comma-separated model ids (default: configured preference list)")
    ap.add_argument("--notes", type=int, default=21, help="how many public notes to try per model")
    args = ap.parse_args()
    models = args.models.split(",") if args.models else list(load_settings().model_preference)
    asyncio.run(run(models, args.notes))


if __name__ == "__main__":
    main()

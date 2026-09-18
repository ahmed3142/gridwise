#!/usr/bin/env python3
"""Print the `input` object of one public sample case (for curl / Postman testing).

    python scripts/make_request.py SAMPLE-06 > sample.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    case_id = sys.argv[1] if len(sys.argv) > 1 else "SAMPLE-01"
    cases = json.loads((ROOT / "tests" / "data" / "public_sample_cases.json").read_text(encoding="utf-8"))["cases"]
    for case in cases:
        if case["id"] == case_id:
            sys.stdout.write(json.dumps(case["input"], indent=2) + "\n")
            return 0
    sys.stderr.write(f"unknown case {case_id}; choose one of {[c['id'] for c in cases]}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())

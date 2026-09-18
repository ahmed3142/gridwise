#!/usr/bin/env python3
"""Local "judge harness": score a running GridWise service the way the organizers' judge does.

For every case in a case file (default: the public sample pack) it
  1. POSTs case.input to <url>/optimize-energy,
  2. checks the response schema and interpretation shape,
  3. compares directive_interpretation with the expected ground truth field by field
     (relevance/no_op, directive_type, hours, numeric values) - like rubric category 1,
  4. replays hourly_plan against the GROUND-TRUTH directives and every GridWise rule - category 2,
  5. scores cost quality min(1, reference_cost / our_cost) - category 3,
and reports latency (p50/p95 including repeats). Exit code 0 only if every case passes.

Usage (any OS; stdlib only + this repo's app.validator):
    python scripts/judge.py --url http://localhost:8080
    python scripts/judge.py --url https://your-app.fly.dev --repeat 3 --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.validator import compare_interpretation, response_violations, schedule_violations  # noqa: E402

DEFAULT_CASES = ROOT / "tests" / "data" / "public_sample_cases.json"


def http(method: str, url: str, payload: dict | None = None, timeout: float = 35.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            status = resp.status
            headers = dict(resp.headers)
    except urllib.error.HTTPError as exc:
        body, status, headers = exc.read(), exc.code, dict(exc.headers)
    elapsed = (time.perf_counter() - started) * 1000
    try:
        parsed = json.loads(body) if body else None
    except ValueError:
        parsed = None
    return status, parsed, elapsed, headers


def score_case(base: str, case: dict) -> dict:
    status, body, ms, headers = http("POST", f"{base}/optimize-energy", case["input"])
    result = {"id": case.get("id", case["input"].get("scenario_id")), "status": status, "ms": ms,
              "degraded": headers.get("X-GridWise-Degraded") == "true", "problems": []}
    if status != 200 or not isinstance(body, dict):
        result["problems"].append(f"HTTP {status}: {str(body)[:200]}")
        return result
    expected = case.get("expected_output", {})
    result["problems"] += [f"schema/self: {v}" for v in response_violations(case["input"], body)]

    fields = {"relevance": 0, "directive_type": 0, "hours": 0, "values": 0}
    exp_dirs = expected.get("directive_interpretation")
    notes = len(case["input"]["operator_notes"])
    if exp_dirs:
        got = body.get("directive_interpretation", [])
        for i, exp in enumerate(exp_dirs):
            cmp = compare_interpretation(got[i] if i < len(got) else {}, exp)
            for k, ok in cmp.items():
                fields[k] += int(ok)
            if not all(cmp.values()):
                g = got[i] if i < len(got) else None
                result["problems"].append(
                    f"note {i}: expected {exp['directive_type']} {exp['structured_adjustment']}, got "
                    f"{g and g.get('directive_type')} {g and g.get('structured_adjustment')}"
                )
        truth_violations = schedule_violations(case["input"], exp_dirs, body)
        result["problems"] += [f"ground-truth replay: {v}" for v in truth_violations]
        result["valid_under_truth"] = not truth_violations
    result["fields"] = {k: f"{v}/{notes}" for k, v in fields.items()}
    ref = expected.get("total_cost_bdt")
    cost = body.get("total_cost_bdt")
    if isinstance(ref, (int, float)) and isinstance(cost, (int, float)) and cost > 0:
        result["cost"] = cost
        result["quality"] = min(1.0, ref / cost) if result.get("valid_under_truth", True) else 0.0
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8080", help="service base URL")
    ap.add_argument("--cases", default=str(DEFAULT_CASES), help="case pack JSON (public-sample format)")
    ap.add_argument("--repeat", type=int, default=1, help="run the whole pack N times (latency/stability)")
    ap.add_argument("--concurrency", type=int, default=1, help="parallel requests")
    ap.add_argument("--report", help="optional path to write a JSON report")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    status, body, ms, _ = http("GET", f"{base}/health", timeout=10)
    health_ok = status == 200 and isinstance(body, dict) and body.get("status") == "ok"
    print(f"GET /health -> {status} {body} ({ms:.0f} ms) {'OK' if health_ok else 'FAIL'}")
    _, version, _, _ = http("GET", f"{base}/version", timeout=10)
    if isinstance(version, dict):
        print(f"service {version.get('version')} prompt {version.get('prompt_version')} "
              f"model {version.get('primary_model')} (fallback {version.get('fallback_model')})")

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    jobs = [c for _ in range(max(1, args.repeat)) for c in cases]
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        results = list(pool.map(lambda c: score_case(base, c), jobs))

    first = results[: len(cases)]
    print()
    print(f"{'case':<12} {'http':>4} {'ms':>7}  {'relev':>5} {'type':>5} {'hours':>5} {'value':>5}  {'valid':>5} {'quality':>7}")
    for r in first:
        f = r.get("fields", {})
        print(f"{r['id']:<12} {r['status']:>4} {r['ms']:>7.0f}  {f.get('relevance','-'):>5} {f.get('directive_type','-'):>5} "
              f"{f.get('hours','-'):>5} {f.get('values','-'):>5}  {str(r.get('valid_under_truth','-')):>5} "
              f"{r.get('quality', 0):>7.4f}{'  DEGRADED' if r['degraded'] else ''}")
        for p in r["problems"][:6]:
            print(f"    - {p}")

    latencies = sorted(r["ms"] for r in results)
    p95 = latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))]
    failures = [r for r in results if r["status"] != 200]
    passed = [r for r in first if not r["problems"] and r["status"] == 200]
    quality = statistics.mean(r.get("quality", 0.0) for r in first) if first else 0.0
    print()
    print(f"cases passed: {len(passed)}/{len(first)}   requests: {len(results)}   non-200: {len(failures)}")
    print(f"latency p50 {statistics.median(latencies):.0f} ms   p95 {p95:.0f} ms   max {latencies[-1]:.0f} ms")
    print(f"optimization quality (mean ratio): {quality:.4f}")
    if args.report:
        Path(args.report).write_text(json.dumps({"health_ok": health_ok, "results": results}, indent=2), encoding="utf-8")
    return 0 if health_ok and len(passed) == len(first) and not failures else 1


if __name__ == "__main__":
    sys.exit(main())

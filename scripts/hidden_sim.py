#!/usr/bin/env python3
"""Hidden-case simulator: random scenarios + freshly templated paraphrased notes with KNOWN ground truth.

For each generated case it
  * builds random demand / solar / tariff / battery data (different from the public pack),
  * writes 1-3 operator notes from many phrasing templates (varied time formats, units, word order,
    distractors, typos), keeping the exact ground-truth directive for each,
  * keeps only scenarios that are feasible under the ground truth (as the organizers guarantee),
  * POSTs to the service and scores it like the judge: interpretation vs truth, plan validity vs truth,
    and cost vs the true LP optimum.

    python scripts/hidden_sim.py --url https://your-app.up.railway.app --cases 40 --concurrency 6 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.directives import Directive  # noqa: E402
from app.optimizer.lp import build_problem, solve  # noqa: E402
from app.schemas import OptimizeRequest  # noqa: E402
from app.validator import compare_interpretation, response_violations, schedule_violations  # noqa: E402


# ------------------------------------------------------------------ time formatting
def t_ampm(h):
    if h in (0, 24):
        return "midnight"
    if h == 12:
        return "noon"
    return f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"


def t_compact(h):
    if h in (0, 24):
        return "12am"
    return f"{h % 12 or 12}{'am' if h < 12 else 'pm'}"


def t_dotted(h):
    if h in (0, 24):
        return "12:00 a.m."
    return f"{h % 12 or 12}:00 {'a.m.' if h < 12 else 'p.m.'}"


def t_24(h):
    return f"{h % 24:02d}:00" if h != 24 else "24:00"


def t_hrs(h):
    return f"{h % 24:02d}00 hrs"


def window_phrase(rng, s, e):
    n = (e - s) % 24 or 24
    fmt = rng.choice([t_ampm, t_compact, t_24, t_dotted, t_hrs, t_ampm, t_24])
    options = [
        f"from {fmt(s)} until {fmt(e)}",
        f"between {fmt(s)} and {fmt(e)}",
        f"from {fmt(s)} to {fmt(e)}",
        f"{fmt(s)}-{fmt(e)}",
        f"from {fmt(s)} till {fmt(e)}",
        f"for {n} hour{'s' if n > 1 else ''} starting at {fmt(s)}",
    ]
    if fmt in (t_24,):
        options.append(f"during {fmt(s)}–{fmt(e)}")
    return rng.choice(options)


# ------------------------------------------------------------------ note templates (NOT the prompt's examples)
SOLAR_REMAIN = [
    "Solar output will drop to about {r}% {w} due to panel cleaning.",
    "Only {r} percent of forecast PV will be usable {w}.",
    "PV generation is expected at roughly {r}% of normal {w} because of smoke haze.",
    "{W}, the rooftop arrays will deliver just {r}% of their forecast.",
    "plan for {r} per cent of the usual solar {w} (scaffolding shade).",
]
SOLAR_REDUCE = [
    "Expect a {p}% reduction in rooftop solar {w} for inverter maintenance.",
    "Solar generation will be cut by {p} percent {w}.",
    "Rooftop PV output falls by {p}% {w} while crews wash the arrays.",
    "{W} solar will be {p}% lower than forecast.",
]
RESERVE_KWH = [
    "Keep at least {k} kWh in the battery {w} for emergency loads.",
    "The battery must not drop below {k} kWh {w}.",
    "Hold a reserve of {k} kWh {w} for the medical centre.",
    "{W}, maintain a minimum stored energy of {k} kWh.",
    "Reserve {k} kWh of battery energy {w} (fire safety requirement).",
]
RESERVE_PCT = [
    "Maintain at least {q}% of battery capacity {w}.",
    "Keep the battery at least {q}% charged {w}.",
    "{W}, the state of charge must stay at or above {q}%.",
]
NO_CHARGE = [
    "The battery charger will be offline {w} for maintenance.",
    "Do not charge the battery {w}.",
    "Battery charging is disabled {w} while technicians inspect the rectifier.",
    "{W}, charging the storage system is not allowed.",
    "no batery charging {w} - charger under repair.",
]
NO_DISCHARGE = [
    "The battery must not discharge {w} during relay testing.",
    "Battery discharge is locked out {w}.",
    "{W}, the battery may not supply any load.",
    "No discharging of the storage system {w}.",
]
MAX_GRID = [
    "Grid import must not exceed {g} kWh in any hour {w}.",
    "The feeder limit is {g} kW {w}.",
    "Keep grid purchases at or below {g} kWh per hour {w}.",
    "{W}, the substation allows at most {g} kWh of grid intake per hour.",
]
DISTRACTORS = [
    "The cafeteria will serve a special lunch tomorrow.",
    "The IT team is upgrading the library Wi-Fi next week.",
    "Next month the utility will inspect the main transformer.",
    "The battery warranty paperwork was filed yesterday.",
    "Please switch off projectors after evening classes.",
    "Yesterday the solar panels were cleaned from 1 PM to 3 PM.",
    "Tomorrow, grid import will be capped at 120 kWh from 6 PM to 9 PM.",
    "The chess club meets at 5 PM in room 204.",
    "A new EV charger for staff cars will be installed next semester.",
]


def fill(tpl, rng, s, e, **kw):
    w = window_phrase(rng, s, e)
    text = tpl.format(w=w, W=w[0].upper() + w[1:], **kw)
    return text[0].upper() + text[1:] if rng.random() < 0.9 else text


def hours_of(s, e):
    return [h % 24 for h in range(s, s + ((e - s) % 24 or 24))]


def make_scenario(rng, idx):
    cap = rng.choice([150, 180, 200, 220, 240, 250, 260, 300])
    mn = rng.choice([20, 25, 30, 35, 40, 50])
    init = rng.randrange(mn + 20, cap - 10, 5)
    rate = rng.choice([40, 45, 50, 55, 60, 65, 75])
    peak = rng.randint(150, 260)
    hours = []
    for h in range(24):
        demand = rng.randint(60, 120) + (rng.randint(40, 110) if 8 <= h <= 21 else 0)
        solar = max(0, int(peak * (1 - abs(h - 12.5) / 6.5))) if 6 <= h <= 18 else 0
        tariff = rng.choice([4, 5, 6, 7, 8]) if h < 6 else rng.choice([22, 26, 28, 30, 32, 35]) if 17 <= h <= 21 else rng.choice([10, 12, 13, 14, 15, 16, 18])
        hours.append({"hour": h, "demand_kwh": demand, "solar_kwh": solar, "tariff_bdt_per_kwh": tariff})
    battery = {"capacity_kwh": cap, "initial_energy_kwh": init, "minimum_energy_kwh": mn,
               "max_charge_kwh_per_hour": rate, "max_discharge_kwh_per_hour": rate}
    return {"scenario_id": f"HIDDEN-{idx:03d}", "hours": hours, "battery": battery}


def make_note(rng, scen):
    cap = scen["battery"]["capacity_kwh"]
    kind = rng.choice(["solar", "solar", "reserve", "reserve", "no_charge", "no_discharge", "grid", "grid"])
    if kind == "solar":
        s = rng.randint(8, 15)
        e = min(s + rng.randint(1, 4), 18)
    else:
        s = rng.randint(0, 21)
        e = s + rng.randint(1, 4)
        if e > 24:
            e = 24
    hrs = hours_of(s, e)
    if kind == "solar":
        if rng.random() < 0.5:
            r = rng.choice([10, 15, 20, 25, 30, 40, 50, 60, 70, 75])
            text = fill(rng.choice(SOLAR_REMAIN), rng, s, e, r=r)
            factor = r / 100
        else:
            p = rng.choice([20, 25, 30, 40, 50, 60, 70, 75, 80, 90])
            text = fill(rng.choice(SOLAR_REDUCE), rng, s, e, p=p)
            factor = round(1 - p / 100, 6)
        return text, {"directive_type": "solar_reduction", "structured_adjustment": {"hours": sorted(hrs), "factor": factor}}
    if kind == "reserve":
        if rng.random() < 0.6:
            k = rng.randrange(40, int(cap * 0.7), 5)
            text = fill(rng.choice(RESERVE_KWH), rng, s, e, k=k)
            value = k
        else:
            q = rng.choice([20, 25, 30, 40, 50])
            text = fill(rng.choice(RESERVE_PCT), rng, s, e, q=q)
            value = cap * q / 100
        return text, {"directive_type": "minimum_battery_reserve", "structured_adjustment": {"hours": sorted(hrs), "minimum_energy_kwh": value}}
    if kind == "no_charge":
        return fill(rng.choice(NO_CHARGE), rng, s, e), {"directive_type": "no_charge_window", "structured_adjustment": {"hours": sorted(hrs)}}
    if kind == "no_discharge":
        return fill(rng.choice(NO_DISCHARGE), rng, s, e), {"directive_type": "no_discharge_window", "structured_adjustment": {"hours": sorted(hrs)}}
    g = rng.randrange(120, 260, 5)
    return fill(rng.choice(MAX_GRID), rng, s, e, g=g), {"directive_type": "max_grid_window", "structured_adjustment": {"hours": sorted(hrs), "max_grid_kwh": g}}


def to_directives(expected):
    out = []
    for i, e in enumerate(expected):
        a = e["structured_adjustment"] or {}
        out.append(Directive(i, e["directive_type"], hours=tuple(a.get("hours", ())), factor=a.get("factor"),
                             minimum_energy_kwh=a.get("minimum_energy_kwh"), max_grid_kwh=a.get("max_grid_kwh")))
    return out


def generate(n, seed):
    rng = random.Random(seed)
    cases = []
    idx = 0
    while len(cases) < n:
        idx += 1
        scen = make_scenario(rng, idx)
        notes, expected = [], []
        for _ in range(rng.choice([1, 2, 2, 3, 3])):
            if rng.random() < 0.25:
                notes.append(rng.choice(DISTRACTORS))
                expected.append({"directive_type": "no_op", "structured_adjustment": None})
            else:
                t, exp = make_note(rng, scen)
                notes.append(t)
                expected.append(exp)
        for i, e in enumerate(expected):
            e["note_index"] = i
            e["applies"] = e["directive_type"] != "no_op"
        payload = dict(scen, operator_notes=notes)
        req = OptimizeRequest.model_validate(payload)
        truth = build_problem(req.hours, req.battery, to_directives(expected), conservative=False)
        sol = solve(truth)
        if sol is None:  # organizers only use feasible scenarios
            continue
        cases.append({"input": payload, "expected": expected, "optimal_cost": sol.cost})
    return cases


def post(url, payload):
    req = urllib.request.Request(f"{url}/optimize-energy", data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            body, status, deg = json.loads(r.read()), r.status, r.headers.get("x-gridwise-degraded")
    except urllib.error.HTTPError as e:
        body, status, deg = None, e.code, None
    except Exception as e:  # noqa: BLE001
        body, status, deg = {"error": str(e)}, 0, None
    return body, status, deg, (time.perf_counter() - started) * 1000


def score(url, case):
    body, status, deg, ms = post(url, case["input"])
    r = {"id": case["input"]["scenario_id"], "status": status, "ms": ms, "degraded": bool(deg), "misses": [], "violations": []}
    if status != 200 or not isinstance(body, dict):
        r["violations"].append(f"HTTP {status}")
        return r
    r["violations"] += response_violations(case["input"], body)
    got = body.get("directive_interpretation", [])
    r["fields"] = {"relevance": 0, "directive_type": 0, "hours": 0, "values": 0}
    for i, exp in enumerate(case["expected"]):
        cmp = compare_interpretation(got[i] if i < len(got) else {}, exp)
        for k, ok in cmp.items():
            r["fields"][k] += int(ok)
        if not all(cmp.values()):
            r["misses"].append((case["input"]["operator_notes"][i], exp, got[i] if i < len(got) else None))
    truth_v = schedule_violations(case["input"], case["expected"], body)
    r["valid"] = not truth_v
    r["violations"] += [f"truth: {v}" for v in truth_v]
    cost = body.get("total_cost_bdt", 0)
    r["quality"] = (1.0 if cost <= case["optimal_cost"] + 0.01 else case["optimal_cost"] / cost) if r["valid"] else 0.0
    r["notes"] = len(case["expected"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--cases", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--dump", help="write generated cases to this JSON file")
    a = ap.parse_args()
    cases = generate(a.cases, a.seed)
    if a.dump:
        Path(a.dump).write_text(json.dumps(cases, indent=1), encoding="utf-8")
    url = a.url.rstrip("/")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.concurrency) as ex:
        res = list(ex.map(lambda c: score(url, c), cases))
    wall = time.perf_counter() - t0
    n_notes = sum(r.get("notes", 0) for r in res)
    tot = {k: sum(r.get("fields", {}).get(k, 0) for r in res) for k in ("relevance", "directive_type", "hours", "values")}
    for r in res:
        for note, exp, got in r["misses"]:
            print(f"MISS {r['id']}: {note!r}\n    expected {exp['directive_type']} {exp['structured_adjustment']}\n"
                  f"    got      {got and got.get('directive_type')} {got and got.get('structured_adjustment')}")
        if r["violations"]:
            print(f"VIOLATION {r['id']}: {r['violations'][:3]}")
    lat = sorted(r["ms"] for r in res)
    print(f"\ncases {len(res)} ({n_notes} notes), wall {wall:.1f}s, non-200 {sum(r['status'] != 200 for r in res)}, degraded {sum(r['degraded'] for r in res)}")
    for k, v in tot.items():
        print(f"  {k:<15} {v}/{n_notes} ({100 * v / max(n_notes, 1):.1f}%)")
    print(f"  plans valid vs ground truth: {sum(r.get('valid', False) for r in res)}/{len(res)}")
    print(f"  mean cost quality (judge formula): {statistics.mean(r.get('quality', 0) for r in res):.4f}")
    print(f"  latency p50 {statistics.median(lat):.0f} ms  p95 {lat[int(0.95 * (len(lat) - 1))]:.0f} ms  max {lat[-1]:.0f} ms")


if __name__ == "__main__":
    main()

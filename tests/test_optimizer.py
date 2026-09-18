"""Optimizer correctness: optimality vs reference, every directive, edge cases, randomized scenarios."""

from __future__ import annotations

import random

import pytest

from app.directives import Directive
from app.optimizer.lp import build_problem, solve, solve_relaxed
from app.optimizer.plan import build_hourly_plan, plan_totals
from app.schemas import OptimizeRequest
from app.validator import schedule_violations


def _directives(entries):
    out = []
    for e in entries:
        a = e["structured_adjustment"] or {}
        out.append(
            Directive(
                note_index=e["note_index"], directive_type=e["directive_type"], hours=tuple(a.get("hours", ())),
                factor=a.get("factor"), minimum_energy_kwh=a.get("minimum_energy_kwh"), max_grid_kwh=a.get("max_grid_kwh"),
            )
        )
    return out


def _run(payload, directives):
    req = OptimizeRequest.model_validate(payload)
    p = build_problem(req.hours, req.battery, directives)
    sol = solve(p)
    assert sol is not None
    plan = build_hourly_plan(p, sol)
    return {"hourly_plan": plan, **plan_totals(plan, p.tariff)}, sol


@pytest.mark.parametrize("index", range(10))
def test_matches_reference_optimum(public_cases, index):
    case = public_cases[index]
    expected = case["expected_output"]
    resp, sol = _run(case["input"], _directives(expected["directive_interpretation"]))
    assert schedule_violations(case["input"], expected["directive_interpretation"], resp, tol=1e-6) == []
    assert abs(resp["total_cost_bdt"] - expected["total_cost_bdt"]) <= 1e-3
    assert resp["peak_grid_kwh"] <= expected["peak_grid_kwh"] + 1e-6  # tie-break never worse than reference
    assert sol.stages == 3


def _flat_payload(**battery):
    hours = []
    for h in range(24):
        hours.append({
            "hour": h,
            "demand_kwh": 100 + 20 * (8 <= h <= 20),
            "solar_kwh": max(0, 60 - 8 * abs(h - 12)) if 6 <= h <= 18 else 0,
            "tariff_bdt_per_kwh": 6 if h < 6 else 22 if 17 <= h <= 21 else 12,
        })
    b = {"capacity_kwh": 200, "initial_energy_kwh": 100, "minimum_energy_kwh": 20,
         "max_charge_kwh_per_hour": 50, "max_discharge_kwh_per_hour": 50}
    b.update(battery)
    return {"scenario_id": "T", "operator_notes": ["x"], "hours": hours, "battery": b}


def test_each_directive_is_enforced():
    payload = _flat_payload()
    d = [
        Directive(0, "no_charge_window", hours=(0, 1, 2)),
        Directive(1, "no_discharge_window", hours=(18, 19)),
        Directive(2, "max_grid_window", hours=(20, 21), max_grid_kwh=90),
    ]
    entries = [x.to_interpretation() for x in d]
    resp, _ = _run(payload, d)
    assert schedule_violations(payload, entries, resp, tol=1e-6) == []
    for e in resp["hourly_plan"]:
        if e["hour"] in (0, 1, 2):
            assert e["battery_action"] != "charge"
        if e["hour"] in (18, 19):
            assert e["battery_action"] != "discharge"
        if e["hour"] in (20, 21):
            assert e["grid_kwh"] <= 90 + 1e-6


def test_reserve_and_solar_reduction():
    payload = _flat_payload()
    d = [Directive(0, "minimum_battery_reserve", hours=(17, 18, 19), minimum_energy_kwh=150),
         Directive(1, "solar_reduction", hours=tuple(range(9, 15)), factor=1 / 3)]
    resp, _ = _run(payload, d)
    assert schedule_violations(payload, [x.to_interpretation() for x in d], resp, tol=1e-6) == []
    for e in resp["hourly_plan"]:
        if e["hour"] in (17, 18, 19):
            assert e["battery_energy_after_kwh"] >= 150 - 1e-6


def test_overlapping_directives_merge_conservatively():
    payload = _flat_payload()
    d = [Directive(0, "solar_reduction", hours=(12,), factor=0.5), Directive(1, "solar_reduction", hours=(12,), factor=0.5),
         Directive(2, "max_grid_window", hours=(19,), max_grid_kwh=100)]
    req = OptimizeRequest.model_validate(payload)
    p = build_problem(req.hours, req.battery, d)
    assert p.solar[12] == pytest.approx(payload["hours"][12]["solar_kwh"] * 0.25)
    d2 = [Directive(0, "max_grid_window", hours=(19,), max_grid_kwh=100), Directive(1, "max_grid_window", hours=(19,), max_grid_kwh=80)]
    assert build_problem(req.hours, req.battery, d2).g_max[19] == 80


def test_never_charge_and_discharge_same_hour_and_idle_is_zero():
    resp, _ = _run(_flat_payload(), [])
    for e in resp["hourly_plan"]:
        assert e["battery_action"] in ("charge", "discharge", "idle")
        if e["battery_action"] == "idle":
            assert e["battery_kwh"] == 0
        assert e["battery_kwh"] >= 0 and e["grid_kwh"] >= 0 and e["solar_used_kwh"] >= 0


def test_neutrality_exact_with_fractional_values():
    payload = _flat_payload(initial_energy_kwh=77.777, max_charge_kwh_per_hour=33.3333, max_discharge_kwh_per_hour=41.1111)
    resp, _ = _run(payload, [Directive(0, "solar_reduction", hours=(10, 11, 12, 13), factor=0.3333)])
    assert resp["hourly_plan"][-1]["battery_energy_after_kwh"] == pytest.approx(77.777, abs=1e-9)


def test_infeasible_directives_detected_and_relaxed():
    payload = _flat_payload(initial_energy_kwh=100)
    # Zero grid for 12 straight hours with a small battery cannot be met.
    d = [Directive(0, "max_grid_window", hours=tuple(range(8, 20)), max_grid_kwh=0)]
    req = OptimizeRequest.model_validate(payload)
    p = build_problem(req.hours, req.battery, d)
    assert solve(p) is None
    relaxed = solve_relaxed(p)
    assert relaxed is not None and relaxed.relaxed and relaxed.violations
    plan = build_hourly_plan(p, relaxed)
    resp = {"hourly_plan": plan, **plan_totals(plan, p.tariff)}
    # physical rules still hold when judged without the impossible directive
    assert schedule_violations(payload, [], resp, tol=1e-6) == []


def test_zero_rate_battery_and_zero_tariffs():
    payload = _flat_payload(max_charge_kwh_per_hour=0, max_discharge_kwh_per_hour=0)
    for h in payload["hours"]:
        h["tariff_bdt_per_kwh"] = 0
    resp, _ = _run(payload, [])
    assert all(e["battery_action"] == "idle" for e in resp["hourly_plan"])
    assert schedule_violations(payload, [], resp, tol=1e-6) == []


@pytest.mark.parametrize("seed", range(40))
def test_random_scenarios_always_valid(seed):
    rng = random.Random(seed)
    cap = rng.choice([100, 180, 250, 400])
    mn = rng.choice([0, 10, 30])
    init = rng.uniform(mn, cap)
    hours = [{"hour": h, "demand_kwh": round(rng.uniform(40, 260), 2), "solar_kwh": round(max(0, rng.gauss(60, 40)) if 6 <= h <= 18 else 0, 2),
              "tariff_bdt_per_kwh": round(rng.uniform(3, 35), 2)} for h in range(24)]
    rate = rng.choice([20, 45, 80])
    payload = {"scenario_id": f"R{seed}", "operator_notes": ["x"], "hours": hours,
               "battery": {"capacity_kwh": cap, "initial_energy_kwh": round(init, 3), "minimum_energy_kwh": mn,
                           "max_charge_kwh_per_hour": rate, "max_discharge_kwh_per_hour": rate}}
    start = rng.randint(0, 20)
    window = tuple(range(start, min(24, start + rng.randint(1, 4))))
    kind = rng.choice(["solar_reduction", "no_charge_window", "no_discharge_window", "max_grid_window", "minimum_battery_reserve"])
    extra = {"solar_reduction": {"factor": round(rng.random(), 3)}, "max_grid_window": {"max_grid_kwh": 400.0},
             "minimum_battery_reserve": {"minimum_energy_kwh": round(min(cap, payload["battery"]["initial_energy_kwh"]), 3)}}.get(kind, {})
    d = [Directive(0, kind, hours=window, **extra)]
    req = OptimizeRequest.model_validate(payload)
    p = build_problem(req.hours, req.battery, d)
    sol = solve(p)
    if sol is None:
        pytest.skip("random directive infeasible")
    plan = build_hourly_plan(p, sol)
    resp = {"hourly_plan": plan, **plan_totals(plan, p.tariff)}
    assert schedule_violations(payload, [x.to_interpretation() for x in d], resp, tol=1e-4) == []

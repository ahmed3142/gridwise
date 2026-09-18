"""The replay validator must catch every class of violation the judge checks."""

from __future__ import annotations

import copy

import pytest

from app.validator import compare_interpretation, interpretation_violations, response_violations, schedule_violations


@pytest.fixture()
def reference(public_cases):
    case = public_cases[5]
    return case["input"], case["expected_output"]


def test_reference_output_is_clean(reference):
    req, out = reference
    assert response_violations(req, out) == []


@pytest.mark.parametrize(
    "tamper, needle",
    [
        (lambda o: o["hourly_plan"][3].update(grid_kwh=o["hourly_plan"][3]["grid_kwh"] + 5), "energy balance"),
        (lambda o: o["hourly_plan"][3].update(battery_action="idle", battery_kwh=5), "idle"),
        (lambda o: o["hourly_plan"][14].update(battery_action="charge", battery_kwh=10,
                                               grid_kwh=o["hourly_plan"][14]["grid_kwh"] + 10), "no_charge_window"),
        (lambda o: o.update(total_cost_bdt=o["total_cost_bdt"] + 1), "total_cost_bdt"),
        (lambda o: o.update(peak_grid_kwh=1), "peak_grid_kwh"),
        (lambda o: o["hourly_plan"].pop(), "24"),
        (lambda o: o["hourly_plan"][10].update(solar_used_kwh=999, grid_kwh=0), "solar"),
        (lambda o: o["hourly_plan"][5].update(grid_kwh=-1), "non-negative"),
        (lambda o: o["hourly_plan"][23].update(battery_energy_after_kwh=0), "battery_energy_after_kwh"),
    ],
)
def test_detects_violations(reference, tamper, needle):
    req, out = reference
    bad = copy.deepcopy(out)
    tamper(bad)
    violations = schedule_violations(req, out["directive_interpretation"], bad)
    assert violations and any(needle in v for v in violations), violations


def test_neutrality_violation(reference):
    req, out = reference
    bad = copy.deepcopy(out)
    for e in bad["hourly_plan"]:
        if e["battery_action"] == "charge":
            e["battery_kwh"] += 1
            e["grid_kwh"] += 1
            break
    v = schedule_violations(req, out["directive_interpretation"], bad)
    assert any("final battery energy" in x for x in v)


def test_interpretation_shape_checks():
    assert interpretation_violations([{"note_index": 0, "applies": True, "directive_type": "no_op",
                                       "structured_adjustment": None, "explanation": ""}], 1)
    assert interpretation_violations([{"note_index": 0, "applies": True, "directive_type": "solar_reduction",
                                       "structured_adjustment": {"hours": [14, 13], "factor": 0.2}, "explanation": ""}], 1)
    assert interpretation_violations([{"note_index": 0, "applies": True, "directive_type": "made_up",
                                       "structured_adjustment": {}, "explanation": ""}], 1)
    assert interpretation_violations([], 1)


def test_compare_interpretation():
    exp = {"note_index": 0, "applies": True, "directive_type": "solar_reduction", "structured_adjustment": {"hours": [13, 14], "factor": 0.2}}
    ok = dict(exp, structured_adjustment={"hours": [13, 14], "factor": 0.205})
    assert all(compare_interpretation(ok, exp).values())
    wrong_hours = dict(exp, structured_adjustment={"hours": [13], "factor": 0.2})
    r = compare_interpretation(wrong_hours, exp)
    assert r["relevance"] and r["directive_type"] and not r["hours"] and r["values"]

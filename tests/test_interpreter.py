"""Deterministic interpretation layer: window->hours, unit conversions, guardrails, cross-checks."""

from __future__ import annotations

import pytest

from app.directives import Directive, guardrail_problems, interpretation_set_problems
from app.interpreter.crosscheck import crosscheck, numbers_in
from app.interpreter.normalize import normalize, window_hours
from app.interpreter.prompt import SYSTEM_PROMPT, user_message
from app.interpreter.semantic import SemanticInterpretation, TimeWindow
from app.schemas import Battery

BAT = Battery(capacity_kwh=200, initial_energy_kwh=120, minimum_energy_kwh=40,
              max_charge_kwh_per_hour=50, max_discharge_kwh_per_hour=50)


def W(sh, eh, sm=0, em=0):
    return TimeWindow(start_hour=sh, start_minute=sm, end_hour=eh, end_minute=em)


def S(dtype, windows, value=None, unit=None, applies=True, te="", qe=""):
    return SemanticInterpretation(
        time_evidence=te, quantity_evidence=qe, applies_to_schedule=applies, directive_type=dtype,
        time_windows=windows, quantity_value=value, quantity_unit=unit, explanation="x",
    )


@pytest.mark.parametrize(
    "window, hours",
    [
        (W(13, 15), [13, 14]),
        (W(12, 14), [12, 13]),
        (W(11, 14), [11, 12, 13]),
        (W(0, 24), list(range(24))),
        (W(0, 0), list(range(24))),
        (W(22, 2), [0, 1, 22, 23]),
        (W(23, 24), [23]),
        (W(13, 15, sm=30), [13, 14]),
        (W(13, 14, em=30), [13, 14]),
        (W(18, 21), [18, 19, 20]),
    ],
)
def test_window_hours(window, hours):
    got, errs = window_hours(window)
    assert not errs and sorted(got) == hours


def test_window_errors():
    assert window_hours(W(25, 3))[1]
    assert window_hours(W(10, 10))[1]


@pytest.mark.parametrize(
    "value, unit, factor",
    [(25, "percent_remaining", 0.25), (80, "percent_reduction", 0.2), (0.5, "fraction_remaining", 0.5),
     (0.75, "fraction_reduction", 0.25), (0, "percent_remaining", 0.0), (100, "percent_reduction", 0.0)],
)
def test_solar_units(value, unit, factor):
    d, problems = normalize(S("solar_reduction", [W(13, 15)], value, unit), BAT, 0, "note with % and numbers")
    assert not problems
    assert d.factor == pytest.approx(factor) and d.hours == (13, 14)


@pytest.mark.parametrize(
    "value, unit, kwh",
    [(120, "kwh", 120), (0.1, "mwh", 100), (50, "percent_of_capacity", 100), (0.25, "fraction_of_capacity", 50),
     (20, "kwh_above_minimum", 60), (90000, "wh", 90)],
)
def test_reserve_units(value, unit, kwh):
    d, problems = normalize(S("minimum_battery_reserve", [W(18, 21)], value, unit), BAT, 0, "keep 50% or kWh")
    assert not problems and d.minimum_energy_kwh == pytest.approx(kwh)


def test_grid_cap_units_and_zero():
    d, p = normalize(S("max_grid_window", [W(18, 21)], 155, "kwh"), BAT, 0)
    assert not p and d.max_grid_kwh == 155
    d, p = normalize(S("max_grid_window", [W(20, 22)], 0, "kwh"), BAT, 0)
    assert not p and d.max_grid_kwh == 0 and d.hours == (20, 21)


def test_no_op_and_irrelevant():
    d, p = normalize(S("no_op", [], applies=False), BAT, 2)
    assert d.directive_type == "no_op" and not d.applies and d.structured_adjustment() is None and not p
    d, _ = normalize(S("no_charge_window", [W(1, 2)], applies=False), BAT, 1)
    assert d.directive_type == "no_op"


def test_strict_problems_and_lenient_repairs():
    _, p = normalize(S("solar_reduction", [W(13, 15)], None, None), BAT, 0)
    assert p
    d, _ = normalize(S("solar_reduction", [W(13, 15)], None, None), BAT, 0, strict=False)
    assert d.factor == 0.0  # conservative: assume no usable solar
    _, p = normalize(S("minimum_battery_reserve", [W(18, 21)], 150, "percent_of_capacity"), BAT, 0)
    assert any("exceeds battery capacity" in x for x in p)
    d, _ = normalize(S("minimum_battery_reserve", [W(18, 21)], 150, "percent_of_capacity"), BAT, 0, strict=False)
    assert d.minimum_energy_kwh == 200
    _, p = normalize(S("solar_reduction", [W(13, 15)], 0.2, "percent_remaining"), BAT, 0, "leave one-fifth of output")
    assert p  # 0.2 with a percent unit while the note states a fraction
    _, p = normalize(S("no_charge_window", [], None, None), BAT, 0)
    assert p
    d, _ = normalize(S("no_charge_window", [], None, None), BAT, 0, strict=False)
    assert d.hours == tuple(range(24))
    _, p = normalize(S("max_grid_window", [W(1, 3)], 50, "percent_of_capacity"), BAT, 0)
    assert p


def test_guardrails():
    ok = Directive(0, "solar_reduction", hours=(13, 14), factor=0.2)
    assert guardrail_problems(ok, BAT) == []
    assert guardrail_problems(Directive(0, "solar_reduction", hours=(14, 13), factor=0.2), BAT)
    assert guardrail_problems(Directive(0, "solar_reduction", hours=(13, 13), factor=0.2), BAT)
    assert guardrail_problems(Directive(0, "solar_reduction", hours=(24,), factor=0.2), BAT)
    assert guardrail_problems(Directive(0, "solar_reduction", hours=(1,), factor=1.2), BAT)
    assert guardrail_problems(Directive(0, "solar_reduction", hours=(), factor=0.5), BAT)
    assert guardrail_problems(Directive(0, "minimum_battery_reserve", hours=(1,), minimum_energy_kwh=201), BAT)
    assert guardrail_problems(Directive(0, "minimum_battery_reserve", hours=(1,), minimum_energy_kwh=-1), BAT)
    assert guardrail_problems(Directive(0, "max_grid_window", hours=(1,), max_grid_kwh=float("inf")), BAT)
    assert guardrail_problems(Directive(0, "curtail_export", hours=(1,)), BAT)
    assert guardrail_problems(Directive(0, "no_op", hours=(1,)), BAT)
    assert interpretation_set_problems([Directive(1, "no_op"), Directive(0, "no_op")], 2)
    assert interpretation_set_problems([Directive(0, "no_op")], 2)


def test_adjustment_shapes():
    assert Directive(0, "solar_reduction", hours=(13, 14), factor=0.2).structured_adjustment() == {"hours": [13, 14], "factor": 0.2}
    assert Directive(0, "minimum_battery_reserve", hours=(18,), minimum_energy_kwh=100.0).structured_adjustment() == {"hours": [18], "minimum_energy_kwh": 100}
    assert Directive(0, "no_charge_window", hours=(2,)).structured_adjustment() == {"hours": [2]}
    assert Directive(0, "max_grid_window", hours=(18,), max_grid_kwh=155).structured_adjustment() == {"hours": [18], "max_grid_kwh": 155}


def test_numbers_in():
    assert {20.0, 13.0, 15.0} <= numbers_in("PV drops to 20% between 13:00 and 15:00")
    nums = numbers_in("Panel washing from one until three leaves roughly one-fifth of output")
    assert {1.0, 3.0, 0.2} <= nums
    assert 25.0 in numbers_in("about twenty-five percent")
    assert 0.75 in numbers_in("lose three quarters of the output")
    assert 1200.0 in numbers_in("limit of 1,200 kWh")


def test_crosscheck_flags_off_by_one_and_bad_numbers():
    note = "The charger is offline from 2 AM until 5 AM."
    good = S("no_charge_window", [W(2, 5)], te="from 2 AM until 5 AM")
    assert crosscheck(good, note) == []
    off_by_one = S("no_charge_window", [W(2, 4)], te="from 2 AM until 5 AM")
    assert crosscheck(off_by_one, note)
    note2 = "Keep at least 90 kWh in the battery from 6 PM until 10 PM."
    wrong_value = S("minimum_battery_reserve", [W(18, 22)], 80, "kwh", te="from 6 PM until 10 PM", qe="at least 90 kWh")
    assert crosscheck(wrong_value, note2)
    invented_quote = S("minimum_battery_reserve", [W(18, 22)], 90, "kwh", te="from 6 PM to 10 PM tonight", qe="90 kWh")
    assert crosscheck(invented_quote, note2)
    # remaining vs reduction conversions done by the model are acceptable
    note3 = "Expect an 80% reduction in rooftop solar between 11 AM and 2 PM."
    assert crosscheck(S("solar_reduction", [W(11, 14)], 20, "percent_remaining", te="between 11 AM and 2 PM", qe="an 80% reduction"), note3) == []
    # durations: "for three hours starting at 5 PM"
    note4 = "No charging for three hours starting at 5 PM."
    assert crosscheck(S("no_charge_window", [W(17, 20)], te="for three hours starting at 5 PM"), note4) == []


def test_prompt_is_static_and_mentions_all_types():
    for t in ("solar_reduction", "minimum_battery_reserve", "no_charge_window", "no_discharge_window", "max_grid_window", "no_op"):
        assert t in SYSTEM_PROMPT
    msg = user_message(["a", "b"], 1)
    assert 'Target note: [1] "b"' in msg


def test_prompt_examples_do_not_copy_public_notes(public_cases):
    for case in public_cases:
        for note in case["input"]["operator_notes"]:
            assert note not in SYSTEM_PROMPT


def test_crosscheck_flags_long_wraparound_ampm_slip():
    note = "Charging is disabled from 10 to 2 while the charger is serviced."
    slip = S("no_charge_window", [W(10, 2)], te="from 10 to 2")
    assert any("wraps past midnight" in i for i in crosscheck(slip, note))
    overnight = S("no_charge_window", [W(22, 2)], te="from 10 PM to 2 AM")
    assert crosscheck(overnight, "Charging is disabled from 10 PM to 2 AM.") == []

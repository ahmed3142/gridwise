"""Deterministic conversion of the LLM's semantic record into a validated `Directive`.

    windows (clock times, end as written)  ->  end-exclusive hour list, wrap-around aware
    quantity + unit                        ->  solar factor / reserve kWh / grid cap kWh

strict=True  returns every problem found (used to decide whether to re-ask the LLM).
strict=False applies conservative repairs so that a usable, guardrail-valid directive (or a no_op)
             always comes out, even after the LLM retry budget is exhausted.
"""

from __future__ import annotations

from ..directives import Directive, guardrail_problems, no_op
from ..schemas import Battery
from .semantic import SemanticInterpretation, TimeWindow

_SOLAR_UNITS = {"percent_remaining", "percent_reduction", "fraction_remaining", "fraction_reduction"}
_ENERGY_UNITS = {"kwh", "mwh", "wh"}
_RESERVE_UNITS = _ENERGY_UNITS | {"percent_of_capacity", "fraction_of_capacity", "kwh_above_minimum"}
_NEEDS_QUANTITY = {"solar_reduction", "minimum_battery_reserve", "max_grid_window"}


def window_hours(w: TimeWindow) -> tuple[set[int], list[str]]:
    """Hour slots whose [h:00, h+1:00) interval overlaps the window. Start-inclusive, end-exclusive;
    a window whose end is earlier than its start wraps past midnight."""
    sh, sm, eh, em = w.start_hour, w.start_minute, w.end_hour, w.end_minute
    label = f"{sh:02d}:{sm:02d}-{eh:02d}:{em:02d}"
    if not (0 <= sh <= 24 and 0 <= eh <= 24 and 0 <= sm <= 59 and 0 <= em <= 59):
        return set(), [f"time window {label} is outside the 24-hour clock"]
    start = (sh % 24) * 60 + (sm if sh < 24 else 0)
    end = eh * 60 + em
    if end > 24 * 60:
        return set(), [f"time window {label} ends after 24:00"]
    if start == end or (start == 0 and end == 0):
        if start == 0:
            return set(range(24)), []  # 00:00-00:00 / 00:00-24:00: whole day
        return set(), [f"time window {label} has zero length"]
    segments = [(start, end)] if end > start else [(start, 24 * 60), (0, end)]
    hours = {
        h
        for h in range(24)
        for a, b in segments
        if min(b, (h + 1) * 60) - max(a, h * 60) > 0
    }
    return hours, []


def _solar_factor(value: float, unit: str) -> float:
    if unit == "percent_remaining":
        return value / 100.0
    if unit == "percent_reduction":
        return 1.0 - value / 100.0
    if unit == "fraction_remaining":
        return value
    return 1.0 - value  # fraction_reduction


def _energy_kwh(value: float, unit: str, battery: Battery) -> float:
    if unit == "kwh":
        return value
    if unit == "mwh":
        return value * 1000.0
    if unit == "wh":
        return value / 1000.0
    if unit == "percent_of_capacity":
        return battery.capacity_kwh * value / 100.0
    if unit == "fraction_of_capacity":
        return battery.capacity_kwh * value
    return battery.minimum_energy_kwh + value  # kwh_above_minimum


def normalize(
    sem: SemanticInterpretation,
    battery: Battery,
    note_index: int,
    note_text: str = "",
    strict: bool = True,
) -> tuple[Directive, list[str]]:
    explanation = (sem.explanation or "").strip()[:300]
    if not sem.applies_to_schedule or sem.directive_type == "no_op":
        return no_op(note_index, explanation or "This note does not change today's energy schedule."), []

    dtype = sem.directive_type
    problems: list[str] = []

    hours: set[int] = set()
    for w in sem.time_windows:
        hs, errs = window_hours(w)
        hours |= hs
        problems += errs
    if not sem.time_windows:
        problems.append("an applicable directive needs at least one time window (use 00:00-24:00 for all day)")
        if not strict:
            hours = set(range(24))
    elif not hours and not strict:
        hours = set(range(24))

    factor = reserve = cap = None
    value, unit = sem.quantity_value, sem.quantity_unit
    if dtype in _NEEDS_QUANTITY:
        allowed = {
            "solar_reduction": _SOLAR_UNITS,
            "minimum_battery_reserve": _RESERVE_UNITS,
            "max_grid_window": _ENERGY_UNITS,
        }[dtype]
        if value is None or unit is None:
            problems.append(f"{dtype} needs quantity_value and quantity_unit")
        elif unit not in allowed:
            problems.append(f"quantity_unit {unit!r} is not valid for {dtype} (use one of {sorted(allowed)})")
        elif value < 0:
            problems.append(f"quantity_value must not be negative (got {value})")
        else:
            if unit.startswith("fraction") and value > 1:
                problems.append(f"{unit} must be between 0 and 1 (got {value}); use a percent unit for percentages")
            if unit.startswith("percent") and 0 < value < 1 and "%" not in note_text and "percent" not in note_text.lower():
                problems.append(
                    f"quantity_value {value} with unit {unit} means {value}%; if the note states a fraction, use a fraction_* unit"
                )
            if dtype == "solar_reduction":
                factor = round(_solar_factor(value, unit), 6)
            elif dtype == "minimum_battery_reserve":
                reserve = round(_energy_kwh(value, unit, battery), 6)
            else:
                cap = round(_energy_kwh(value, unit, battery), 6)
    elif value is not None or unit is not None:
        if strict:
            problems.append(f"{dtype} takes no quantity; quantity_value and quantity_unit must be null")

    if not strict:
        if dtype == "solar_reduction":
            if factor is None:
                factor = 0.0  # unknown remaining share: assume none (the plan stays valid for any true factor)
            factor = min(max(factor, 0.0), 1.0)
        elif dtype == "minimum_battery_reserve":
            if reserve is None:
                return no_op(note_index, "Reserve amount could not be determined; treated as no_op.", "fallback"), problems
            reserve = min(max(reserve, 0.0), battery.capacity_kwh)
        elif dtype == "max_grid_window" and cap is None:
            return no_op(note_index, "Grid limit could not be determined; treated as no_op.", "fallback"), problems

    directive = Directive(
        note_index=note_index,
        directive_type=dtype,
        hours=tuple(sorted(hours)),
        factor=factor,
        minimum_energy_kwh=reserve,
        max_grid_kwh=cap,
        explanation=explanation,
    )
    guard = guardrail_problems(directive, battery)
    problems += [g for g in guard if g not in problems]
    if not strict and guard:
        return no_op(note_index, "Interpretation failed validation; treated as no_op.", "fallback"), problems
    return directive, problems

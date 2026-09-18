"""Structured directives and the deterministic guardrails of Problem Statement section 08.

A `Directive` is the validated, machine-checkable form of one operator note. It is the ONLY thing the
optimizer ever sees: raw note text and raw LLM output never reach the math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .schemas import DIRECTIVE_TYPES, Battery

NUMERIC_DECIMALS = 6


def clean_number(value: float, decimals: int = NUMERIC_DECIMALS) -> float | int:
    """Round away float noise; return an int when the value is integral (e.g. 100.0 -> 100)."""
    rounded = round(float(value), decimals)
    if rounded == 0:
        return 0
    if float(rounded).is_integer():
        return int(rounded)
    return rounded


@dataclass(frozen=True)
class Directive:
    note_index: int
    directive_type: str
    hours: tuple[int, ...] = ()
    factor: float | None = None
    minimum_energy_kwh: float | None = None
    max_grid_kwh: float | None = None
    explanation: str = ""
    # Diagnostics only (never part of the API contract).
    source: str = "llm"
    notes: tuple[str, ...] = field(default=(), compare=False)

    @property
    def applies(self) -> bool:
        return self.directive_type != "no_op"

    def structured_adjustment(self) -> dict | None:
        hours = list(self.hours)
        if self.directive_type == "solar_reduction":
            return {"hours": hours, "factor": clean_number(self.factor)}
        if self.directive_type == "minimum_battery_reserve":
            return {"hours": hours, "minimum_energy_kwh": clean_number(self.minimum_energy_kwh)}
        if self.directive_type in ("no_charge_window", "no_discharge_window"):
            return {"hours": hours}
        if self.directive_type == "max_grid_window":
            return {"hours": hours, "max_grid_kwh": clean_number(self.max_grid_kwh)}
        return None

    def to_interpretation(self) -> dict:
        return {
            "note_index": self.note_index,
            "applies": self.applies,
            "directive_type": self.directive_type,
            "structured_adjustment": self.structured_adjustment(),
            "explanation": self.explanation or default_explanation(self),
        }


def no_op(note_index: int, explanation: str, source: str = "llm") -> Directive:
    return Directive(note_index=note_index, directive_type="no_op", explanation=explanation, source=source)


def format_hours(hours: tuple[int, ...] | list[int]) -> str:
    """[13, 14, 15, 20] -> '13-15, 20'."""
    hours = sorted(hours)
    if not hours:
        return "none"
    parts: list[str] = []
    start = prev = hours[0]
    for h in hours[1:]:
        if h == prev + 1:
            prev = h
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = h
    parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ", ".join(parts)


def default_explanation(d: Directive) -> str:
    hours = format_hours(d.hours)
    if d.directive_type == "solar_reduction":
        return f"Usable solar limited to {clean_number(d.factor)} of forecast in hours {hours}."
    if d.directive_type == "minimum_battery_reserve":
        return f"Battery must hold at least {clean_number(d.minimum_energy_kwh)} kWh after hours {hours}."
    if d.directive_type == "no_charge_window":
        return f"Battery charging is not allowed in hours {hours}."
    if d.directive_type == "no_discharge_window":
        return f"Battery discharging is not allowed in hours {hours}."
    if d.directive_type == "max_grid_window":
        return f"Grid import capped at {clean_number(d.max_grid_kwh)} kWh per hour in hours {hours}."
    return "This note does not change today's energy schedule."


# ----------------------------------------------------------------------------------------------
# Guardrails (Problem Statement section 08)
# ----------------------------------------------------------------------------------------------


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def guardrail_problems(d: Directive, battery: Battery) -> list[str]:
    """Return every section-08 violation for one directive (empty list = valid)."""
    problems: list[str] = []
    if d.directive_type not in DIRECTIVE_TYPES:
        return [f"unsupported directive_type {d.directive_type!r}"]

    if d.directive_type == "no_op":
        if d.hours or d.factor is not None or d.minimum_energy_kwh is not None or d.max_grid_kwh is not None:
            problems.append("no_op must not carry hours or numeric values")
        return problems

    hours = list(d.hours)
    if not hours:
        problems.append("hours must not be empty for an applicable directive")
    if any(not isinstance(h, int) or isinstance(h, bool) for h in hours):
        problems.append("hours must be integers")
    elif any(h < 0 or h > 23 for h in hours):
        problems.append("hours must be within 0-23")
    elif hours != sorted(set(hours)):
        problems.append("hours must be unique and ascending")

    if d.directive_type == "solar_reduction":
        if not _finite(d.factor) or not (0.0 <= d.factor <= 1.0):
            problems.append(f"solar factor must be finite and within [0, 1] (got {d.factor!r})")
    elif d.directive_type == "minimum_battery_reserve":
        if not _finite(d.minimum_energy_kwh) or d.minimum_energy_kwh < 0:
            problems.append(f"reserve must be finite and non-negative (got {d.minimum_energy_kwh!r})")
        elif d.minimum_energy_kwh > battery.capacity_kwh + 1e-9:
            problems.append(
                f"reserve {d.minimum_energy_kwh} kWh exceeds battery capacity {battery.capacity_kwh} kWh"
            )
    elif d.directive_type == "max_grid_window":
        if not _finite(d.max_grid_kwh) or d.max_grid_kwh < 0:
            problems.append(f"max_grid_kwh must be finite and non-negative (got {d.max_grid_kwh!r})")

    extra = {
        "solar_reduction": ("minimum_energy_kwh", "max_grid_kwh"),
        "minimum_battery_reserve": ("factor", "max_grid_kwh"),
        "no_charge_window": ("factor", "minimum_energy_kwh", "max_grid_kwh"),
        "no_discharge_window": ("factor", "minimum_energy_kwh", "max_grid_kwh"),
        "max_grid_window": ("factor", "minimum_energy_kwh"),
    }[d.directive_type]
    for name in extra:
        if getattr(d, name) is not None:
            problems.append(f"{d.directive_type} must not carry {name}")
    return problems


def interpretation_set_problems(directives: list[Directive], note_count: int) -> list[str]:
    """Exactly one entry per note, in note_index order 0..N-1 (section 05.1 / 08 note mapping)."""
    indices = [d.note_index for d in directives]
    if indices != list(range(note_count)):
        return [f"expected note_index order {list(range(note_count))}, got {indices}"]
    return []

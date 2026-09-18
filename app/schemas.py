"""Request and response contracts (Problem Statement sections 07 and 10).

Design choices:
* Unknown extra keys are IGNORED, never rejected: a judge request that carries an extra field must
  still be scored. Missing/ill-typed required fields are rejected (HTTP 400).
* Numbers must be real JSON numbers (strings such as "100" are rejected), finite, and non-negative.
  Booleans are never accepted as numbers.
* Cross-field battery consistency (initial energy inside [minimum, capacity]) is checked separately
  and returned as HTTP 422 (semantically invalid but well-formed).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

DIRECTIVE_TYPES: tuple[str, ...] = (
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
)
DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]
BatteryAction = Literal["charge", "discharge", "idle"]

# strict=True: JSON numbers only (ints are accepted for floats); no numeric strings, no booleans.
NonNegNumber = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
HourIndex = Annotated[int, Field(strict=True, ge=0, le=23)]


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore")


class HourEntry(_Lenient):
    hour: HourIndex
    demand_kwh: NonNegNumber
    solar_kwh: NonNegNumber
    tariff_bdt_per_kwh: NonNegNumber

    @field_validator("hour", mode="before")
    @classmethod
    def _integral_float_hour(cls, value: Any) -> Any:
        # Some JSON encoders emit 5.0 for an integer; accept it, but never bools or 5.5.
        if isinstance(value, float) and not isinstance(value, bool) and value.is_integer():
            return int(value)
        return value


class Battery(_Lenient):
    capacity_kwh: NonNegNumber
    initial_energy_kwh: NonNegNumber
    minimum_energy_kwh: NonNegNumber
    max_charge_kwh_per_hour: NonNegNumber
    max_discharge_kwh_per_hour: NonNegNumber


class OptimizeRequest(_Lenient):
    scenario_id: StrictStr
    operator_notes: Annotated[list[StrictStr], Field(min_length=1, max_length=3)]
    hours: Annotated[list[HourEntry], Field(min_length=24, max_length=24)]
    battery: Battery

    @field_validator("scenario_id")
    @classmethod
    def _scenario_id_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("scenario_id must be a non-empty string")
        return value

    @field_validator("operator_notes")
    @classmethod
    def _notes_not_blank(cls, notes: list[str]) -> list[str]:
        for index, note in enumerate(notes):
            if not note.strip():
                raise ValueError(f"operator_notes[{index}] must be a non-empty string")
        return notes

    @model_validator(mode="after")
    def _hours_cover_the_day(self) -> "OptimizeRequest":
        seen = [entry.hour for entry in self.hours]
        if sorted(seen) != list(range(24)):
            missing = sorted(set(range(24)) - set(seen))
            duplicates = sorted({h for h in seen if seen.count(h) > 1})
            raise ValueError(
                "hours must contain each hour 0-23 exactly once"
                + (f"; missing {missing}" if missing else "")
                + (f"; duplicated {duplicates}" if duplicates else "")
            )
        return self

    def sorted_hours(self) -> list[HourEntry]:
        return sorted(self.hours, key=lambda entry: entry.hour)


def semantic_problems(request: OptimizeRequest) -> list[str]:
    """Well-formed but physically inconsistent inputs (HTTP 422)."""
    b = request.battery
    problems: list[str] = []
    if b.minimum_energy_kwh > b.capacity_kwh:
        problems.append("battery.minimum_energy_kwh exceeds battery.capacity_kwh")
    if b.initial_energy_kwh > b.capacity_kwh:
        problems.append("battery.initial_energy_kwh exceeds battery.capacity_kwh")
    if b.initial_energy_kwh < b.minimum_energy_kwh:
        problems.append("battery.initial_energy_kwh is below battery.minimum_energy_kwh")
    return problems


# ----------------------------------------------------------------------------------------------
# Response models (documented in OpenAPI; responses are produced as plain dicts that follow them)
# ----------------------------------------------------------------------------------------------


class DirectiveInterpretation(BaseModel):
    note_index: int
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: dict[str, Any] | None
    explanation: str


class HourlyPlanEntry(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: BatteryAction
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: list[DirectiveInterpretation]
    hourly_plan: list[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: list[str] = []
    request_id: str | None = None

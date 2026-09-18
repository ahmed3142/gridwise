"""The schema the LLM fills for ONE operator note (sent as an OpenAI Structured Outputs schema).

The LLM reports WHAT the note says - clock times as written and the quantity with its unit.
Deterministic code (normalize.py) converts that into the machine-checkable structured_adjustment:
end-exclusive hour lists, solar factors, and kWh values. This split removes the classic LLM failure
modes (inclusive/exclusive off-by-one errors, "80% reduction" vs "80% remaining", "% of capacity").

Every field is required and has no default (OpenAI strict mode); optional values are nullable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

QuantityUnit = Literal[
    "percent_remaining",
    "percent_reduction",
    "fraction_remaining",
    "fraction_reduction",
    "kwh",
    "mwh",
    "wh",
    "percent_of_capacity",
    "fraction_of_capacity",
    "kwh_above_minimum",
]

SemanticDirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]


class TimeWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_hour: int = Field(description="24-hour clock hour when the condition starts, 0-23.")
    start_minute: int = Field(description="Minute of the start time, 0-59 (usually 0).")
    end_hour: int = Field(
        description="24-hour clock hour when the condition ENDS, exactly as written, 0-24 (24 = end of day)."
    )
    end_minute: int = Field(description="Minute of the end time, 0-59 (usually 0).")


class SemanticInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_evidence: str = Field(description='Exact words from the note that state the time period, or "" if none.')
    quantity_evidence: str = Field(
        description='Exact words from the note that state the amount/percentage/limit, or "" if none.'
    )
    applies_to_schedule: bool = Field(
        description="True only if the note imposes one of the five supported constraints on the scheduled day."
    )
    directive_type: SemanticDirectiveType
    time_windows: list[TimeWindow] = Field(
        description="Clock-time windows during which the constraint holds. Empty for no_op."
    )
    quantity_value: float | None = Field(
        description="The number as stated in the note (word numbers as digits); null when not applicable."
    )
    quantity_unit: QuantityUnit | None = Field(description="Unit/meaning of quantity_value; null when not applicable.")
    explanation: str = Field(description="One short sentence (max 25 words) explaining the interpretation.")

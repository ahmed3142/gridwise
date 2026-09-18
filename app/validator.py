"""Independent, judge-style replay of a finished response (Problem Statement section 11).

Deliberately written WITHOUT reusing the optimizer's data structures: it works on plain JSON-like
dicts (the request, a list of directive-interpretation entries, and the response), exactly like an
external judge would. It is used
  * inside the service, as a final self-check before every response is returned;
  * by the test-suite and by scripts/judge.py against the public sample expectations.
"""

from __future__ import annotations

import math
from typing import Any

DIRECTIVE_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}
ADJUSTMENT_KEYS = {
    "solar_reduction": {"hours", "factor"},
    "minimum_battery_reserve": {"hours", "minimum_energy_kwh"},
    "no_charge_window": {"hours"},
    "no_discharge_window": {"hours"},
    "max_grid_window": {"hours", "max_grid_kwh"},
}
PLAN_NUMERIC = ("grid_kwh", "solar_used_kwh", "battery_kwh", "battery_energy_after_kwh")
TOLERANCE = 0.01


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _as_dict(request: Any) -> dict:
    if isinstance(request, dict):
        return request
    return request.model_dump()  # pydantic model


# ----------------------------------------------------------------------------------------------
# Interpretation shape checks (sections 05.1, 08, 10.2, 11.1)
# ----------------------------------------------------------------------------------------------


def interpretation_violations(entries: Any, note_count: int, capacity: float | None = None) -> list[str]:
    v: list[str] = []
    if not isinstance(entries, list):
        return ["directive_interpretation must be an array"]
    if len(entries) != note_count:
        v.append(f"directive_interpretation has {len(entries)} entries for {note_count} notes")
    for pos, entry in enumerate(entries):
        where = f"directive_interpretation[{pos}]"
        if not isinstance(entry, dict):
            v.append(f"{where} must be an object")
            continue
        for key in ("note_index", "applies", "directive_type", "structured_adjustment", "explanation"):
            if key not in entry:
                v.append(f"{where} missing {key}")
        if entry.get("note_index") != pos:
            v.append(f"{where}.note_index must be {pos} (note_index order), got {entry.get('note_index')!r}")
        dtype = entry.get("directive_type")
        if dtype not in DIRECTIVE_TYPES:
            v.append(f"{where}.directive_type {dtype!r} is not supported")
            continue
        applies = entry.get("applies")
        adj = entry.get("structured_adjustment")
        if not isinstance(entry.get("explanation"), str):
            v.append(f"{where}.explanation must be a string")
        if dtype == "no_op":
            if applies is not False:
                v.append(f"{where}: no_op requires applies = false")
            if adj is not None:
                v.append(f"{where}: no_op requires structured_adjustment = null")
            continue
        if applies is not True:
            v.append(f"{where}: {dtype} requires applies = true")
        if not isinstance(adj, dict):
            v.append(f"{where}: {dtype} requires a structured_adjustment object")
            continue
        if set(adj.keys()) != ADJUSTMENT_KEYS[dtype]:
            v.append(f"{where}: {dtype} adjustment keys must be {sorted(ADJUSTMENT_KEYS[dtype])}, got {sorted(adj)}")
        hours = adj.get("hours")
        if (
            not isinstance(hours, list)
            or not hours
            or any(not isinstance(h, int) or isinstance(h, bool) or h < 0 or h > 23 for h in hours)
            or hours != sorted(set(hours))
        ):
            v.append(f"{where}: hours must be a non-empty list of unique ascending integers 0-23")
        if dtype == "solar_reduction":
            f = adj.get("factor")
            if not _num(f) or not (0 <= f <= 1):
                v.append(f"{where}: factor must be within [0, 1]")
        elif dtype == "minimum_battery_reserve":
            r = adj.get("minimum_energy_kwh")
            if not _num(r) or r < 0 or (capacity is not None and r > capacity + 1e-9):
                v.append(f"{where}: minimum_energy_kwh must be finite, >= 0 and <= capacity")
        elif dtype == "max_grid_window":
            g = adj.get("max_grid_kwh")
            if not _num(g) or g < 0:
                v.append(f"{where}: max_grid_kwh must be finite and >= 0")
    return v


# ----------------------------------------------------------------------------------------------
# Schedule replay (sections 09, 11.2, 11.3)
# ----------------------------------------------------------------------------------------------


def schedule_violations(request: Any, directives: list[dict], response: dict, tol: float = TOLERANCE) -> list[str]:
    """Replay `response` against `directives` (ground truth or our own) and every GridWise rule."""
    req = _as_dict(request)
    battery = req["battery"]
    hours_in = {int(e["hour"]): e for e in req["hours"]}
    capacity = float(battery["capacity_kwh"])
    initial = float(battery["initial_energy_kwh"])
    base_min = float(battery["minimum_energy_kwh"])
    max_c = float(battery["max_charge_kwh_per_hour"])
    max_d = float(battery["max_discharge_kwh_per_hour"])

    factor = [1.0] * 24
    lower = [base_min] * 24
    no_charge: set[int] = set()
    no_discharge: set[int] = set()
    cap = [math.inf] * 24
    for d in directives:
        dtype = _get(d, "directive_type")
        if dtype == "no_op" or not _get(d, "applies"):
            continue
        adj = _get(d, "structured_adjustment") or {}
        for h in adj.get("hours", []):
            if dtype == "solar_reduction":
                factor[h] *= float(adj["factor"])
            elif dtype == "minimum_battery_reserve":
                lower[h] = max(lower[h], float(adj["minimum_energy_kwh"]))
            elif dtype == "no_charge_window":
                no_charge.add(h)
            elif dtype == "no_discharge_window":
                no_discharge.add(h)
            elif dtype == "max_grid_window":
                cap[h] = min(cap[h], float(adj["max_grid_kwh"]))

    v: list[str] = []
    plan = response.get("hourly_plan")
    if not isinstance(plan, list) or len(plan) != 24:
        return ["hourly_plan must contain exactly 24 entries"]
    hours = [e.get("hour") if isinstance(e, dict) else None for e in plan]
    if sorted(h for h in hours if isinstance(h, int)) != list(range(24)) or len(set(hours)) != 24:
        return [f"hourly_plan hours must be exactly 0..23 once each, got {hours}"]
    if hours != list(range(24)):
        v.append("hourly_plan should be ordered by hour 0..23")
    by_hour = {e["hour"]: e for e in plan}

    energy = initial
    grids: list[float] = []
    for h in range(24):
        e = by_hour[h]
        bad = [k for k in PLAN_NUMERIC if not _num(e.get(k)) or e.get(k) < 0]
        if bad:
            v.append(f"hour {h}: {bad} must be finite non-negative numbers")
            continue
        action = e.get("battery_action")
        if action not in ("charge", "discharge", "idle"):
            v.append(f"hour {h}: battery_action {action!r} invalid")
            continue
        amount = float(e["battery_kwh"])
        grid = float(e["grid_kwh"])
        solar_used = float(e["solar_used_kwh"])
        grids.append(grid)
        if action == "idle" and amount != 0:
            v.append(f"hour {h}: idle requires battery_kwh = 0 (got {amount})")
        charge = amount if action == "charge" else 0.0
        discharge = amount if action == "discharge" else 0.0
        if charge > max_c + tol:
            v.append(f"hour {h}: charge {charge} exceeds max {max_c}")
        if discharge > max_d + tol:
            v.append(f"hour {h}: discharge {discharge} exceeds max {max_d}")
        if h in no_charge and charge > tol:
            v.append(f"hour {h}: charging inside a no_charge_window")
        if h in no_discharge and discharge > tol:
            v.append(f"hour {h}: discharging inside a no_discharge_window")
        energy = energy + charge - discharge
        if abs(energy - float(e["battery_energy_after_kwh"])) > tol:
            v.append(f"hour {h}: battery_energy_after_kwh {e['battery_energy_after_kwh']} != replayed {energy:.6f}")
        if energy < lower[h] - tol:
            v.append(f"hour {h}: battery energy {energy:.4f} below required minimum {lower[h]}")
        if energy > capacity + tol:
            v.append(f"hour {h}: battery energy {energy:.4f} above capacity {capacity}")
        effective = float(hours_in[h]["solar_kwh"]) * factor[h]
        if solar_used > effective + tol:
            v.append(f"hour {h}: solar_used {solar_used} exceeds effective solar {effective:.4f}")
        demand = float(hours_in[h]["demand_kwh"])
        if abs(grid + solar_used + discharge - demand - charge) > tol:
            v.append(f"hour {h}: energy balance off by {grid + solar_used + discharge - demand - charge:.6f}")
        if grid > cap[h] + tol:
            v.append(f"hour {h}: grid {grid} exceeds cap {cap[h]}")
    if abs(energy - initial) > tol:
        v.append(f"final battery energy {energy:.6f} != initial {initial}")

    if len(grids) == 24:
        tariffs = [float(hours_in[h]["tariff_bdt_per_kwh"]) for h in range(24)]
        total_grid = sum(grids)
        total_cost = sum(g * t for g, t in zip(grids, tariffs))
        peak = max(grids)
        for key, expected in (("total_grid_kwh", total_grid), ("total_cost_bdt", total_cost), ("peak_grid_kwh", peak)):
            got = response.get(key)
            if not _num(got) or abs(float(got) - expected) > tol:
                v.append(f"{key} {got!r} != recalculated {expected:.6f}")
    return v


def response_violations(request: Any, response: dict, tol: float = TOLERANCE) -> list[str]:
    """Top-level schema + interpretation shape + replay against the response's own directives."""
    req = _as_dict(request)
    v: list[str] = []
    for key in (
        "scenario_id",
        "directive_interpretation",
        "hourly_plan",
        "total_grid_kwh",
        "total_cost_bdt",
        "peak_grid_kwh",
        "plan_summary",
    ):
        if key not in response:
            v.append(f"response missing {key}")
    if response.get("scenario_id") != req["scenario_id"]:
        v.append("scenario_id does not echo the request")
    if not isinstance(response.get("plan_summary"), str):
        v.append("plan_summary must be a string")
    v += interpretation_violations(
        response.get("directive_interpretation"), len(req["operator_notes"]), float(req["battery"]["capacity_kwh"])
    )
    if not v:
        v += schedule_violations(req, response["directive_interpretation"], response, tol)
    return v


# ----------------------------------------------------------------------------------------------
# Interpretation accuracy vs. ground truth (Participant Guide section 07, category 1)
# ----------------------------------------------------------------------------------------------


def compare_interpretation(actual: dict, expected: dict, tol: float = TOLERANCE) -> dict[str, bool]:
    """Per-field agreement for one note: relevance, type, hours, numeric values/shape."""
    exp_type = expected["directive_type"]
    act_type = actual.get("directive_type")
    result = {
        "relevance": bool(actual.get("applies")) == bool(expected["applies"]),
        "directive_type": act_type == exp_type,
    }
    exp_adj = expected.get("structured_adjustment")
    act_adj = actual.get("structured_adjustment")
    if exp_adj is None:
        result["hours"] = act_adj is None
        result["values"] = act_adj is None
        return result
    if not isinstance(act_adj, dict):
        result["hours"] = False
        result["values"] = False
        return result
    result["hours"] = act_adj.get("hours") == exp_adj.get("hours")
    ok = act_type == exp_type and set(act_adj) == set(exp_adj)
    for key, value in exp_adj.items():
        if key == "hours":
            continue
        got = act_adj.get(key)
        ok = ok and _num(got) and abs(float(got) - float(value)) <= tol
    result["values"] = bool(ok)
    return result

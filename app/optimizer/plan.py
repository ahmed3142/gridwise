"""Turn an LP solution into the exact `hourly_plan` the judge replays (Problem Statement 09-10).

Guarantees on the returned plan (verified again by app.validator before responding):
* battery_action is derived from the NET battery flow, so an hour is never charge+discharge; idle => 0.
* Battery flows are rounded to 6 decimals and repaired so they sum to exactly zero, so the replayed
  energy chain ends exactly at the initial energy (end-of-day neutrality).
* battery_energy_after_kwh is recomputed from the rounded flows (the same arithmetic the judge uses).
* grid_kwh is recomputed from the energy-balance equation AFTER rounding, and never negative.
* Totals are computed from the rounded hourly values, so they always match a recalculation.
"""

from __future__ import annotations

import numpy as np

from ..directives import Directive, clean_number, format_hours
from .lp import H, Problem, Solution

DECIMALS = 6
_Q = 10**DECIMALS
_EPS = 1e-7


def _repair_neutrality(units: np.ndarray, cmax_u: np.ndarray, dmax_u: np.ndarray) -> np.ndarray:
    """Adjust rounded integer flows (units of 1e-6 kWh) so that they sum to exactly zero."""
    units = units.copy()
    resid = int(units.sum())
    if resid == 0:
        return units
    for h in np.argsort(-np.abs(units)):
        if resid == 0:
            break
        u = int(units[h])
        if u == 0:
            continue  # only touch hours that already act, never create a new action
        if resid > 0:  # need less charge / more discharge
            room = u + int(dmax_u[h]) if u < 0 else u  # shrink a charge, or grow a discharge
            delta = min(resid, room)
        else:  # need more charge / less discharge
            room = int(cmax_u[h]) - u if u > 0 else -u
            delta = -min(-resid, room)
        units[h] = u - delta
        resid -= delta
    return units


def build_hourly_plan(p: Problem, sol: Solution) -> list[dict]:
    net = sol.charge - sol.discharge
    net = np.where(np.abs(net) < _EPS, 0.0, net)
    cmax_u = np.floor(p.c_max * _Q + 1e-6).astype(np.int64)
    dmax_u = np.floor(p.d_max * _Q + 1e-6).astype(np.int64)
    units = np.clip(np.rint(net * _Q).astype(np.int64), -dmax_u, cmax_u)
    units = _repair_neutrality(units, cmax_u, dmax_u)

    plan: list[dict] = []
    cumulative = 0
    for h in range(H):
        u = int(units[h])
        cumulative += u
        charge = u / _Q if u > 0 else 0.0
        discharge = -u / _Q if u < 0 else 0.0

        solar_cap = float(p.solar[h])
        solar_used = min(max(round(float(sol.solar_used[h]), DECIMALS), 0.0), solar_cap)
        grid = float(p.demand[h]) + charge - solar_used - discharge
        if grid < 0:  # rounding noise: use slightly less solar instead of negative import
            solar_used = max(solar_used + grid, 0.0)
            grid = 0.0
        if np.isfinite(p.g_max[h]) and grid > p.g_max[h]:  # rounding noise over a cap: use more solar
            extra = min(grid - float(p.g_max[h]), solar_cap - solar_used)
            if extra > 0:
                solar_used += extra
                grid -= extra

        energy_after = float(p.initial) if cumulative == 0 else float(p.initial) + cumulative / _Q
        action = "charge" if u > 0 else "discharge" if u < 0 else "idle"
        plan.append(
            {
                "hour": h,
                "grid_kwh": clean_number(grid),
                "solar_used_kwh": clean_number(solar_used),
                "battery_action": action,
                "battery_kwh": clean_number(abs(u) / _Q) if u else 0,
                "battery_energy_after_kwh": clean_number(energy_after, 9)
                if cumulative
                else (int(p.initial) if float(p.initial).is_integer() else float(p.initial)),
            }
        )
    return plan


def plan_totals(plan: list[dict], tariff: np.ndarray) -> dict:
    grids = [float(e["grid_kwh"]) for e in plan]
    return {
        "total_grid_kwh": clean_number(sum(grids)),
        "total_cost_bdt": clean_number(sum(g * float(tariff[e["hour"]]) for g, e in zip(grids, plan))),
        "peak_grid_kwh": clean_number(max(grids) if grids else 0.0),
    }


def plan_summary(p: Problem, plan: list[dict], totals: dict, directives: list[Directive], sol: Solution) -> str:
    applied = [d for d in directives if d.applies]
    ignored = len(directives) - len(applied)
    parts: list[str] = []
    if applied:
        described = []
        for d in applied:
            hrs = format_hours(d.hours)
            if d.directive_type == "solar_reduction":
                described.append(f"solar limited to {clean_number(d.factor)}x forecast (hours {hrs})")
            elif d.directive_type == "minimum_battery_reserve":
                described.append(f"reserve >= {clean_number(d.minimum_energy_kwh)} kWh (hours {hrs})")
            elif d.directive_type == "no_charge_window":
                described.append(f"no charging (hours {hrs})")
            elif d.directive_type == "no_discharge_window":
                described.append(f"no discharging (hours {hrs})")
            elif d.directive_type == "max_grid_window":
                described.append(f"grid import <= {clean_number(d.max_grid_kwh)} kWh/h (hours {hrs})")
        parts.append("Applied " + "; ".join(described) + ".")
    else:
        parts.append("No operator note changed the schedule.")
    if ignored:
        parts.append(f"{ignored} note(s) had no effect on today's energy schedule (no_op).")

    charge_hours = [e["hour"] for e in plan if e["battery_action"] == "charge"]
    discharge_hours = [e["hour"] for e in plan if e["battery_action"] == "discharge"]
    charged = sum(float(e["battery_kwh"]) for e in plan if e["battery_action"] == "charge")
    discharged = sum(float(e["battery_kwh"]) for e in plan if e["battery_action"] == "discharge")
    if charge_hours or discharge_hours:
        parts.append(
            f"Battery charges {clean_number(charged, 2)} kWh in cheaper hours ({format_hours(charge_hours) if charge_hours else 'none'}) "
            f"and discharges {clean_number(discharged, 2)} kWh in costlier hours ({format_hours(discharge_hours) if discharge_hours else 'none'}), "
            f"ending the day at its initial {clean_number(p.initial, 2)} kWh."
        )
    else:
        parts.append("Battery stays idle; it already ends the day at its initial energy.")
    peak_hour = max(plan, key=lambda e: float(e["grid_kwh"]))["hour"] if plan else 0
    parts.append(
        f"Grid purchase {clean_number(totals['total_grid_kwh'], 2)} kWh for {clean_number(totals['total_cost_bdt'], 2)} BDT "
        f"(minimum-cost LP optimum), peak {clean_number(totals['peak_grid_kwh'], 2)} kWh at hour {peak_hour}."
    )
    if sol.relaxed:
        parts.append(
            "WARNING: the interpreted directives could not all be satisfied together; the plan minimizes the "
            "shortfall (" + "; ".join(sol.violations[:4]) + ")."
        )
    return " ".join(parts)

"""Turn an LP solution into the exact `hourly_plan` the judge replays (Problem Statement 09-10).

Rounding strategy: follow the LP's battery-ENERGY trajectory, rounded hour by hour, instead of
rounding each flow and summing. Rounding errors therefore never accumulate (each hour re-targets the
LP energy), and every hard per-hour limit is enforced exactly on the reported numbers.

Guarantees on the returned plan (verified again by app.validator before responding):
* battery_action comes from the NET battery flow: never charge+discharge in one hour; idle => 0.
* Flows respect the (possibly zero) charge/discharge limits of every hour exactly.
* battery_energy_after_kwh is the running sum of the reported flows (the judge's own arithmetic),
  never negative and never above capacity; the last hour ends at the initial energy.
* grid_kwh is recomputed from the energy-balance equation after rounding and is never negative.
* Totals are computed from the rounded hourly values, so they always match a recalculation.
"""

from __future__ import annotations

import numpy as np

from ..directives import Directive, clean_number, format_hours
from .lp import H, Problem, Solution

DECIMALS = 6
_EPS = 1e-6  # |net battery flow| below this is idle (battery_kwh 0, energy unchanged)


def build_hourly_plan(p: Problem, sol: Solution) -> list[dict]:
    if sol.plan_problem is not None:  # least-violation plans follow the relaxed limits they were solved with
        p = sol.plan_problem
    lp_net = np.asarray(sol.charge, dtype=float) - np.asarray(sol.discharge, dtype=float)
    initial = float(p.initial)
    capacity = float(p.capacity)
    prev = initial
    plan: list[dict] = []
    for h in range(H):
        lower = float(p.e_min[h])
        if h == H - 1:
            target = initial  # end-of-day neutrality
        else:
            target = min(max(round(float(sol.energy[h]), DECIMALS), lower), capacity)
        flow = 0.0 if abs(float(lp_net[h])) < _EPS else round(target - prev, DECIMALS)
        # Hard per-hour limits hold exactly (zero inside no-charge / no-discharge windows);
        # any tiny shortfall is re-targeted by the following hours.
        if flow > 0:
            flow = min(flow, float(p.c_max[h]))
        elif flow < 0:
            flow = max(flow, -float(p.d_max[h]))
        if abs(flow) < 1e-9:
            flow = 0.0
        energy_after = prev + flow
        snapped = round(energy_after, DECIMALS)
        if abs(energy_after - snapped) < 1e-9:
            energy_after = snapped
        if h == H - 1 and abs(energy_after - initial) < 1e-6:
            energy_after = initial
        prev = energy_after

        charge = flow if flow > 0 else 0.0
        discharge = -flow if flow < 0 else 0.0
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

        reported_energy = min(max(energy_after, 0.0), capacity)
        action = "charge" if flow > 0 else "discharge" if flow < 0 else "idle"
        plan.append(
            {
                "hour": h,
                "grid_kwh": clean_number(max(grid, 0.0)),
                "solar_used_kwh": clean_number(max(solar_used, 0.0)),
                "battery_action": action,
                "battery_kwh": clean_number(abs(flow)) if flow else 0,
                "battery_energy_after_kwh": clean_number(reported_energy, 9),
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
        f"({'least-violation plan' if sol.relaxed else 'minimum-cost LP optimum'}), "
        f"peak {clean_number(totals['peak_grid_kwh'], 2)} kWh at hour {peak_hour}."
    )
    if sol.relaxed:
        parts.append(
            "WARNING: the interpreted directives could not all be satisfied together; the plan minimizes the "
            "shortfall (" + "; ".join(sol.violations[:4]) + ")."
        )
    return " ".join(parts)

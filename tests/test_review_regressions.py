"""Regression tests for bugs found by the independent reviews (numerics, LLM layer, API)."""

from __future__ import annotations

import asyncio
import copy
import json

import pytest

from app.directives import Directive
from app.interpreter.crosscheck import crosscheck
from app.interpreter.normalize import normalize, window_hours
from app.interpreter.semantic import SemanticInterpretation, TimeWindow
from app.interpreter.service import InterpretationService
from app.main import settings
from app.optimizer.lp import build_problem, solve, solve_relaxed
from app.optimizer.plan import build_hourly_plan, plan_totals
from app.schemas import Battery, OptimizeRequest
from app.validator import response_violations, schedule_violations
from tests.conftest import make_client
from tests.fakes import FakeLLMClient

DEMAND = [225, 123, 120, 201, 57, 136, 196, 104, 241, 132, 50, 162, 133, 111, 218, 137, 204, 188, 190, 74, 247, 58, 62, 153]
SOLAR = [0, 0, 0, 0, 0, 0, 134, 0, 73, 163, 16, 55, 81, 78, 63, 123, 170, 115, 0, 0, 0, 0, 0, 0]
TARIFF = [12, 12, 20, 12, 12, 12, 20, 20, 20, 12, 20, 12, 12, 12, 12, 20, 20, 20, 12, 12, 12, 20, 12, 12]


def _payload(demand, solar, tariff, battery, notes=("The cafeteria menu changes tomorrow.",)):
    return {"scenario_id": "REG", "operator_notes": list(notes),
            "hours": [{"hour": h, "demand_kwh": demand[h], "solar_kwh": solar[h], "tariff_bdt_per_kwh": tariff[h]}
                      for h in range(24)],
            "battery": battery}


def test_battery_energy_never_negative_when_minimum_is_zero():
    pl = _payload(DEMAND, SOLAR, TARIFF, {"capacity_kwh": 209, "initial_energy_kwh": 159, "minimum_energy_kwh": 0,
                                          "max_charge_kwh_per_hour": 81, "max_discharge_kwh_per_hour": 79})
    req = OptimizeRequest.model_validate(pl)
    p = build_problem(req.hours, req.battery, [])
    plan = build_hourly_plan(p, solve(p))
    assert all(e["battery_energy_after_kwh"] >= 0 for e in plan)
    assert schedule_violations(pl, [], {"hourly_plan": plan, **plan_totals(plan, p.tariff)}, tol=1e-4) == []


def test_relaxed_plan_keeps_physical_rules():
    demand = [103, 212, 126, 69, 104, 127, 71, 102, 90, 110, 197, 228, 248, 161, 213, 133, 125, 90, 111, 216, 183, 176, 51, 242]
    solar = [0, 0, 0, 0, 0, 0, 138, 143, 127, 100, 35, 96, 5, 72, 154, 169, 58, 29, 0, 0, 0, 0, 0, 0]
    tariff = [7, 8, 27, 13, 15, 25, 12, 14, 17, 18, 30, 16, 8, 22, 17, 9, 11, 27, 16, 23, 29, 16, 18, 4]
    pl = _payload(demand, solar, tariff, {"capacity_kwh": 250, "initial_energy_kwh": 20, "minimum_energy_kwh": 20,
                                          "max_charge_kwh_per_hour": 20, "max_discharge_kwh_per_hour": 50})
    dirs = [Directive(0, "no_discharge_window", hours=(20,)),
            Directive(1, "max_grid_window", hours=(20, 21, 22, 23), max_grid_kwh=50),
            Directive(2, "minimum_battery_reserve", hours=(23,), minimum_energy_kwh=43)]
    req = OptimizeRequest.model_validate(pl)
    p = build_problem(req.hours, req.battery, dirs)
    assert solve(p) is None
    sol = solve_relaxed(p)
    plan = build_hourly_plan(p, sol)  # the builder itself follows the relaxed limits
    assert schedule_violations(pl, [], {"hourly_plan": plan, **plan_totals(plan, p.tariff)}, tol=1e-6) == []


def test_huge_rate_limits_rejected_not_garbage(api, sample_input):
    data = copy.deepcopy(sample_input)
    data["battery"]["max_charge_kwh_per_hour"] = 1e13
    assert api.post("/optimize-energy", json=data).status_code == 400
    data["battery"]["max_charge_kwh_per_hour"] = 1e8  # large but allowed: clamped internally
    r = api.post("/optimize-energy", json=data)
    assert r.status_code == 200 and response_violations(data, r.json()) == []


def test_conservative_rounding_falls_back_to_exact_values(api, public_cases):
    case = copy.deepcopy(public_cases[2])
    case["input"]["battery"]["initial_energy_kwh"] = 66.6667
    note = case["input"]["operator_notes"][0]
    fake = FakeLLMClient()
    ans = copy.deepcopy(fake.answers[note])
    ans["quantity_value"] = 66.6667
    ans["quantity_unit"] = "kwh"
    ans["quantity_evidence"] = ""
    ans["time_windows"] = [{"start_hour": 18, "start_minute": 0, "end_hour": 24, "end_minute": 0}]
    fake.answers[note] = ans
    seen_feedback = []
    original = fake.interpret

    async def spy(model, messages, timeout):
        seen_feedback.extend(m["content"] for m in messages if m["role"] == "user" and "impossible" in m["content"])
        return await original(model, messages, timeout)

    fake.interpret = spy
    client = make_client(fake)
    try:
        r = client.post("/optimize-energy", json=case["input"])
        body = r.json()
        assert r.status_code == 200 and r.headers.get("x-gridwise-degraded") is None
        assert "WARNING" not in body["plan_summary"]  # solved normally, not a least-violation plan
        assert seen_feedback == []  # no misleading "impossible" re-ask of a correct note
    finally:
        client.__exit__(None, None, None)


def test_deep_nesting_and_oversized_body_are_400(api):
    assert api.post("/optimize-energy", content=b"[" * 50000 + b"]" * 50000).status_code == 400
    assert api.post("/optimize-energy", content=b'{"a":"' + b"x" * 1_100_000 + b'"}').status_code == 400


def test_openapi_document_is_valid(api):
    spec = api.get("/openapi.json").json()
    assert "#/$defs/" not in json.dumps(spec)
    assert {"Battery", "HourEntry"} <= set(spec["components"]["schemas"])


def test_reinterpretation_is_not_cached_across_scenarios(public_cases):
    fake = FakeLLMClient()
    svc = InterpretationService(fake, settings)
    case = public_cases[6]  # reserve 90 kWh note
    notes = case["input"]["operator_notes"]
    battery = Battery.model_validate(case["input"]["battery"])

    async def run():
        loop = asyncio.get_running_loop()
        first = await svc.interpret(notes, battery, loop.time() + 20)
        bad = copy.deepcopy(fake.answers[notes[0]])
        bad["quantity_value"] = 60
        fake.first_answers = {notes[0]: bad}
        await svc.reinterpret(0, notes, battery, loop.time() + 20, ["infeasible"], previous=first[0].semantic)
        again = await svc.interpret(notes, battery, loop.time() + 20)
        return first, again

    first, again = asyncio.run(run())
    assert again[0].directive.minimum_energy_kwh == first[0].directive.minimum_energy_kwh == 90


def test_slow_llm_is_cut_by_the_deadline(sample_input):
    class Slow(FakeLLMClient):
        async def interpret(self, model, messages, timeout):
            await asyncio.sleep(60)  # ignores its timeout completely

    svc = InterpretationService(Slow(), settings)
    battery = Battery.model_validate(sample_input["battery"])

    async def run():
        loop = asyncio.get_running_loop()
        started = loop.time()
        res = await svc.interpret(sample_input["operator_notes"], battery, loop.time() + 6)
        return loop.time() - started, res

    elapsed, res = asyncio.run(run())
    assert elapsed < 6.5
    assert all(r.directive.directive_type == "no_op" and r.degraded for r in res)


def _sem(dtype, windows, value=None, unit=None, te="", qe=""):
    return SemanticInterpretation(time_evidence=te, quantity_evidence=qe, applies_to_schedule=dtype != "no_op",
                                  directive_type=dtype, time_windows=windows, quantity_value=value,
                                  quantity_unit=unit, explanation="x")


def W(sh, eh):
    return TimeWindow(start_hour=sh, start_minute=0, end_hour=eh, end_minute=0)


@pytest.mark.parametrize(
    "note, sem",
    [
        ("The charging circuit is offline this afternoon.", _sem("no_charge_window", [W(12, 17)], te="this afternoon")),
        ("Keep the battery at least one third full during the evening peak.",
         _sem("minimum_battery_reserve", [W(17, 21)], 0.3333, "fraction_of_capacity", "during the evening peak", "at least one third full")),
        ("Keep the battery fully charged from 6 PM to 9 PM.",
         _sem("minimum_battery_reserve", [W(18, 21)], 100, "percent_of_capacity", "from 6 PM to 9 PM", "fully charged")),
        ("No discharging in the two hours before midnight.", _sem("no_discharge_window", [W(22, 24)], te="the two hours before midnight")),
        ("No charging 6pm-9pm.", _sem("no_charge_window", [W(18, 21)], te="6 PM-9 PM")),
        ("Charging is blocked overnight from 8 PM to 9 AM.", _sem("no_charge_window", [W(20, 9)], te="from 8 PM to 9 AM")),
        ("The panels are offline from 10 AM to 2 PM.",
         _sem("solar_reduction", [W(10, 14)], 100, "percent_reduction", "from 10 AM to 2 PM", "offline")),
        ("Solar output will be at 3/4 of forecast from 1 PM to 4 PM.",
         _sem("solar_reduction", [W(13, 16)], 0.75, "fraction_remaining", "from 1 PM to 4 PM", "3/4 of forecast")),
    ],
)
def test_crosscheck_accepts_correct_answers(note, sem):
    assert crosscheck(sem, note) == []


def test_crosscheck_still_catches_real_slips():
    assert crosscheck(_sem("no_charge_window", [W(10, 2)], te="from 10 to 2"), "Charging is disabled from 10 to 2.")
    assert crosscheck(_sem("no_charge_window", [W(2, 4)], te="from 2 AM until 5 AM"), "Charger offline from 2 AM until 5 AM.")


def test_lenient_repairs_do_not_invent_constraints():
    bat = Battery(capacity_kwh=200, initial_energy_kwh=100, minimum_energy_kwh=20,
                  max_charge_kwh_per_hour=50, max_discharge_kwh_per_hour=50)
    assert window_hours(TimeWindow(start_hour=18, start_minute=0, end_hour=18, end_minute=0))[0] == {18}
    d, _ = normalize(_sem("solar_reduction", [W(13, 15)], 20, "fraction_remaining"), bat, 0, "20%", strict=False)
    assert d.factor == pytest.approx(0.2)


def test_keep_warm_calls_periodically_and_survives_errors():
    calls = []

    class Flaky(FakeLLMClient):
        async def interpret(self, model, messages, timeout):
            calls.append(model)
            if len(calls) == 1:
                raise RuntimeError("provider hiccup")
            return await super().interpret(model, messages, timeout)

    fake = Flaky(answers={"Charging is not allowed from 1 AM to 2 AM.": FakeLLMClient().answers[
        "The battery charger will be isolated from 2 AM until 5 AM for electrical maintenance."]})
    svc = InterpretationService(fake, settings)

    async def run():
        task = asyncio.create_task(svc.keep_warm(interval_s=0.05))
        await asyncio.sleep(0.4)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    assert len(calls) >= 3  # kept going after the first failure

"""API contract, HTTP codes, robustness and end-to-end public-sample behaviour (mock LLM)."""

from __future__ import annotations

import copy
import json

import pytest

from app.validator import compare_interpretation, response_violations, schedule_violations
from tests.conftest import make_client
from tests.fakes import FakeLLMClient

TOL = 0.01


def test_health_get_and_head(api):
    r = api.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    assert api.head("/health").status_code == 200


def test_root_and_version(api):
    assert api.get("/").status_code == 200
    body = api.get("/version").json()
    assert "prompt_version" in body and "api_key" not in json.dumps(body).lower()


@pytest.mark.parametrize("index", range(10))
def test_public_case_end_to_end(api, public_cases, index):
    case = public_cases[index]
    r = api.post("/optimize-energy", json=case["input"])
    assert r.status_code == 200, r.text
    body = r.json()
    expected = case["expected_output"]
    # 1) interpretation matches ground truth field by field
    assert len(body["directive_interpretation"]) == len(expected["directive_interpretation"])
    for got, exp in zip(body["directive_interpretation"], expected["directive_interpretation"]):
        assert all(compare_interpretation(got, exp).values()), (got, exp)
    # 2) response is internally consistent and schema-valid
    assert response_violations(case["input"], body) == []
    # 3) plan obeys the ORGANIZER ground-truth directives (not just our own)
    assert schedule_violations(case["input"], expected["directive_interpretation"], body) == []
    # 4) optimal: matches the reference optimal cost
    assert abs(body["total_cost_bdt"] - expected["total_cost_bdt"]) <= TOL
    assert body["scenario_id"] == case["input"]["scenario_id"]


def test_response_shape_exact(api, sample_input):
    body = api.post("/optimize-energy", json=sample_input).json()
    assert set(body) == {
        "scenario_id", "directive_interpretation", "hourly_plan", "total_grid_kwh",
        "total_cost_bdt", "peak_grid_kwh", "plan_summary",
    }
    for entry in body["directive_interpretation"]:
        assert set(entry) == {"note_index", "applies", "directive_type", "structured_adjustment", "explanation"}
        if entry["directive_type"] == "no_op":
            assert entry["applies"] is False and entry["structured_adjustment"] is None
    assert [e["hour"] for e in body["hourly_plan"]] == list(range(24))
    for e in body["hourly_plan"]:
        assert set(e) == {"hour", "grid_kwh", "solar_used_kwh", "battery_action", "battery_kwh", "battery_energy_after_kwh"}
        if e["battery_action"] == "idle":
            assert e["battery_kwh"] == 0


# ------------------------------------------------------------------ request validation -> 400 / 422


def _mutate(sample, fn):
    data = copy.deepcopy(sample)
    fn(data)
    return data


BAD_400 = {
    "missing_scenario": lambda d: d.pop("scenario_id"),
    "missing_notes": lambda d: d.pop("operator_notes"),
    "missing_battery": lambda d: d.pop("battery"),
    "missing_hours": lambda d: d.pop("hours"),
    "23_hours": lambda d: d["hours"].pop(),
    "25_hours": lambda d: d["hours"].append(dict(d["hours"][0])),
    "duplicate_hour": lambda d: d["hours"][1].update(hour=0),
    "hour_out_of_range": lambda d: d["hours"][23].update(hour=24),
    "negative_demand": lambda d: d["hours"][3].update(demand_kwh=-5),
    "string_number": lambda d: d["hours"][3].update(demand_kwh="100"),
    "bool_number": lambda d: d["hours"][3].update(solar_kwh=True),
    "null_tariff": lambda d: d["hours"][3].update(tariff_bdt_per_kwh=None),
    "no_notes": lambda d: d.update(operator_notes=[]),
    "four_notes": lambda d: d.update(operator_notes=["a", "b", "c", "d"]),
    "blank_note": lambda d: d.update(operator_notes=["   "]),
    "note_not_string": lambda d: d.update(operator_notes=[42]),
    "notes_not_list": lambda d: d.update(operator_notes="Keep 50 kWh"),
    "hours_not_list": lambda d: d.update(hours={"0": 1}),
    "battery_missing_field": lambda d: d["battery"].pop("capacity_kwh"),
    "negative_capacity": lambda d: d["battery"].update(capacity_kwh=-1),
    "blank_scenario": lambda d: d.update(scenario_id=" "),
    "numeric_scenario": lambda d: d.update(scenario_id=101),
}


@pytest.mark.parametrize("name", sorted(BAD_400))
def test_structurally_invalid_is_400(api, sample_input, name):
    r = api.post("/optimize-energy", json=_mutate(sample_input, BAD_400[name]))
    assert r.status_code == 400, (name, r.text)
    body = r.json()
    assert body["error"] and isinstance(body["details"], list)


@pytest.mark.parametrize(
    "raw",
    [b"{not json", b"", b"[]", b"null", b'"text"', b'{"scenario_id": NaN}', "{\"a\": \"é\"".encode()],
)
def test_malformed_json_is_400(api, raw):
    r = api.post("/optimize-energy", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_nan_number_is_400(api, sample_input):
    raw = json.dumps(sample_input).replace('"demand_kwh": 180', '"demand_kwh": NaN', 1)
    if "NaN" not in raw:
        raw = json.dumps(sample_input)
        data = json.loads(raw)
        data["hours"][0]["demand_kwh"] = float("nan")
        raw = json.dumps(data)
    r = api.post("/optimize-energy", content=raw.encode(), headers={"content-type": "application/json"})
    assert r.status_code == 400


@pytest.mark.parametrize(
    "fn",
    [
        lambda d: d["battery"].update(initial_energy_kwh=d["battery"]["capacity_kwh"] + 1),
        lambda d: d["battery"].update(initial_energy_kwh=0, minimum_energy_kwh=10),
        lambda d: d["battery"].update(minimum_energy_kwh=d["battery"]["capacity_kwh"] + 5),
    ],
)
def test_semantically_invalid_is_422(api, sample_input, fn):
    r = api.post("/optimize-energy", json=_mutate(sample_input, fn))
    assert r.status_code == 422, r.text


def test_extra_fields_are_ignored(api, sample_input):
    data = _mutate(sample_input, lambda d: d.update(comment="judge metadata", site="BUP"))
    data["hours"][0]["label"] = "midnight"
    data["battery"]["chemistry"] = "LFP"
    r = api.post("/optimize-energy", json=data)
    assert r.status_code == 200


def test_missing_content_type_and_shuffled_hours(api, sample_input):
    data = copy.deepcopy(sample_input)
    data["hours"] = list(reversed(data["hours"]))
    r = api.post("/optimize-energy", content=json.dumps(data).encode())
    assert r.status_code == 200
    body = r.json()
    assert [e["hour"] for e in body["hourly_plan"]] == list(range(24))
    assert response_violations(data, body) == []


def test_integral_float_hour_accepted(api, sample_input):
    data = copy.deepcopy(sample_input)
    data["hours"][5]["hour"] = 5.0
    assert api.post("/optimize-energy", json=data).status_code == 200


def test_unknown_route_and_wrong_method(api):
    assert api.get("/nope").status_code == 404
    assert api.get("/optimize-energy").status_code == 405


# ------------------------------------------------------------------ provider failure / degraded mode


@pytest.mark.parametrize("kind", ["timeout", "auth", "server", "rate_limit", "quota", "not_found", "refusal"])
def test_llm_failure_is_controlled(sample_input, kind):
    client = make_client(FakeLLMClient(fail_with=kind))
    try:
        r = client.post("/optimize-energy", json=sample_input)
        assert r.status_code == 200
        assert r.headers.get("x-gridwise-degraded") == "true"
        body = r.json()
        assert all(e["directive_type"] == "no_op" for e in body["directive_interpretation"])
        assert response_violations(sample_input, body) == []
        assert "Traceback" not in r.text
    finally:
        client.__exit__(None, None, None)


def test_no_llm_configured_still_serves(sample_input):
    client = make_client(None)
    try:
        r = client.post("/optimize-energy", json=sample_input)
        assert r.status_code == 200
        assert response_violations(sample_input, r.json()) == []
    finally:
        client.__exit__(None, None, None)


def test_reask_fixes_off_by_one_window(public_cases):
    case = public_cases[1]  # charger isolated 2 AM until 5 AM -> [2, 3, 4]
    note = case["input"]["operator_notes"][0]
    fake = FakeLLMClient()
    wrong = copy.deepcopy(fake.answers[note])
    wrong["time_windows"][0]["end_hour"] = 4  # classic "subtract an hour" slip
    fake.first_answers = {note: wrong}
    client = make_client(fake)
    try:
        body = client.post("/optimize-energy", json=case["input"]).json()
        assert body["directive_interpretation"][0]["structured_adjustment"]["hours"] == [2, 3, 4]
        assert len(fake.calls) == 2  # original + one re-ask
    finally:
        client.__exit__(None, None, None)


def test_cached_repeat_request_skips_llm(public_cases):
    fake = FakeLLMClient()
    client = make_client(fake)
    try:
        payload = public_cases[5]["input"]
        first = client.post("/optimize-energy", json=payload).json()
        calls = len(fake.calls)
        second = client.post("/optimize-energy", json=payload).json()
        assert len(fake.calls) == calls
        assert first == second
    finally:
        client.__exit__(None, None, None)


def test_infeasible_interpretation_triggers_reinterpretation(public_cases):
    """A misread reserve (percent read as kWh above capacity-feasible level) is repaired by re-asking."""
    case = copy.deepcopy(public_cases[2])  # reserve 50% of 200 kWh from 18-21
    note = case["input"]["operator_notes"][0]
    fake = FakeLLMClient()
    bad = copy.deepcopy(fake.answers[note])
    bad["time_windows"] = [{"start_hour": 0, "start_minute": 0, "end_hour": 24, "end_minute": 0}]
    bad["quantity_value"] = 95
    bad["quantity_unit"] = "percent_of_capacity"  # 190 kWh all day incl. hour 23 > initial 120 -> infeasible
    fake.first_answers = {note: bad}
    client = make_client(fake)
    try:
        body = client.post("/optimize-energy", json=case["input"]).json()
        assert body["directive_interpretation"][0]["structured_adjustment"] == {"hours": [18, 19, 20], "minimum_energy_kwh": 100}
        assert response_violations(case["input"], body) == []
    finally:
        client.__exit__(None, None, None)

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

# Tests never touch a real provider unless explicitly asked (tests/test_live.py, RUN_LIVE=1).
if os.environ.get("RUN_LIVE") != "1":
    os.environ["OPENAI_API_KEY"] = ""

from fastapi.testclient import TestClient  # noqa: E402

from app.interpreter.service import InterpretationService  # noqa: E402
from app.main import app, settings  # noqa: E402
from tests.fakes import FakeLLMClient  # noqa: E402

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def public_cases() -> list[dict]:
    return json.loads((DATA / "public_sample_cases.json").read_text(encoding="utf-8"))["cases"]


@pytest.fixture()
def sample_input(public_cases) -> dict:
    return copy.deepcopy(public_cases[5]["input"])


def make_client(fake: FakeLLMClient | None) -> TestClient:
    client = TestClient(app)
    client.__enter__()
    app.state.service = InterpretationService(fake, settings)
    return client


@pytest.fixture()
def api():
    client = make_client(FakeLLMClient())
    yield client
    client.__exit__(None, None, None)

"""Offline stand-ins for the OpenAI client (tests only)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.interpreter.llm import LLMCallError
from app.interpreter.semantic import SemanticInterpretation

DATA = Path(__file__).parent / "data"
_TARGET_RE = re.compile(r"Target note: \[(\d+)\] (.+)")


def target_note(messages: list[dict]) -> str:
    for message in messages:
        if message["role"] == "user":
            match = _TARGET_RE.search(message["content"])
            if match:
                return json.loads(match.group(2).splitlines()[0])
    raise AssertionError("no target note in messages")


def load_public_answers() -> dict[str, dict]:
    return json.loads((DATA / "public_semantic.json").read_text(encoding="utf-8"))["answers"]


class FakeLLMClient:
    """Answers from a {note text: semantic dict} table; can script failures and bad first answers."""

    provider = "fake"

    def __init__(self, answers: dict[str, dict] | None = None, fail_with: str | None = None,
                 first_answers: dict[str, dict] | None = None):
        self.answers = answers if answers is not None else load_public_answers()
        self.fail_with = fail_with
        self.first_answers = dict(first_answers or {})
        self.calls: list[tuple[str, str]] = []
        self.primary, self.fallback = "fake-primary", "fake-fallback"
        self.dead: set[str] = set()

    async def model_chain(self) -> list[str]:
        return [m for m in (self.primary, self.fallback) if m not in self.dead]

    def mark_dead(self, model: str) -> None:
        self.dead.add(model)

    async def resolve_models(self) -> None:
        return None

    async def aclose(self) -> None:
        return None

    async def interpret(self, model: str, messages: list[dict], timeout: float) -> SemanticInterpretation:
        note = target_note(messages)
        self.calls.append((model, note))
        if self.fail_with:
            raise LLMCallError(self.fail_with, "scripted failure", retryable=self.fail_with in ("timeout", "server"))
        if note in self.first_answers:
            return SemanticInterpretation.model_validate(self.first_answers.pop(note))
        if note not in self.answers:
            raise LLMCallError("invalid_output", f"no scripted answer for {note!r}", retryable=True)
        return SemanticInterpretation.model_validate(self.answers[note])

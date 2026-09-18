"""Deterministic consistency checks on the LLM's answer (a second channel, NOT an interpreter).

The LLM remains the only component that interprets a note. These checks only look for evidence
that the answer disagrees with the literal note text - a quoted phrase that is not in the note, a
number that appears nowhere in the note, or a window boundary that matches no time written in the
note (the typical "end hour minus one" slip). Any finding triggers ONE re-ask of the LLM with the
finding as feedback; the LLM's second answer is accepted.
"""

from __future__ import annotations

import re

from .semantic import SemanticInterpretation

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_FRACTIONS = {
    "half": 0.5, "halve": 0.5, "halved": 0.5, "halves": 0.5, "halving": 0.5, "third": 1 / 3, "thirds": 1 / 3, "quarter": 0.25, "quarters": 0.25,
    "fifth": 0.2, "fifths": 0.2, "tenth": 0.1, "tenths": 0.1,
}
_NUMBER_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")
_AMPM_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m\b\.?", re.IGNORECASE)
_CLOCK_RE = re.compile(r"\b([01]?\d|2[0-4])[:.]([0-5]\d)\b")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower().replace("’", "'"))


def numbers_in(text: str) -> set[float]:
    """Every number a note states: digits, word numbers (incl. 'twenty-five') and word fractions."""
    found: set[float] = set()
    for whole, frac in _NUMBER_RE.findall(text):
        found.add(float(whole.replace(",", "") + ("." + frac if frac else "")))
    tokens = _tokens(text)
    for i, tok in enumerate(tokens):
        if tok in _UNITS:
            found.add(float(_UNITS[tok]))
        if tok in _TENS:
            value = _TENS[tok]
            if i + 1 < len(tokens) and tokens[i + 1] in _UNITS and 0 < _UNITS[tokens[i + 1]] < 10:
                value += _UNITS[tokens[i + 1]]
            found.add(float(value))
        if tok == "hundred":
            prev = _UNITS.get(tokens[i - 1], 1) if i else 1
            found.add(float(prev * 100))
        if tok in _FRACTIONS:
            base = _FRACTIONS[tok]
            numerator = _UNITS.get(tokens[i - 1], 1) if i else 1
            if numerator == 0:
                numerator = 1
            found.add(base * numerator)
            found.add(base)
    return found


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 0.02 * max(1.0, abs(b)) and abs(a - b) <= 0.5


def quantity_issues(sem: SemanticInterpretation, note: str) -> list[str]:
    if sem.quantity_value is None or not sem.applies_to_schedule or sem.directive_type == "no_op":
        return []
    value = float(sem.quantity_value)
    if value == 0:
        return []  # "no solar", "no grid import" legitimately state zero without a digit
    stated = numbers_in(note)
    candidates: set[float] = set()
    for n in stated:
        candidates |= {n, n * 100, n / 100, 100 - n, 1 - n, n * 1000, n / 1000}
    if any(_close(value, c) for c in candidates):
        return []
    shown = ", ".join(f"{n:g}" for n in sorted(stated)) or "none"
    return [f"quantity_value {value:g} does not match any number stated in the note (numbers found: {shown})"]


def _explicit_clock_hours(note: str) -> set[int]:
    hours: set[int] = set()
    for h, _m, ap in _AMPM_RE.findall(note):
        h = int(h) % 12
        hours.add(h + 12 if ap.lower() == "p" else h)
    for h, _m in _CLOCK_RE.findall(note):
        hours.add(int(h))
    low = note.lower()
    if "noon" in low or "midday" in low or "mid-day" in low:
        hours.add(12)
    if "midnight" in low:
        hours |= {0, 24}
    return hours


_UNIT_NUMBER_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|percent\b|per\s*cent\b|kwh\b|kw\b|mwh\b|wh\b|bdt\b|taka\b)", re.IGNORECASE
)
_DURATION_RE = re.compile(
    r"\b(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)[\s-]*(?:hours?|hrs?|h)\b",
    re.IGNORECASE,
)


def _durations(note: str) -> set[int]:
    out: set[int] = set()
    for token in _DURATION_RE.findall(note):
        token = token.lower()
        value = _UNITS.get(token)
        if value is None:
            try:
                value = float(token)
            except ValueError:
                continue
        if float(value).is_integer():
            out.add(int(value))
    return out


def time_issues(sem: SemanticInterpretation, note: str) -> list[str]:
    if not sem.applies_to_schedule or sem.directive_type == "no_op" or not sem.time_windows:
        return []
    explicit = _explicit_clock_hours(note)
    # Numbers that are not part of an explicit clock time and not a quantity (%, kWh, ...).
    remainder = _CLOCK_RE.sub(" ", _AMPM_RE.sub(" ", _UNIT_NUMBER_RE.sub(" ", note)))
    bare = {int(n) for n in numbers_in(remainder) if float(n).is_integer() and 0 <= n <= 24}
    if not explicit and not bare:
        return []  # named periods ("evening") cannot be checked literally
    allowed = set(explicit) | {0, 24}
    slot_language = re.search(r"\binclusive\b|\bslots?\b|\bhours?\s+\d|\bhour\s+(?:index|number)", note, re.IGNORECASE)
    for b in bare:
        allowed |= {b, b % 24}
        if b <= 12:
            allowed |= {b + 12, (b + 12) % 24}
        if slot_language:  # "slots 18 through 20 inclusive" -> window ends at 21:00
            allowed |= {b + 1, (b + 1) % 24}
    starts = {w.start_hour for w in sem.time_windows}
    for s in starts:  # "for N hours starting at X"
        for n in _durations(note):
            allowed |= {s + n, (s + n) % 24}
    issues = []
    for w in sem.time_windows:
        start = w.start_hour * 60 + w.start_minute
        end = w.end_hour * 60 + w.end_minute
        if end < start and (24 * 60 - start + end) > 12 * 60:
            issues.append(
                f"window {w.start_hour:02d}:{w.start_minute:02d}-{w.end_hour:02d}:{w.end_minute:02d} wraps past "
                f"midnight and lasts {(24 * 60 - start + end) / 60:g} hours; check AM/PM"
            )
        for label, value in (("start", w.start_hour), ("end", w.end_hour)):
            if value not in allowed:
                issues.append(
                    f"window {label} {value:02d}:00 does not correspond to any time written in the note "
                    "(remember: report the end time exactly as written, do not subtract an hour)"
                )
    return issues


def evidence_issues(sem: SemanticInterpretation, note: str) -> list[str]:
    note_tokens = set(_tokens(note))
    issues = []
    for name in ("time_evidence", "quantity_evidence"):
        text = getattr(sem, name) or ""
        missing = [t for t in _tokens(text) if t not in note_tokens]
        if missing:
            issues.append(f"{name} {text!r} is not a verbatim quote from the note (unknown words: {missing[:5]})")
    return issues


def crosscheck(sem: SemanticInterpretation, note: str) -> list[str]:
    return evidence_issues(sem, note) + quantity_issues(sem, note) + time_issues(sem, note)

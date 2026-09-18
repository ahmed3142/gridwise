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
            base = prev * 100
            found.add(float(base))
            # "two hundred and fifty", "one hundred twenty five"
            j = i + 1
            if j < len(tokens) and tokens[j] == "and":
                j += 1
            rest = 0
            if j < len(tokens) and tokens[j] in _TENS:
                rest = _TENS[tokens[j]]
                if j + 1 < len(tokens) and tokens[j + 1] in _UNITS and 0 < _UNITS[tokens[j + 1]] < 10:
                    rest += _UNITS[tokens[j + 1]]
            elif j < len(tokens) and tokens[j] in _UNITS:
                rest = _UNITS[tokens[j]]
            if rest:
                found.add(float(base + rest))
        if tok in _FRACTIONS:
            base = _FRACTIONS[tok]
            numerator = _UNITS.get(tokens[i - 1], 1) if i else 1
            if numerator == 0:
                numerator = 1
            found.add(base * numerator)
            found.add(base)
    for num, den in re.findall(r"\b(\d+)\s*/\s*(\d+)\b", text):  # "3/4"
        if int(den):
            found.add(int(num) / int(den))
    return found


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 0.02 * max(1.0, abs(b)) and abs(a - b) <= 0.5


def quantity_issues(sem: SemanticInterpretation, note: str) -> list[str]:
    if sem.quantity_value is None or not sem.applies_to_schedule or sem.directive_type == "no_op":
        return []
    value = float(sem.quantity_value)
    if value == 0:
        return []  # "no solar", "no grid import" legitimately state zero without a digit
    if value in (1.0, 100.0) and re.search(
        r"\bfull(?:y)?\b|\btopped[- ]up\b|\bentire(?:ly)?\b|\bcompletely\b|\boffline\b|\bunavailable\b"
        r"|\bdisconnected\b|\bisolated\b|\bshut\b|\bzero\b|\bno (?:solar|pv|generation)\b",
        note.lower(),
    ):
        return []  # implied values the prompt mandates ("kept full" -> 100 percent_of_capacity)
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
    if re.search(r"\b(?:noon|midday|mid-day)\b", low):  # whole words: "afternoon" contains "noon"
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
    for minutes in re.findall(r"\b(\d+)\s*(?:minutes?|mins?)\b", note, re.IGNORECASE):  # "for 90 minutes"
        m = int(minutes)
        out |= {m // 60, -(-m // 60)}
    return out


# The prompt's own named periods are legitimate window boundaries.
_PERIODS = {"morning": (6, 12), "afternoon": (12, 17), "evening": (17, 21), "night": (21, 24),
            "tonight": (21, 24), "overnight": (0, 6)}


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
    for word, (a, b) in _PERIODS.items():
        if re.search(rf"\b{word}\b", note, re.IGNORECASE):
            allowed |= {a, b}
    durations = _durations(note)
    starts = {w.start_hour for w in sem.time_windows}
    for s in starts:  # "for N hours starting at X"
        for n in durations:
            allowed |= {s + n, (s + n) % 24}
    for e in explicit | {24}:  # "the two hours before midnight", "the last three hours of the day"
        for n in durations:
            allowed |= {e - n, (e - n) % 24}
    issues = []
    for w in sem.time_windows:
        start = w.start_hour * 60 + w.start_minute
        end = w.end_hour * 60 + w.end_minute
        both_written = w.start_hour in explicit and (w.end_hour in explicit or w.end_hour % 24 in explicit)
        if end < start and (24 * 60 - start + end) > 12 * 60 and not both_written:
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


def _evidence_tokens(text: str) -> list[str]:
    """Tokens that ignore formatting-only differences: '2 p.m.' ~ '2 PM', '6pm' ~ '6 PM', '1,200' ~ '1200'."""
    text = re.sub(r"\b([ap])\.\s*m\.?", r"\1m", text.lower())
    text = re.sub(r"(\d),(\d{3})\b", r"\1\2", text)
    text = re.sub(r"(\d)([a-z])", r"\1 \2", text)
    return [t for t in _tokens(text) if t not in {"to", "and", "until", "till", "from", "the"}]


def evidence_issues(sem: SemanticInterpretation, note: str) -> list[str]:
    if not sem.applies_to_schedule or sem.directive_type == "no_op":
        return []  # evidence of a no_op changes nothing in the output: never re-ask for it
    note_tokens = set(_evidence_tokens(note))
    issues = []
    for name in ("time_evidence", "quantity_evidence"):
        text = getattr(sem, name) or ""
        missing = [t for t in _evidence_tokens(text) if t not in note_tokens]
        if missing:
            issues.append(f"{name} {text!r} is not a verbatim quote from the note (unknown words: {missing[:5]})")
    return issues


_BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def crosscheck(sem: SemanticInterpretation, note: str) -> list[str]:
    literal = note.translate(_BANGLA_DIGITS)  # Bangla numerals are checked like ASCII digits
    return evidence_issues(sem, note) + quantity_issues(sem, literal) + time_issues(sem, literal)

"""Arithmetic grading with an explicit clock-hour answer contract."""

from __future__ import annotations

import re

import teacher_format
from solver_answers import extract_final_answer


def clock_question(question):
    request = re.split(r"(?<=[.?!])\s+", question.strip())[-1].casefold()
    if re.search(
        r"\b(?:how\s+long|how\s+many\s+(?:hours|minutes|seconds)|time\s+(?:taken|spent|needed|elapsed))\b", request
    ):
        return False
    return bool(
        re.search(
            r"\bwhat(?:\s+is|\s+would\s+be)?\s+(?:the\s+)?"
            r"(?:(?:earliest|latest|starting|finishing|arrival|departure)\s+)?time\b",
            request,
        )
    )


def clock_value(value):
    text = value.strip().strip("$")
    text = re.sub(r"\\(?:text|mathrm|textrm)\{([^{}]*)\}", r"\1", text)
    text = text.replace("{:}", ":")
    text = re.sub(r"\\[,!; ]|~", " ", text)
    text = re.sub(r"^\s*at\s+", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*o['’]clock\s*$", "", text, flags=re.IGNORECASE)
    match = re.fullmatch(r"(\d{1,2})(?::([0-5]\d))?\s*([ap]\.?m\.?)?\.?", text, re.IGNORECASE)
    if match is None:
        return None
    hour, minute = int(match[1]), int(match[2] or 0)
    period = match[3].replace(".", "").lower() if match[3] else None
    if hour > 23 or (period is not None and not 1 <= hour <= 12):
        return None
    return {"hour": hour, "minute": minute, "period": period, "explicit_minutes": match[2] is not None}


def grade_reference(question, reference, response):
    final = extract_final_answer(response)
    expected = clock_value(reference) if clock_question(question) else None
    actual = clock_value(final) if expected is not None and final else None
    if expected is None or actual is None:
        return teacher_format.grade_reference(question, reference, response)
    if expected["period"] is None and expected["hour"] <= 12:
        correct = expected["hour"] % 12 == actual["hour"] % 12 and expected["minute"] == actual["minute"]
        contract = "The numeric source reference gives a 12-hour clock value without an AM/PM distinction."
    else:

        def minutes(value):
            hour = value["hour"]
            if value["period"]:
                hour = hour % 12 + (12 if value["period"] == "pm" else 0)
            return 60 * hour + value["minute"]

        correct = minutes(expected) == minutes(actual)
        contract = "The source reference specifies a period or a 24-hour clock value."
    return {
        "correct": bool(correct),
        "status": "graded",
        "final_answer": final,
        "clock_reference": expected,
        "clock_prediction": actual,
        "clock_contract": contract,
        "original_reference": reference,
    }

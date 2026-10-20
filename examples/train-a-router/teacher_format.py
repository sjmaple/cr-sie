"""Question-only normalization of explicit final-answer formatting.

No dataset identities, benchmark values or solver correctness enter these rules.
Original answer strings and applied transformations remain in every receipt.
Removing an approximation marker does not introduce a numeric tolerance.
"""

from __future__ import annotations

import re

from benefit_expanded_gold_v2 import grade_teacher_reference
from solver_answers import _clean, _equivalent, _parse, extract_final_answer

PROTOCOL = "question-only-final-format-v1"
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_SPACING = re.compile(r"\\(?:quad|qquad)\b|\\[,;:! ]")
_MONEY = re.compile(r"\b(?:dollars?|cents?|nickels?|dimes?|coins?|price|costs?|tax|pay|paid|worth)\b", re.IGNORECASE)
_UNITS = (
    "degrees",
    "radians",
    "minutes per day",
    "miles per gallon",
    "miles per hour",
    "kilometers per hour",
    "kilometres per hour",
    "dollars",
    "cents",
    "kilometers",
    "kilometres",
    "hours",
    "minutes",
    "miles",
    "meters",
    "metres",
    "centimeters",
    "centimetres",
    "inches",
    "feet",
    "years",
    "cm",
    "km",
)


def _unit_name(unit: str) -> str:
    unit = " ".join(unit.lower().split()).replace("metre", "meter")
    unit = {"cm": "centimeter", "km": "kilometer", "inches": "inch", "feet": "foot"}.get(unit, unit)
    return unit.removesuffix("s")


def _math_prefix(value: str) -> bool:
    no_commands = re.sub(r"\\[A-Za-z]+", "", value)
    return value.count("{") == value.count("}") and re.search(r"[A-Za-z]{2,}", no_commands) is None


def answer_unit(question: str) -> str | None:
    """Infer the requested unit from the final request, never incidental facts."""
    question = re.sub(r"\[asy\].*?\[/asy\]", " ", question, flags=re.IGNORECASE | re.DOTALL)
    requests = [
        match
        for match in re.finditer(
            r"\b(?:how\s+(?:many|much|old|long)|what|which|find|determine|compute|give|calculate)\b",
            question,
            re.IGNORECASE,
        )
        if re.match(r"how\b|what\b", match[0], re.IGNORECASE)
        or re.search(r"(?:^|[.!?,;:]\s*|\band\s+)$", question[: match.start()], re.IGNORECASE)
    ]
    clause = question[requests[-1].start() :] if requests else question
    if requests:
        preceding = re.search(r"\bIn\s+([A-Za-z ]+),\s*$", question[: requests[-1].start()], re.IGNORECASE)
        if preceding:
            clause = preceding[0] + clause
        requested_variable = re.search(
            r"(?:values?\s+of|what\s+is|find|determine)"
            r"\s+\$?([A-Za-z]|\\(?:theta|alpha|beta|gamma|delta|phi|omega))\$?\s*[?.]",
            clause,
            re.IGNORECASE,
        )
        if requested_variable:
            variable = re.escape(requested_variable[1])
            prefix = question[: requests[-1].start()]
            unit_pattern = "|".join(re.escape(unit.removesuffix("s")) + "s?" for unit in _UNITS)
            adjacent = {
                _unit_name(match[1])
                for match in re.finditer(
                    r"(?<![A-Za-z])\$?" + variable + r"(?:\$\s*|\s+)(?i:(" + unit_pattern + r"))\b",
                    prefix,
                )
            }
            if re.search(r"(?<![A-Za-z])\$?" + variable + r"\s*\^\s*\{?\s*\\circ\s*\}?", prefix):
                adjacent.add("degree")
            if len(adjacent) == 1:
                unit = next(iter(adjacent))
                return {"degree": "angle_degree", "radian": "angle_radian"}.get(unit, unit)
            declarations = list(
                re.finditer(
                    r"(?i:\blet\s+)\$?" + variable + r"\$?(?i:\s+(?:be|denote|represent)\s+)([^.!?]+)",
                    prefix,
                )
            )
            if declarations:
                declaration = re.split(
                    r"(?i:\s+(?:and|while|whereas)\s+(?:let\s+)?\$?[A-Za-z]\$?\s+(?:be|denote|represent)\b)",
                    declarations[-1][1],
                    maxsplit=1,
                )[0]
                clause = declaration + ". " + clause
    if re.search(r"\b(?:what\s+is\s+the\s+first|which|what)\s+year\b[^?]*\?\s*$", question, re.IGNORECASE):
        return "year"
    if re.search(r"\bpercent(?:age)?\b", clause, re.IGNORECASE):
        return "percentage"
    head = re.sub(
        r"^(?:(?:what|which)(?:\s+(?:is|are))?|find|determine|compute|give|calculate)\s+",
        "",
        clause,
        flags=re.IGNORECASE,
    )
    head = re.sub(r"^(?:the|a|an)\s+", "", head, flags=re.IGNORECASE)
    if re.match(r"ratio\b", head, re.IGNORECASE):
        return "ratio"
    counting = re.match(r"(?:how\s+many|number\s+of)\s+(?!degrees?\b|radians?\b)", head, re.IGNORECASE)
    if not counting and re.search(r"\bangles?\b|\\angle\b", clause, re.IGNORECASE):
        if re.search(r"\bin\s+degrees?\b", clause, re.IGNORECASE):
            return "angle_degree"
        if re.search(r"\bin\s+radians?\b", clause, re.IGNORECASE):
            return "angle_radian"
        given_degrees = re.search(r"\\circ\b|°|\bdegrees?\b", question, re.IGNORECASE)
        given_radians = re.search(r"\bradians?\b", question, re.IGNORECASE)
        return "angle_inferred_degree" if given_degrees and not given_radians else "angle_unspecified"
    if re.search(r"\bratio\b", clause, re.IGNORECASE):
        return "ratio"
    if re.search(r"\b(?:what|which|first)\s+(?:positive\s+)?year\b", clause, re.IGNORECASE):
        return "year"
    for unit in _UNITS:
        singular = unit.removesuffix("s")
        pattern = (
            r"\b(?:how\s+many|number\s+of|in|rate\s+of)\s+(?:more\s+|additional\s+|total\s+)?"
            + re.escape(singular)
            + r"s?\b"
        )
        if re.search(pattern, clause, re.IGNORECASE):
            return _unit_name(unit)
    quantity = re.search(r"\b(?:how\s+many|number\s+of)\s+([A-Za-z ]+)", clause, re.IGNORECASE)
    if quantity:
        words = re.split(r"\s+", quantity[1].strip().lower())
        stop = {
            "is",
            "are",
            "was",
            "were",
            "did",
            "does",
            "do",
            "will",
            "would",
            "can",
            "must",
            "should",
            "of",
            "needed",
            "necessary",
            "required",
            "scheduled",
            "bought",
            "to",
            "for",
            "by",
            "after",
            "if",
            "in",
            "during",
            "before",
            "without",
            "with",
            "through",
            "over",
            "under",
            "between",
            "where",
            "whose",
            "which",
            "that",
            "from",
        }
        nouns = []
        for word in words:
            if word in stop:
                break
            if word not in {"more", "additional", "total"}:
                nouns.append(word)
        if nouns:
            return "count:" + _unit_name(nouns[-1])
    if re.search(r"\bareas?\b", clause, re.IGNORECASE):
        return "count:square unit"
    if re.search(r"\b(?:height|length|distance|radius|diameter|perimeter)\b", clause, re.IGNORECASE):
        length_units = {
            _unit_name(unit)
            for unit in _UNITS
            if _unit_name(unit) in {"meter", "kilometer", "centimeter", "mile", "inch", "foot"}
            and re.search(r"\b" + re.escape(unit.removesuffix("s")) + r"s?\b", question, re.IGNORECASE)
        }
        if len(length_units) == 1:
            return next(iter(length_units))
    if _MONEY.search(question) and re.search(
        r"\b(?:cost|worth|pay|price|bill|money|dollars?|cents?|total)\b", clause, re.IGNORECASE
    ):
        return "dollar"
    return None


def explicit_unit(value: str) -> tuple[str, str] | None:
    degree = re.fullmatch(r"(.+?)\s*(?:\^\s*\{?\s*\\circ\s*\}?|°)", value)
    if degree and _math_prefix(degree[1]):
        return degree[1].strip(), "degree"
    prefix = re.fullmatch(r"(?:\\\$|\$|\\textdollar\s*)\s*(.+)", value, re.DOTALL)
    if prefix and _math_prefix(prefix[1]):
        return prefix[1].strip(), "dollar"
    for unit in _UNITS:
        singular = unit.removesuffix("s")
        suffix = re.fullmatch(
            r"(.+?)\s*(?:\\(?:text|mathrm)\{\s*)?(" + re.escape(singular) + r"s?)\s*\}?", value, re.IGNORECASE
        )
        if suffix and _math_prefix(suffix[1]):
            return suffix[1].strip(), _unit_name(unit)
    # Unknown noun units must not be silently discarded by Math-Verify either.
    suffix = re.fullmatch(r"(.+?)(?:\\(?:text|mathrm)\{\s*([A-Za-z][A-Za-z -]+)\s*\}|\s+([A-Za-z][A-Za-z -]+))", value)
    if suffix and _math_prefix(suffix[1]):
        return suffix[1].strip(), "count:" + _unit_name(suffix[2] or suffix[3])
    return None


def normalize_final(question: str, final: str) -> dict:
    """Return a value and an auditable list of generic formatting operations."""
    value, steps = _clean(final), []
    clean = _SPACING.sub(" ", value).strip()
    if clean != value:
        value = clean
        steps.append("tex_spacing")
    approximate = re.match(r"^(?:\\approx(?![A-Za-z])|≈)\s*(.+)$", value, re.DOTALL)
    if approximate:
        value = approximate[1].strip()
        steps.append("approximation_marker_removed_without_tolerance")

    value, real_radical = _real_odd_radicals(question, value)
    if real_radical:
        steps.append("explicit_real_odd_radical_negative_radicand")

    binary_question = re.search(
        r"(?:^|[.!?]\s+|\bProblem:\s*)(?:must|can|could|does|do|is|are|will|would)\b[^?]*\?\s*$",
        question,
        re.IGNORECASE | re.DOTALL,
    )
    binary_answer = re.match(r"^(?:\\(?:text|mathrm)\{\s*)?(yes|no)\b(?:[.,;:]|\s|\}|$)", value, re.IGNORECASE)
    if binary_question and binary_answer:
        value = binary_answer[1].capitalize()
        steps.append("explicit_yes_no_question_leading_decision")

    # Only names may precede these assignments; 8=7 is never unwrapped.
    assignment = re.fullmatch(
        r"(?:[A-Za-z]|\(\s*[A-Za-z](?:\s*,\s*[A-Za-z])+\s*\))\s*(?:=|\\in\b|∈)\s*(.+)",
        value,
        re.DOTALL,
    )
    if assignment and assignment[1].startswith(("(", r"\{")):
        value = assignment[1].strip()
        steps.append("named_tuple_or_set_assignment")

    target, stated = answer_unit(question), explicit_unit(value)
    if target == "year":
        year = re.fullmatch(r"(?:\\(?:text|mathrm)\{\s*)?Year\s*\}?\s*(" + _NUMBER + r")", value, re.IGNORECASE)
        if year:
            value = year[1]
            stated = None
            steps.append("requested_year_prefix")
    unit_status = "no_explicit_unit"
    if target and target.startswith("angle_"):
        if stated and stated[1] not in {"degree", "radian"}:
            unit_status = "incompatible"
        else:
            scalar = stated[0] if stated else value
            degree = stated is not None and stated[1] == "degree"
            if stated is None:
                degree = target == "angle_degree" or (
                    target == "angle_inferred_degree" and not re.search(r"\\pi\b|π", scalar)
                )
            value = "(" + scalar + r")*\pi/180" if degree else scalar
            unit_status = "compatible" if stated else "no_explicit_unit"
            if degree:
                steps.append("angle_degrees_to_radians_exact")
            elif stated:
                steps.append("explicit_angle_radians")
            elif target == "angle_inferred_degree" and re.search(r"\\pi\b|π", scalar):
                steps.append("unspecified_angle_pi_expression_in_radians")
    elif stated:
        scalar, unit = stated
        if target is None:
            unit_status = "unknown_requested_unit"
        elif unit == target or {unit, target} <= {"cent", "dollar"}:
            unit_status = "compatible"
            value = scalar
            if unit == "cent":
                value = "(" + value + ")/100"
                steps.append("explicit_cents_to_dollars")
            else:
                steps.append("explicit_requested_unit:" + unit)
        else:
            unit_status = "incompatible"
    elif target == "cent" and re.fullmatch(_NUMBER, value):
        value = "(" + value + ")/100"
        steps.append("requested_cents_to_dollars")

    ratio = re.fullmatch(r"(" + _NUMBER + r")\s*:\s*(" + _NUMBER + r")", value)
    if ratio and target == "ratio":
        value = "(" + ratio[1] + ")/(" + ratio[2] + ")"
        steps.append("explicit_ratio_context")
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", value):
        value = value.replace(",", "")
        steps.append("whole_numeric_thousands_separators")
    return {
        "original": final,
        "normalized": value,
        "operations": steps,
        "protocol": PROTOCOL,
        "requested_unit": target,
        "explicit_unit": stated[1] if stated else None,
        "unit_status": unit_status,
    }


def normalize_reference(question: str, answer: str) -> tuple[str, list[str]]:
    result = normalize_final(question, answer)
    return result["normalized"], result["operations"]


def _real_odd_radicals(question: str, value: str) -> tuple[str, bool]:
    """Move a negative sign outside explicit odd radicals in a real-answer task.

    Arbitrary fractional powers retain their complex principal-value meaning.
    Only bounded, symbol-free radicands proved real and negative are changed.
    """
    if not re.search(r"\breal\b", question, re.IGNORECASE) or len(value) > 2000:
        return value, False
    from sympy import latex

    matches = list(re.finditer(r"\\sqrt\s*\[\s*(\d{1,2})\s*\]\s*\{", value))
    changed = False
    for match in reversed(matches[:20]):
        index = int(match[1])
        if index < 3 or index % 2 == 0:
            continue
        depth, end = 1, match.end()
        while end < len(value) and depth:
            depth += (value[end] == "{") - (value[end] == "}")
            end += 1
        if depth:
            continue
        radicand = value[match.end() : end - 1]
        try:
            parsed = _parse(radicand)
            if parsed["kind"] != "math" or len(parsed["value"]) != 1:
                continue
            expression = parsed["value"][0]
            if expression.free_symbols or expression.is_real is not True or expression.is_negative is not True:
                continue
            positive = latex(-expression)
        except Exception:  # noqa: BLE001, S112 — retain unsupported text for the bounded base grader.
            continue
        value = value[: match.start()] + r"(-\sqrt[" + str(index) + "]{" + positive + "})" + value[end:]
        changed = True
    return value, changed


def _complex_value(parsed: dict) -> dict:
    # Math-Verify assigns inconsistent real assumptions to bare i depending on
    # fraction spelling. In a question explicitly about complex numbers, i is
    # the imaginary unit in both answers. Preserve every other parsed symbol.
    from sympy import I

    if parsed["kind"] == "math":
        return parsed | {
            "value": [value.xreplace({s: I for s in value.free_symbols if s.name == "i"}) for value in parsed["value"]]
        }
    if parsed["kind"] in ("tuple", "set_of_tuples"):
        return parsed | {"value": [_complex_value(value) for value in parsed["value"]]}
    return parsed


def _same_base_integer_power(reference: str, prediction: str) -> bool | None:
    """Compare one identical integer base>1 using bounded integer exponents.

    Positive-integer power towers are evaluated only up to a fixed saturation
    bound. One saturated exponent and one exact smaller exponent prove unequal
    powers; two saturated, structurally different exponents stay unsupported.
    """
    cap = 1_000_000

    def parse(value):
        if not value or len(value) > 512 or re.search(r"\d\s+\d", value):
            raise ValueError("Unsupported power syntax")
        compact = re.sub(r"\s+", "", value)
        tokens = re.findall(r"\d+|[(){}^]", compact)
        if "".join(tokens) != compact or len(tokens) > 256:
            raise ValueError("Unsupported power syntax")
        index = 0

        def power(depth=0):
            nonlocal index
            if depth > 32 or index >= len(tokens):
                raise ValueError("Power syntax bound")
            token = tokens[index]
            index += 1
            if token.isdigit():
                node = int(token)
                if node < 1:
                    raise ValueError("Only positive integers")
            elif token in ("(", "{"):
                node = power(depth + 1)
                close = ")" if token == "(" else "}"
                if index >= len(tokens) or tokens[index] != close:
                    raise ValueError("Unbalanced power syntax")
                index += 1
            else:
                raise ValueError("Unsupported power atom")
            if index < len(tokens) and tokens[index] == "^":
                index += 1
                node = (node, power(depth + 1))
            return node

        result = power()
        if index != len(tokens):
            raise ValueError("Trailing power syntax")
        return result

    def capped(node):
        if isinstance(node, int):
            return min(node, cap)
        base, exponent = capped(node[0]), capped(node[1])
        result = 1
        while exponent:
            if exponent & 1:
                result = min(result * base, cap)
                if result == cap:
                    return cap
            exponent >>= 1
            base = min(base * base, cap)
        return result

    try:
        left, right = parse(reference), parse(prediction)
    except (ValueError, RecursionError):
        return None
    if not (
        isinstance(left, tuple)
        and isinstance(right, tuple)
        and isinstance(left[0], int)
        and left[0] > 1
        and left[0] == right[0]
    ):
        return None
    if left == right:
        return True
    a, b = capped(left[1]), capped(right[1])
    if a == b == cap:
        return None
    return a == b


def grade_reference(question: str, reference_final: str, solver_response: str) -> dict:
    """Compare two independent final answers using question-only formatting."""
    final = extract_final_answer(solver_response)
    reference = normalize_final(question, reference_final)
    prediction = normalize_final(question, final)
    receipts = {"reference": reference, "prediction": prediction}
    if (
        reference["unit_status"] == prediction["unit_status"] == "unknown_requested_unit"
        and reference["explicit_unit"] == prediction["explicit_unit"]
    ):
        # Equal explicit units can be compared without guessing an omitted
        # unit. A bare number paired with an unknown unit remains unresolved.
        for record in receipts.values():
            record["normalized"] = explicit_unit(record["normalized"])[0]
            record["unit_status"] = "same_explicit_units"
            record["operations"].append("same_explicit_units_on_both_answers")
    if any(record["unit_status"] in {"unknown_requested_unit", "incompatible"} for record in receipts.values()):
        return {
            "correct": False,
            "status": "graded"
            if all(record["unit_status"] != "unknown_requested_unit" for record in receipts.values())
            else "unit_contract_unresolved",
            "final_answer": final,
            "format_normalization": receipts,
            "unit_check": "Explicit answer units must match the requested quantity; no incidental-question-unit stripping",
        }
    powers = _same_base_integer_power(reference["normalized"], prediction["normalized"])
    if powers is not None:
        return {
            "correct": powers,
            "status": "graded",
            "final_answer": final,
            "format_normalization": receipts,
            "integer_power_guard": "Identical integer base>1; exact or saturated positive-integer exponent comparison",
        }
    result = grade_teacher_reference(
        question,
        reference["normalized"],
        r"\boxed{" + prediction["normalized"] + "}" if final else solver_response,
    )
    if (
        final
        and result.get("exact_numeric_guard") != "applied"
        and re.search(r"\bcomplex\s+numbers?\b", question, re.IGNORECASE)
    ):
        try:
            result = result | {
                "correct": _equivalent(
                    _complex_value(_parse(reference["normalized"])), _complex_value(_parse(prediction["normalized"]))
                ),
                "complex_context": "Bare i is the imaginary unit in both parsed answers",
                "status": "graded",
            }
        except Exception as error:  # noqa: BLE001 — preserve the bounded base parser's failure result.
            result = result | {"complex_context_error": type(error).__name__}
    return result | {"final_answer": final, "format_normalization": {"reference": reference, "prediction": prediction}}

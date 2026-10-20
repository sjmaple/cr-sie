"""Source-only adjudications fixed before inspecting any solver answers.

Rules bind to both row ID and original answer. Two numeric transcription errors
are corrected; the other rules declare notation/semantics already established
by the source problem and solution. No evaluation rows are removed.
"""

from __future__ import annotations

import re
from fractions import Fraction
from itertools import pairwise

from data import SOURCE_HASHES, digest, file_hash
from solver_answers import extract_final_answer, grade_answer, preflight_gold

PROTOCOL = "source-only-gold-adjudication-v2"
# Original answer, canonical answer, semantic type. Full source reasoning and
# problem/solution hashes accompany the published gold-adjudication artifact.
RULES = {
    "e2h-amc-train-977": ("6:0", "6:00", "clock_time"),
    "e2h-amc-eval-1694": (r"1{,}10{,}0", "1010000", "source_correction"),
    "e2h-amc-train-945": ("(7,9)", "(7,9)", "ordered_tuple"),
    "e2h-amc-eval-2468": ("(8,2)", "(8,2)", "ordered_tuple"),
    "e2h-amc-eval-1731": ("-6,0,2,8", r"\{-6,0,2,8\}", "finite_set"),
    "e2h-amc-eval-718": ("100,110,11", r"\{11,100,110\}", "finite_set"),
    "e2h-amc-train-711": ("1346,2016,2019", r"\{1346,2016,2019\}", "finite_set"),
    "e2h-amc-eval-2077": ("(-1,2)", "(-1,2)", "ordered_tuple"),
    "e2h-amc-train-285": ("1,2,3,4", r"\{1,2,3,4\}", "finite_set"),
    "e2h-amc-train-684": ("1:2", r"\frac{1}{2}", "ratio"),
    "e2h-amc-eval-111": ("(2,2),(12,17),(17,12)", r"\{(2,2),(12,17),(17,12)\}", "set_of_ordered_tuples"),
    "e2h-amc-eval-1815": ("2:3", r"\frac{2}{3}", "ratio"),
    "e2h-amc-eval-2927": (r"2 \sqrt{2}, 4+2 \sqrt{2}", r"\{2\sqrt{2},4+2\sqrt{2}\}", "finite_set"),
    "e2h-amc-eval-1773": (r"0, \pm i \sqrt{6}", r"\{-i\sqrt{6},0,i\sqrt{6}\}", "finite_set"),
    "e2h-amc-eval-1728": (
        "(12,3,4),(6,5.4,7.6)",
        r"\{(12,3,4),(6,\frac{27}{5},\frac{38}{5})\}",
        "set_of_ordered_tuples",
    ),
    "e2h-amc-train-645": (r"0, \frac{\sqrt{5}-1}{2}", r"\{0,\frac{\sqrt{5}-1}{2}\}", "finite_set"),
    "e2h-amc-eval-769": ("7,15", r"\{7,15\}", "finite_set"),
    "e2h-amc-eval-2738": (r"0,3,6,9, \ldots, 6048", r"\{3k\mid k\in\mathbb{Z},0\leq k\leq2016\}", "progression"),
    "e2h-amc-eval-2152": ("(-1030, -990)", "(-1030,-990)", "ordered_tuple"),
    "e2h-amc-eval-620": (r"21{,}0", "21000", "source_correction"),
    "e2h-amc-eval-2906": ("4:1", "4", "ratio"),
    # These seven questions explicitly request a percentage (or the integer N
    # in N percent), unlike other questions merely containing percentage data.
    "e2h-amc-train-56": (r"80 \%", r"80\%", "percentage"),
    "e2h-amc-eval-2677": ("40", r"40\%", "percentage"),
    "e2h-amc-train-146": ("78", r"78\%", "percentage"),
    "e2h-amc-train-140": ("54", r"54\%", "percentage"),
    "e2h-amc-eval-55": ("200", r"200\%", "percentage"),
    "e2h-amc-eval-812": ("75", r"75\%", "percentage"),
    "e2h-amc-eval-1803": ("19", r"19\%", "percentage"),
}
CORRECTION_REASONS = {
    "e2h-amc-eval-1694": "The double sum is 100*5050 + 100*5050 = 1010000; the source lost zero digits.",
    "e2h-amc-eval-620": "The logarithmic product simplifies to (20*21/2)*100 = 21000; the source lost zero digits.",
}


def _clock_minutes(value: str) -> int | None:
    value = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\[,;! ]|~", " ", value).strip().casefold()
    value = re.sub(r"\b([ap])\s*\.\s*m\.?", r"\1m", value)
    value = re.sub(r"\s+in the (?:afternoon|evening)$", " pm", value)
    value = re.sub(r"\s+in the morning$", " am", value)
    match = re.fullmatch(r"(\d{1,2})(?::(\d{1,2}))?\s*(am|pm)?", value)
    if match is None:
        return None
    hour, minute, meridiem = int(match[1]), int(match[2] or 0), match[3]
    if hour > 23 or minute > 59 or (meridiem and not 1 <= hour <= 12):
        return None
    if meridiem:
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
    elif hour < 12:
        # This source question explicitly starts at noon and asks about later
        # the same day. A bare hour therefore uses its afternoon interpretation.
        hour += 12
    return 60 * hour + minute


def _compact(value: str) -> str:
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = re.sub(r"\\(?:ldots|cdots|dots)|…", "...", value)
    value = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\mathbb\s*\{?Z\}?", "Z", value)
    for old, new in ((r"\leq", "≤"), (r"\le", "≤"), (r"\in", "∈"), (r"\mid", ":"), (r"\cdot", "*")):
        value = value.replace(old, new)
    value = re.sub(r"\\[,;! ]|\s+", "", value)
    if value.startswith(r"\{") and value.endswith(r"\}"):
        value = value[2:-2]
    elif value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    return value


def _progression(value: str) -> tuple[int, int, int] | None:
    """Recognize bounded integer progressions, without evaluating expressions."""
    compact = _compact(value)
    listed = re.fullmatch(r"((?:-?\d+,){2,32})\.\.\.,(-?\d+)", compact)
    if listed:
        prefix = [int(part) for part in listed[1].rstrip(",").split(",")]
        start, step, last = prefix[0], prefix[1] - prefix[0], int(listed[2])
        if (
            step > 0
            and all(b - a == step for a, b in pairwise(prefix))
            and last >= prefix[-1]
            and (last - start) % step == 0
        ):
            return start, step, last
        return None
    builder = re.fullmatch(r"(-?\d+)\*?([a-z])[:|](.+)", compact)
    if builder:
        coefficient, variable, domain = int(builder[1]), builder[2], builder[3]
        if coefficient <= 0:
            return None
        # Integer domain is mandatory: a real interval is not a discrete set.
        member = variable + "∈Z"
        if domain.startswith(member + ","):
            bounds = domain[len(member) + 1 :]
        elif domain.endswith("," + member):
            bounds = domain[: -len(member) - 1]
        else:
            bounds = ""
        bound = re.fullmatch(r"(-?\d+)(≤|<)" + variable + r"(≤|<)(-?\d+)", bounds)
        if bound:
            low = int(bound[1]) + int(bound[2] == "<")
            high = int(bound[4]) - int(bound[3] == "<")
            return (coefficient * low, coefficient, coefficient * high) if low <= high else None
        indices = re.fullmatch(variable + r"=(0,1,(?:2,)?\.\.\.,\d+)", domain)
        if indices:
            return 0, coefficient, coefficient * int(domain.rsplit(",", 1)[1])
    divisible = re.fullmatch(r"([a-z])∈Z[:|](-?\d+)(≤|<)\1(≤|<)(-?\d+),(\d+)[:|]\1", compact)
    if divisible:
        low = int(divisible[2]) + int(divisible[3] == "<")
        high = int(divisible[5]) - int(divisible[4] == "<")
        step = int(divisible[6])
        if step > 0:
            first, last = -(-low // step) * step, (high // step) * step
            return (first, step, last) if first <= last else None
    words = re.sub(r"\s+", " ", value.casefold().strip())
    natural = re.fullmatch(
        r"(?:all )?multiples of (\d+) (?:from (-?\d+) to|between (-?\d+) and) (-?\d+)( inclusive)?", words
    )
    if natural and (natural[2] is not None or natural[5]):
        step, start, last = int(natural[1]), int(natural[2] or natural[3]), int(natural[4])
        if step > 0 and start <= last and start % step == 0 and last % step == 0:
            return start, step, last
    return None


def _special(correct: bool, final: str, kind: str, expected, observed) -> dict:
    return {
        "correct": correct,
        "status": "graded" if final else "no_final_answer",
        "final_answer": final,
        "gold": {"kind": kind, "value": expected},
        "prediction": {"kind": kind, "value": observed},
    }


def grade_reviewed(identity: str, originalgold: str, response: str) -> dict:
    """Use the fixed semantic rule for this source row, preserving every row."""
    if identity not in RULES:
        return grade_answer(originalgold, response)
    original, canonical, kind = RULES[identity]
    if originalgold != original:
        raise ValueError("Adjudication does not match the original source answer")
    final = extract_final_answer(response)
    if kind == "clock_time":
        observed = _clock_minutes(final) if final else None
        result = _special(observed == 18 * 60, final, kind, 18 * 60, observed)
    elif kind == "progression":
        observed = _progression(final) if final else None
        result = _special(observed == (0, 3, 6048), final, kind, (0, 3, 6048), observed)
    elif kind == "percentage":
        # In "what percentage?", 40 and 40% both mean forty percent. Explicit
        # 0.4% must still be rejected. Do not apply this convention to counts.
        expected = canonical if "%" in final else canonical.removesuffix(r"\%")
        result = grade_answer(expected, response)
    else:
        if kind == "ratio":
            ratio = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*:\s*([+-]?\d+(?:\.\d+)?)", final)
            if ratio:
                if not Fraction(ratio[2]):
                    result = _special(False, final, kind, canonical, "zero denominator")
                    return result | {"adjudication": {"protocol": PROTOCOL, "id": identity, "kind": kind}}
                normalized = Fraction(ratio[1]) / Fraction(ratio[2])
                response = "\\boxed{" + str(normalized) + "}"
        result = grade_answer(canonical, response)
    return result | {"adjudication": {"protocol": PROTOCOL, "id": identity, "kind": kind}}


def reviewed_gold(gold: dict[str, str], source=None) -> tuple[dict, dict]:
    """Apply only source-bound rules and return explicit approval/coverage metadata."""
    if source is not None:
        for name, expected in SOURCE_HASHES.items():
            if file_hash(source / name) != expected:
                raise ValueError("Gold adjudications require the pinned source")
    corrected = dict(gold)
    approved, corrections = [], []
    for identity, original in gold.items():
        if identity not in RULES:
            continue
        expected, canonical, kind = RULES[identity]
        if original != expected:
            raise ValueError(f"Source gold changed for adjudicated row: {identity}")
        measured = grade_reviewed(identity, original, "\\boxed{" + canonical + "}")
        if measured["status"] != "graded" or not measured["correct"]:
            raise ValueError(f"Canonical reviewed answer cannot be graded: {identity}")
        corrected[identity] = canonical
        approved.append(identity)
        corrections.append(
            {
                "id": identity,
                "original": original,
                "canonical": canonical,
                "kind": kind,
                "source_answer_correction": kind == "source_correction",
                "reason": CORRECTION_REASONS.get(
                    identity,
                    "Source question explicitly asks for a percentage; an omitted percent sign is allowed."
                    if kind == "percentage"
                    else "Source problem and solution establish this answer type.",
                ),
            }
        )
    preflight = preflight_gold(gold)
    unresolved = {key: value for key, value in preflight["needs_review"].items() if key not in approved}
    return corrected, {
        "protocol": PROTOCOL,
        "source_hashes": dict(SOURCE_HASHES),
        "original_gold_sha256": digest(gold),
        "canonical_gold_sha256": digest(corrected),
        "approved_ids": sorted(approved),
        "correction_records": corrections,
        "n": len(gold),
        "excluded": [],
        "ready": not unresolved,
        "needs_review": unresolved,
    }

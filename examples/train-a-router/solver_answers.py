"""Join pinned solver golds and grade explicit final answers without model judging.

Run ``preflight_gold`` before collecting solver outputs. Its review list is an
audit gate, not permission to exclude difficult-to-grade problems. Math-Verify
uses signal timeouts: grade on the main thread, or in a separate process.
"""

from __future__ import annotations

import importlib.metadata
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

from data import SOURCE_HASHES, SOURCE_REVISION, audit, digest, file_hash, render_request

MAX_ANSWER_CHARS = 4096
MAX_RESPONSE_CHARS = 100_000
PARSING_SECONDS = 2
GRADER_VERSION = "explicit-final-math-verify-v3"
_BOX = re.compile(r"\\(?:boxed|fbox)\s*\{")
_FINAL = re.compile(
    r"(?im)(?=(?:^|(?<=[.!?,;])\s+)(?:(?:therefore|thus|hence|so)[,:]?\s+)?"
    r"(?:\*\*)?(?:the\s+)?(?:final\s+answer|answer)"
    r"(?:\*\*)?\s*(?::|=|\bis\b)\s*(.+)$)"
)


def prepare_inputs(source: Path, dataset: dict) -> tuple[dict, dict, dict]:
    """Return solver-only inputs, separate answers and checked source provenance."""
    import pyarrow.parquet as pq

    audit(dataset)
    if dataset.get("source_revision") != SOURCE_REVISION:
        raise ValueError("Dataset source revision differs from the pinned source")
    originals = {}
    for name, expected in SOURCE_HASHES.items():
        path = source / name
        if file_hash(path) != expected:
            raise ValueError(f"Source hash differs: {name}")
        rows = pq.read_table(path, columns=["problem", "answer", "tag", "year"]).to_pylist()
        originals[name.split(".")[0]] = rows

    inputs, gold = {}, {}
    for split in ("validation", "calibration", "test"):
        inputs[split] = []
        for row in dataset["splits"][split]:
            match = re.fullmatch(r"e2h-amc-(train|eval)-(0|[1-9]\d*)", row["id"])
            if match is None or int(match[2]) >= len(originals[match[1]]):
                raise ValueError("Unknown pinned-source row ID")
            original = originals[match[1]][int(match[2])]
            if (
                row["request"] != render_request(original["tag"], original["problem"])
                or row["contest"] != original["tag"]
                or row["year"] != original["year"]
            ):
                raise ValueError(f"Source/request join differs: {row['id']}")
            answer = original["answer"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Missing source answer: {row['id']}")
            inputs[split].append({"id": row["id"], "request": row["request"]})
            gold[row["id"]] = answer
    return (
        inputs,
        gold,
        {
            "source_revision": SOURCE_REVISION,
            "source_hashes": dict(SOURCE_HASHES),
            "dataset_sha256": digest(dataset),
            "inputs_sha256": digest(inputs),
            "gold_sha256": digest(gold),
            "counts": {split: len(rows) for split, rows in inputs.items()},
        },
    )


def _boxed_body(text: str, opening: int) -> str | None:
    depth = 1
    for index in range(opening + 1, len(text)):
        if index and text[index - 1] == "\\":
            continue
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if not depth:
                return text[opening + 1 : index]
    return None


def _boxes(text: str) -> list[tuple[int, str | None]]:
    result, end = [], -1
    for match in _BOX.finditer(text):
        if match.start() < end:
            continue
        body = _boxed_body(text, match.end() - 1)
        result.append((match.start(), body))
        if body is None:
            break
        end = match.end() + len(body) + 1
    return result


def _clean(text: str) -> str:
    text = text.strip().removeprefix("**").removesuffix("**").strip()
    if text.endswith("."):
        text = text[:-1].rstrip()
    for left, right in (("$$", "$$"), ("$", "$"), (r"\[", r"\]"), (r"\(", r"\)")):
        if text.startswith(left) and text.endswith(right):
            text = text[len(left) : -len(right)].strip()
            break
    return text.replace(r"\left", "").replace(r"\right", "").strip()


def _answer_only(text: str) -> bool:
    if "\n" in text or len(text) > MAX_ANSWER_CHARS:
        return False
    # A single name/Yes/No is an answer; a sentence containing a number is not.
    if re.fullmatch(r"[A-Za-z]+", text):
        return True
    words = re.findall(r"(?<!\\)\b[A-Za-z]{2,}\b", re.sub(r"\\[A-Za-z]+", "", text))
    return not words


def extract_final_answer(response: str) -> str:
    """Extract the last explicit answer, or an answer-only response.

    Balanced boxes preserve nested fractions. An unfinished final box does not
    fall back to an earlier intermediate answer. Unmarked derivations return "".
    """
    if not isinstance(response, str) or len(response) > MAX_RESPONSE_CHARS:
        return ""
    candidates = _boxes(response)
    for match in _FINAL.finditer(response):
        value = match[1].strip()
        box = _boxes(value)
        if box:
            value = box[-1][1]
        candidates.append((match.start(), value))
    if candidates:
        value = max(candidates, key=lambda item: item[0])[1]
        answer = _clean(value) if value is not None else ""
    else:
        answer = _clean(response)
        if not _answer_only(answer):
            return ""
    return answer if 0 < len(answer) <= MAX_ANSWER_CHARS else ""


def _parts(text: str) -> list[str]:
    depth, start, parts = 0, 0, []
    for index, character in enumerate(text):
        if character in "({[":
            depth += 1
        elif character in ")}]":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced answer delimiters")
        elif character == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    if depth:
        raise ValueError("Unbalanced answer delimiters")
    return parts + [text[start:].strip()]


def _text_answer(text: str) -> str | None:
    match = re.fullmatch(r"\\(?:text|mathrm)\{([A-Za-z][A-Za-z -]*)\}", text)
    if match:
        return " ".join(match[1].casefold().split())
    if re.fullmatch(r"[A-Z][a-z]+", text) or text.casefold() in ("yes", "no"):
        return text.casefold()
    return None


def _parse(text: str, depth: int = 0) -> dict:
    from math_verify import LatexExtractionConfig, parse

    text = _clean(text)
    if not text or len(text) > MAX_ANSWER_CHARS or depth > 4 or "$" in text:
        raise ValueError("Invalid answer length or nested math environment")
    named = _text_answer(text)
    if named is not None:
        return {"kind": "text", "value": named}
    tuple_container = text
    if text.startswith(r"\{") and text.endswith(r"\}"):
        tuple_container = text[2:-2].strip()
    parts = _parts(tuple_container)
    if len(parts) > 1 and all(part.startswith("(") and part.endswith(")") for part in parts):
        if len(parts) > 32:
            raise ValueError("Too many answer tuples")
        return {"kind": "set_of_tuples", "value": [_parse(part, depth + 1) for part in parts]}
    if text.startswith("(") and text.endswith(")"):
        parts = _parts(text[1:-1])
        if len(parts) > 1:
            if len(parts) > 32 or any(not part for part in parts):
                raise ValueError("Invalid tuple")
            return {"kind": "tuple", "value": [_parse(part, depth + 1) for part in parts]}
    ratio = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*:\s*([+-]?\d+(?:\.\d+)?)", text)
    if ratio:
        # A score such as 6:0 must not become an infinity or an equivalent 3:0.
        return {"kind": "ratio", "value": (Fraction(ratio[1]), Fraction(ratio[2]))}
    # Math-Verify deliberately accepts 9% == 9. Keep mathematical scaling
    # explicit here; source questions asking for a percentage have a separate
    # reviewed rule for an omitted percent sign.
    text = text.replace(r"\%", r"\cdot\frac{1}{100}").replace("%", r"\cdot\frac{1}{100}")
    values = parse(
        "$" + text + "$",
        extraction_config=[LatexExtractionConfig(boxed_match_priority=0)],
        fallback_mode="no_fallback",
        extraction_mode="first_match",
        parsing_timeout=PARSING_SECONDS,
        raise_on_error=True,
    )
    if not values:
        raise ValueError("No mathematical expression parsed")
    return {"kind": "math", "value": values}


def _summary(parsed: dict) -> dict:
    if parsed["kind"] in ("tuple", "set_of_tuples"):
        return {"kind": parsed["kind"], "items": [_summary(part) for part in parsed["value"]]}
    if parsed["kind"] != "math":
        return {"kind": parsed["kind"], "value": str(parsed["value"])[:160]}
    # Avoid formatting enormous symbolic objects outside the parser's timeout.
    return {"kind": "math", "types": [type(value).__name__ for value in parsed["value"]]}


def _equivalent(gold: dict, prediction: dict) -> bool:
    from math_verify import verify
    from sympy import Equality

    if gold["kind"] != prediction["kind"]:
        return False
    if gold["kind"] == "set_of_tuples":
        remaining = list(prediction["value"])
        for item in gold["value"]:
            matching = next((i for i, value in enumerate(remaining) if _equivalent(item, value)), None)
            if matching is None:
                return False
            remaining.pop(matching)
        return not remaining
    if gold["kind"] == "tuple":
        return len(gold["value"]) == len(prediction["value"]) and all(
            _equivalent(a, b) for a, b in zip(gold["value"], prediction["value"], strict=True)
        )
    if gold["kind"] != "math":
        return gold["value"] == prediction["value"]
    for value in prediction["value"]:
        relations = value.atoms(Equality)
        if len(relations) > 32:
            raise ValueError("Too many answer equalities")
        for relation in relations:
            left, right = relation.lhs, relation.rhs
            if left.is_Symbol and left not in right.free_symbols:
                continue  # A final assignment such as x=7 may name the value.
            if right.is_Symbol and right not in left.free_symbols:
                continue
            # Math-Verify otherwise unwraps any equation's RHS, even 8=7.
            # Only a true equality may supply a final value this way.
            if not verify(left, right, strict=True, timeout_seconds=PARSING_SECONDS, raise_on_error=True):
                return False
    return bool(
        verify(
            gold["value"],
            prediction["value"],
            strict=True,
            timeout_seconds=PARSING_SECONDS,
            raise_on_error=True,
        )
    )


def _review_reasons(answer: str) -> list[str]:
    text, reasons = _clean(answer), []
    if re.search(r"\\(?:l|c)dots|\.\.\.", text):
        reasons.append("ellipsis_requires_explicit_answer_set")
    if ":" in text:
        reasons.append("ratio_or_score_semantics")
    commas = text.replace("{,}", ",")
    top_level = _parts(commas)
    if len(top_level) > 1 and not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", commas):
        reasons.append("bare_list_or_multiple_answers")
    if len(top_level) == 1 and text.startswith("(") and text.endswith(")") and len(_parts(text[1:-1])) == 2:
        reasons.append("ordered_pair_or_interval")
    return reasons


def grader_metadata() -> dict:
    return {
        "protocol": GRADER_VERSION,
        "math_verify": importlib.metadata.version("math-verify"),
        "strict": True,
        "parsing_timeout_seconds": PARSING_SECONDS,
        "comparison_timeout_seconds": PARSING_SECONDS,
        "extraction": "last explicit final answer or answer-only response; no intermediate-expression search",
        "percentage_semantics": "percent is multiplication by 1/100; omitted signs require a reviewed source rule",
        "equations": "named assignments or verified equalities; false equations cannot contribute only their RHS",
    }


def preflight_gold(goldmap: dict[str, str]) -> dict:
    """Audit every answer; keep unresolved IDs visible instead of dropping rows."""
    from math_verify.errors import TimeoutException

    review, counts = {}, Counter()
    for identity, answer in goldmap.items():
        reasons, parsed = [], None
        try:
            reasons = _review_reasons(answer)
            value = _parse(answer)
            parsed = _summary(value)
            if not _equivalent(value, value):
                reasons.append("gold_does_not_verify_against_itself")
        except (Exception, TimeoutException) as error:  # noqa: BLE001 — preserve parser failures in the audit.
            reasons.append("gold_parse_error:" + type(error).__name__)
        if reasons:
            review[identity] = {"answer": answer, "reasons": reasons, "parsed": parsed}
        counts["needs_review" if reasons else "ready"] += 1
    return {
        "grader": grader_metadata(),
        "gold_sha256": digest(goldmap),
        "n": len(goldmap),
        "ready": not review,
        "counts": dict(counts),
        "needs_review": review,
    }


def grade_answer(gold: str, response: str) -> dict:
    """Grade a final answer; gold/parser errors remain distinct from wrong answers."""
    from math_verify.errors import TimeoutException

    final = extract_final_answer(response)
    result = {"correct": False, "status": "no_final_answer", "final_answer": final, "gold": None, "prediction": None}
    try:
        expected = _parse(gold)
        result["gold"] = _summary(expected)
        result["gold_review_reasons"] = _review_reasons(gold)
    except (Exception, TimeoutException) as error:  # noqa: BLE001 — gold failures must not disappear.
        return result | {"status": "gold_unparseable", "error_type": type(error).__name__}
    if not final:
        return result
    try:
        if expected["kind"] == "text":
            literal = _text_answer(final) or " ".join(final.casefold().split())
            predicted = {"kind": "text", "value": literal}
        else:
            predicted = _parse(final)
        result["prediction"] = _summary(predicted)
    except (Exception, TimeoutException) as error:  # noqa: BLE001 — distinguish invalid output from a wrong answer.
        return result | {"status": "prediction_unparseable", "error_type": type(error).__name__}
    try:
        return result | {"status": "graded", "correct": _equivalent(expected, predicted)}
    except (Exception, TimeoutException) as error:  # noqa: BLE001 — expose symbolic-comparison failures.
        return result | {"status": "grading_error", "error_type": type(error).__name__}

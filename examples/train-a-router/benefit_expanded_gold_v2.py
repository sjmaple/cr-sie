"""Versioned outcome grading after an independent training-reference audit.

The previous graders and reports stay unchanged. Exact rational arithmetic is
used whenever both final expressions fit the bounded grammar below; symbolic
or named answers fall back explicitly to the previous reviewed Math-Verify
protocol. The teacher-reference API uses question wording, never a benchmark
answer, row identity, or an identity-specific correction.
"""

from __future__ import annotations

import re
from fractions import Fraction
from math import isqrt
from pathlib import Path

import benefit_data
import benefit_expanded_gold as previous
import solver_gold
from data import digest, file_hash
from solver_answers import extract_final_answer, grade_answer

PROTOCOL = "source-reviewed-outcomes-exact-rational-v2"
MAX_CHARS = 4096
MAX_TOKENS = 2048
MAX_DEPTH = 64
MAX_BITS = 8192
MAX_EXPONENT = 4096
EXCLUSIONS = previous.EXCLUSIONS
RULES = {
    "e2h-amc-eval-1181": (r"\sqrt{\frac{127}{512}}", r"\frac{127}{512}", "source_correction"),
    "e2h-amc-eval-2735": (r"\frac 34", r"\frac34", "ratio"),
    "e2h-amc-eval-911": ("96", "96", "game_count"),
    "e2h-amc-eval-776": (
        r"\left(\frac{1+2^{1005}}{2^{1007}}\right)^{2}",
        r"\frac{1}{16}(1+\frac{1}{2^{1005}})^2",
        "exact_rational",
    ),
}
REASONS = {
    "e2h-amc-eval-1181": "For k clockwise steps, 2*k-10 must be divisible by5. The allowed k are0,5,10, giving (1+252+1)/1024=127/512. An independent five-state integer walk recurrence agrees; the source's final square root is erroneous.",
    "e2h-amc-eval-2735": "Let home/current distance be x and stadium/current distance be y. Equal times imply x+(x+y)/7=y, hence x/y=3/4. A colon or fraction expresses the same ordered ratio.",
    "e2h-amc-eval-911": "Two divisions each have C(6,2) pairs playing twice, plus6*6 cross-division games: 2*2*15+36=96. A terminal games unit is cosmetic.",
    "e2h-amc-eval-776": "The row and column random walks are independent. Integer path counting on the eight-cycle for2012 steps gives return probability (1+2^1005)/2^1007 for each coordinate. Squaring equals (1/16)*(1+2^-1005)^2 exactly; it is greater than1/16. No numerical tolerance is allowed for this exact rational answer.",
}
OUTCOME_PROTOCOL = {
    "name": PROTOCOL,
    "exact_grammar": "integers, exact finite decimals, + - * /, implicit multiplication, integer powers, parentheses/braces, LaTeX fractions, percentages, rational square roots, and single-letter assignments",
    "unsupported_ambiguities": "Digit-space-digit sequences and unmarked digit-to-LaTeX-fraction adjacency are not concatenated or treated as products; they retain the previous mixed-number parser via explicit fallback.",
    "limits": {
        "characters": MAX_CHARS,
        "tokens": MAX_TOKENS,
        "nesting": MAX_DEPTH,
        "numerator_or_denominator_bits": MAX_BITS,
        "absolute_integer_exponent": MAX_EXPONENT,
    },
    "fallback": "If either expression is outside the bounded rational grammar, retain the previous reviewed Math-Verify result and mark the exact guard unsupported. Explicit source approximation rubrics and non-scalar semantics remain unchanged. The declared exact_rational source rule fails closed on an unsupported answer.",
    "teacher_context": "Only an explicit percentage request, ratio request, or count-of-games request permits the corresponding format normalization. No row IDs, known gold values, or source corrections are used.",
    "amendments": RULES,
}
_TOKEN = re.compile(r"\\(?:frac|sqrt)|\d+(?:\.\d*)?|\.\d+|[()+\-*/^{}%]")


class _RationalParser:
    """Small full-input parser; Fraction operations only, with no eval or floats."""

    def __init__(self, text):
        if not isinstance(text, str) or not text or len(text) > MAX_CHARS:
            raise ValueError("Unsupported expression length")
        text = text.strip()
        for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
            if text.startswith(left) and text.endswith(right):
                text = text[len(left) : -len(right)]
                break
        text = re.sub(r"^[A-Za-z]\s*=\s*", "", text)
        for old, new in (
            (r"\left", ""),
            (r"\right", ""),
            (r"\dfrac", r"\frac"),
            (r"\tfrac", r"\frac"),
            (r"\cdot", "*"),
            (r"\times", "*"),
            (r"\div", "/"),
            (r"\%", "%"),
            ("×", "*"),
            ("÷", "/"),
            ("−", "-"),
            ("[", "("),
            ("]", ")"),
        ):
            text = text.replace(old, new)
        text = re.sub(r"\\(?:qquad|quad|,|;|!| )", " ", text)
        # Preserve token boundaries. Joining digits can turn a mixed number or
        # grouped integer into a different value (for example1 1/5 ->11/5).
        # Likewise1\frac15 conventionally denotes a mixed number, not a product.
        # These ambiguous forms retain the previous parser's declared behavior.
        if re.search(r"\d\s+\d", text):
            raise ValueError("Ambiguous separated numeric literals")
        text = re.sub(r"\s+", "", text)
        if re.search(r"\d\\frac", text):
            raise ValueError("Ambiguous mixed-number adjacency")
        self.tokens = _TOKEN.findall(text)
        if "".join(self.tokens) != text or not self.tokens or len(self.tokens) > MAX_TOKENS:
            raise ValueError("Unsupported numeric grammar")
        self.index = 0
        self.depth = 0

    def peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else ""

    def take(self):
        token = self.peek()
        self.index += 1
        return token

    @staticmethod
    def bounded(value):
        if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > MAX_BITS:
            raise ValueError("Exact arithmetic size limit")
        return value

    def expression(self):
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise ValueError("Expression nesting limit")
        value = self.product()
        while self.peek() in ("+", "-"):
            operator = self.take()
            other = self.product()
            value = self.bounded(value + other if operator == "+" else value - other)
        self.depth -= 1
        return value

    def product(self):
        value = self.unary()
        while self.peek() and self.peek() not in ("+", "-", ")", "}"):
            operator = self.peek()
            if operator in ("*", "/"):
                self.take()
            elif operator not in ("(", "{", r"\frac", r"\sqrt"):
                raise ValueError("Ambiguous implicit multiplication")
            other = self.unary()
            value = self.bounded(value / other if operator == "/" else value * other)
        return value

    def unary(self):
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise ValueError("Expression nesting limit")
        if self.peek() in ("+", "-"):
            sign = self.take()
            value = self.unary()
            self.depth -= 1
            return -value if sign == "-" else value
        value = self.atom()
        if self.peek() == "^":
            self.take()
            exponent = self.unary()
            if exponent.denominator != 1 or abs(exponent.numerator) > MAX_EXPONENT:
                raise ValueError("Unsupported exponent")
            exponent = exponent.numerator
            if not value and exponent <= 0:
                raise ValueError("Undefined zero power")
            size = max(abs(value.numerator).bit_length(), value.denominator.bit_length())
            if size * abs(exponent) > MAX_BITS:
                raise ValueError("Exponent size limit")
            value = self.bounded(value**exponent)
        while self.peek() == "%":
            self.take()
            value = self.bounded(value / 100)
        self.depth -= 1
        return value

    def argument(self):
        # An unbraced LaTeX fraction argument is one digit, even in \frac34.
        token = self.peek()
        if re.fullmatch(r"\d+", token):
            self.tokens[self.index] = token[1:]
            if len(token) == 1:
                self.index += 1
            return Fraction(token[0])
        return self.atom()

    def atom(self):
        token = self.take()
        if token in ("(", "{"):
            value = self.expression()
            if self.take() != (")" if token == "(" else "}"):
                raise ValueError("Unbalanced expression")
            return value
        if token == r"\frac":
            numerator, denominator = self.argument(), self.argument()
            return self.bounded(numerator / denominator)
        if token == r"\sqrt":
            value = self.argument()
            if value < 0:
                raise ValueError("Non-rational square root")
            numerator, denominator = isqrt(value.numerator), isqrt(value.denominator)
            if numerator**2 != value.numerator or denominator**2 != value.denominator:
                raise ValueError("Non-rational square root")
            return Fraction(numerator, denominator)
        if re.fullmatch(r"\d+(?:\.\d*)?|\.\d+", token):
            return self.bounded(Fraction(token))
        raise ValueError("Expected a numeric atom")

    def parse(self):
        value = self.expression()
        if self.index != len(self.tokens):
            raise ValueError("Unconsumed expression")
        return value


def exact_rational(final):
    """Return an exact Fraction, or None when the bounded grammar is unsupported."""
    try:
        return _RationalParser(final).parse()
    except (ValueError, ZeroDivisionError, OverflowError, RecursionError):
        return None


def _numeric_summary(value):
    # Hash the complete value while keeping large powers readable in reports.
    numerator, denominator = str(value.numerator), str(value.denominator)
    return {
        "kind": "exact_rational",
        "numerator": numerator if len(numerator) <= 120 else numerator[:60] + "…",
        "denominator": denominator if len(denominator) <= 120 else denominator[:60] + "…",
        "value_sha256": digest([numerator, denominator]),
    }


def exact_comparison(expected_final, observed_final):
    """Compare full final expressions exactly; None explicitly requests fallback."""
    expected, observed = exact_rational(expected_final), exact_rational(observed_final)
    if expected is None or observed is None:
        return None
    return {
        "correct": expected == observed,
        "status": "graded",
        "final_answer": observed_final,
        "gold": _numeric_summary(expected),
        "prediction": _numeric_summary(observed),
        "exact_numeric_guard": "applied",
    }


def outcome_grader_metadata():
    return {
        "outcome_grader_source_sha256": file_hash(Path(__file__)),
        "outcome_grader_protocol_sha256": digest(OUTCOME_PROTOCOL),
    }


def _ratio(value):
    if value.count(":") == 1:
        left, right = value.split(":")
        return r"\frac{" + left + "}{" + right + "}"
    return value


def _game_count(value):
    plain = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", value)
    plain = re.sub(r"\\[,;! ]|~", " ", plain).strip()
    match = re.fullmatch(r"(.+?)\s*(?:conference\s+)?games?", plain, flags=re.IGNORECASE)
    return match[1] if match and exact_rational(match[1]) is not None else value


def _guard(expected, final, fallback):
    compared = exact_comparison(expected, final)
    if compared is None:
        return fallback | {"exact_numeric_guard": "unsupported; previous reviewed grader retained"}
    return fallback | compared


def grade_expanded(identity, original, response):
    """Apply new source amendments, then exactness to eligible scalar answers."""
    final = extract_final_answer(response)
    rule = RULES.get(identity)
    if rule:
        expected_original, canonical, kind = rule
        if original != expected_original:
            raise ValueError("Amended source adjudication differs from original answer")
        if not final:
            return grade_answer(canonical, response) | {"exact_numeric_guard": "no final answer"}
        normalized = _ratio(final) if kind == "ratio" else _game_count(final) if kind == "game_count" else final
        result = _guard(canonical, normalized, grade_answer(canonical, r"\boxed{" + normalized + "}"))
        if kind == "exact_rational" and exact_rational(normalized) is None:
            result = result | {
                "correct": False,
                "status": "prediction_unparseable",
                "exact_numeric_guard": "required; unsupported expression",
            }
        return result | {"final_answer": final, "adjudication": {"protocol": PROTOCOL, "id": identity, "kind": kind}}
    result = previous.grade_expanded(identity, original, response)
    canonical, kind = original, "scalar"
    for rules in (previous.RULES, benefit_data.RULES, solver_gold.RULES):
        if identity in rules:
            _, canonical, kind = rules[identity]
            break
    if kind not in {"scalar", "source_correction", "notation", "ratio", "percentage"}:
        return result | {"exact_numeric_guard": "not applicable to reviewed non-exact or non-scalar semantics"}
    if kind == "percentage":
        canonical = canonical if "%" in final else canonical.removesuffix(r"\%")
    normalized = _ratio(final) if kind == "ratio" else final
    return _guard(canonical, normalized, result) | {"final_answer": final}


def grade_teacher_reference(question, teacher_final, solver_response):
    """Question-only formatting; no IDs, benchmark references, or corrections.

    teacher_final is the teacher's declared answer field. solver_response is the
    solver's full response and must contain an extractable final answer.
    """
    final = extract_final_answer(solver_response)
    expected = extract_final_answer(r"\boxed{" + teacher_final + "}")
    observed = final
    contexts = []
    percentage = re.search(
        r"\b(?:what|which)\s+(?:is\s+(?:the\s+)?)?percent(?:age)?\b|\b(?:find|determine|compute|give)\s+(?:the\s+)?percent(?:age)?\b",
        question,
        re.IGNORECASE,
    )
    ratio = re.search(r"\b(?:what\s+is|find|determine|compute|give)\s+(?:the\s+)?ratio\b", question, re.IGNORECASE)
    games = re.search(r"\bhow\s+many(?:\s+\w+){0,5}\s+games\b", question, re.IGNORECASE)
    if percentage:
        expected = expected if "%" in expected else "(" + expected + r")\%"
        observed = observed if "%" in observed else "(" + observed + r")\%"
        contexts.append("explicit_percentage_request")
    if ratio:
        expected, observed = _ratio(expected), _ratio(observed)
        contexts.append("explicit_ratio_request")
    if games:
        expected, observed = _game_count(expected), _game_count(observed)
        contexts.append("explicit_game_count_request")
    fallback = grade_answer(expected, r"\boxed{" + observed + "}" if final else solver_response)
    result = _guard(expected, observed, fallback)
    return result | {
        "final_answer": final,
        "reference_mode": "teacher_final_and_question_context_only",
        "context_normalizations": contexts,
    }


def review_expanded(gold, sources):
    """Keep old review receipts intact; emit an explicitly amended new review."""
    corrected, preflight, old_review = previous.review_expanded(gold, sources)
    records = list(old_review["correction_records"])
    for identity, (original, canonical, kind) in RULES.items():
        if identity not in gold:
            continue
        source = sources[identity]
        if gold[identity] != original or source["answer"] != original:
            raise ValueError("Amended source review differs from the pinned answer")
        if not grade_expanded(identity, original, r"\boxed{" + canonical + "}")["correct"]:
            raise ValueError("Amended canonical answer does not grade")
        corrected[identity] = canonical
        records.append(
            {
                "id": identity,
                "original": original,
                "canonical": canonical,
                "kind": kind,
                "source_answer_correction": kind == "source_correction",
                "reason": REASONS[identity],
                "problem_sha256": digest(source["problem"]),
                "solution_sha256": digest(source["solution"]),
            }
        )
    unresolved = {identity: value for identity, value in old_review["needs_review"].items() if identity not in RULES}
    review = (
        old_review
        | {
            "protocol": PROTOCOL,
            "previous_review_protocol": previous.PROTOCOL,
            "previous_review_source_sha256": file_hash(Path(previous.__file__)),
            "canonical_gold_sha256": digest(corrected),
            "approved_ids": sorted(set(gold) - set(unresolved)),
            "adjudicated_ids": sorted(set(old_review["adjudicated_ids"]) | (set(gold) & set(RULES))),
            "unresolved_ids": sorted(unresolved),
            "needs_review": unresolved,
            "ready": not unresolved,
            "correction_records": records,
            "expanded_gold_rules_sha256": digest(
                {
                    "previous_rules": previous.RULES,
                    "rules": RULES,
                    "exclusions": EXCLUSIONS,
                    "outcome_protocol": OUTCOME_PROTOCOL,
                }
            ),
            "expanded_gold_review_source_sha256": file_hash(Path(__file__)),
            "review_scope": "Four amendments independently checked against source and exact arithmetic after a teacher audit on original training rows; earlier reports are preserved. No expanded solver outcomes were consulted. The generic rational exactness guard is not an independent solution audit of every question.",
        }
        | outcome_grader_metadata()
    )
    return corrected, preflight, review

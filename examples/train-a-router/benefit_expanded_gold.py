"""Source-only answer review for the expanded paired-training population.

The original 511-example grading protocol is delegated unchanged. New rules
were decided from benchmark questions and solutions before collecting any new
solver answers. Incomplete questions are excluded, never silently repaired.
"""

from __future__ import annotations

import re
from math import factorial
from pathlib import Path

import benefit_data
from data import digest, file_hash
from solver_answers import _parts, extract_final_answer, grade_answer

PROTOCOL = "source-only-expanded-training-gold-v1"


def _plain(value):
    value = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\[,;! ]|~", " ", value)
    return " ".join(value.casefold().split()).strip()


def _clock_minutes(value):
    value = re.sub(r"\b([ap])\s*\.\s*m\.?", r"\1m", _plain(value))
    match = re.fullmatch(r"(\d{1,2})(?::(\d{1,2}))?\s*(am|pm)?", value)
    if match is None:
        return None
    hour, minute, meridiem = int(match[1]), int(match[2] or 0), match[3]
    if hour > 23 or minute > 59 or (meridiem and not 1 <= hour <= 12):
        return None
    if meridiem:
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
    return 60 * hour + minute


def _expand_plus_minus(value):
    """Expand one explicit plus/minus per finite-list member, with a small cap."""
    value = value.replace(r"\left", "").replace(r"\right", "").replace("±", r"\pm")
    if value.startswith(r"\{") and value.endswith(r"\}"):
        value = value[2:-2]
    elif value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    try:
        parts = _parts(value)
    except ValueError:
        return None
    if len(parts) > 16 or any(part.count(r"\pm") > 1 for part in parts):
        return None
    expanded = [
        item
        for part in parts
        for item in ([part.replace(r"\pm", "+"), part.replace(r"\pm", "-")] if r"\pm" in part else [part])
    ]
    return r"\{" + ",".join(expanded) + r"\}"


def _special(correct, final, kind, expected, observed):
    return {
        "correct": bool(correct),
        "status": "graded" if final else "no_final_answer",
        "final_answer": final,
        "gold": {"kind": kind, "value": str(expected)},
        "prediction": {"kind": kind, "value": str(observed) if observed is not None else None},
    }


def grade_expanded(identity, original, response):
    """Grade one eligible row, using only a source-bound explicit final answer."""
    if identity in EXCLUSIONS or identity in benefit_data.EXCLUSIONS:
        raise ValueError("Source-invalid training question must be excluded before collection")
    if identity not in RULES:
        return benefit_data.grade_training(identity, original, response)
    expected_original, canonical, kind = RULES[identity]
    if original != expected_original:
        raise ValueError("Expanded source adjudication differs from the original answer")
    final = extract_final_answer(response)
    if kind == "clock_time":
        observed = _clock_minutes(final) if final else None
        result = _special(observed == 660, final, kind, "11:00 AM", observed)
    elif kind == "division_operator":
        observed = _plain(final)
        accepted = {r"\div", "÷", "/", "division", "divide", "division operator"}
        result = _special(observed in accepted, final, kind, "division", observed)
    elif kind == "ordered_labels":
        observed = _plain(final)
        for name, letter in (("medium", "m"), ("large", "l"), ("small", "s")):
            observed = re.sub(r"\b" + name + r"\b", letter, observed)
        observed = observed.strip("()[]{}").replace(r"\to", "> ").replace("→", ">")
        correct = observed == "mls" or re.fullmatch(r"m[\s,>]+l[\s,>]+s", observed) is not None
        result = _special(correct, final, kind, "medium, large, small", observed)
    elif kind == "weekday":
        observed = _plain(final)
        result = _special(
            observed in {"friday", "day 5", "fifth day", "the fifth day"}, final, kind, "Friday", observed
        )
    else:
        normalized = final
        if kind == "percentage":
            canonical = canonical if "%" in final else canonical.removesuffix(r"\%")
        elif kind == "ratio" and final.count(":") == 1:
            left, right = final.split(":")
            if left.strip() and right.strip():
                normalized = r"\frac{" + left + "}{" + right + "}"
        elif kind in {"finite_set", "set_of_ordered_tuples"} and (r"\pm" in final or "±" in final):
            normalized = _expand_plus_minus(final) or final
        elif kind == "official_alternatives":
            # The spaced source form15 !^2 is not parsed reliably by Math-Verify.
            # Replace only this declared factorial, without evaluating arbitrary code.
            normalized = re.sub(r"(?<!\d)15\s*!", "(" + str(factorial(15)) + ")", final)
        normalized_response = r"\boxed{" + normalized + "}" if final else response
        result = grade_answer(canonical, normalized_response)
        if kind == "official_alternatives" and not result["correct"]:
            for alternative in ("2", r"\{2," + canonical + r"\}"):
                candidate = grade_answer(alternative, normalized_response)
                if candidate["correct"]:
                    result = candidate
                    break
        result["final_answer"] = final
    return result | {"adjudication": {"protocol": PROTOCOL, "id": identity, "kind": kind}}


def review_expanded(gold, sources):
    """Review eligible source gold; the caller pins and joins the source files."""
    if set(gold) & (set(EXCLUSIONS) | set(benefit_data.EXCLUSIONS)):
        raise ValueError("Remove predeclared source-invalid rows before reviewing eligible gold")
    corrected, preflight, review = benefit_data.review_training(gold, sources)
    records = list(review["correction_records"])
    for identity, (original, canonical, kind) in RULES.items():
        if identity not in gold:
            continue
        if (
            gold[identity] != original
            or not grade_expanded(identity, original, r"\boxed{" + canonical + "}")["correct"]
        ):
            raise ValueError("Expanded source review differs or cannot grade its canonical answer")
        source = sources[identity]
        if source["answer"] != original:
            raise ValueError("Source row and reviewed gold disagree")
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
    unresolved = {identity: value for identity, value in review["needs_review"].items() if identity not in RULES}
    return (
        corrected,
        preflight,
        review
        | {
            "protocol": PROTOCOL,
            "base_review_protocol": benefit_data.PROTOCOL,
            "canonical_gold_sha256": digest(corrected),
            "approved_ids": sorted(set(gold) - set(unresolved)),
            "adjudicated_ids": sorted(set(gold) & (set(RULES) | set(benefit_data.RULES))),
            "unresolved_ids": sorted(unresolved),
            "needs_review": unresolved,
            "ready": not unresolved,
            "correction_records": records,
            "expanded_gold_rules_sha256": digest({"rules": RULES, "exclusions": EXCLUSIONS}),
            "expanded_gold_review_source_sha256": file_hash(Path(__file__)),
            "review_scope": "Original511 rules unchanged; expanded parser/context-flagged answers reviewed from source only before new solver outcomes. This is not an independent solution audit of every question. Exclusions and duplicate checks are bound separately by the preparation eligibility receipt.",
        },
    )


RULES = {
    "e2h-amc-train-774": ("400,0", "400000", "source_correction"),
    "e2h-amc-train-720": ("\\overline{\\frac{53}{188}}", "\\frac{53}{188}", "notation"),
    "e2h-amc-eval-799": (
        "(1,3),\\left(3, \\frac{1}{3}\\right) \\quad",
        "\\{(1,3),(3,\\frac13)\\}",
        "set_of_ordered_tuples",
    ),
    "e2h-amc-eval-2835": (
        "\\frac{-1+\\sqrt{5}}{2}, \\frac{1+\\sqrt{5}}{2}",
        "\\{\\frac{-1+\\sqrt5}{2},\\frac{1+\\sqrt5}{2}\\}",
        "finite_set",
    ),
    "e2h-amc-eval-2435": ("\\left(1009,2^{1009}-2\\right)", "(1009,2^{1009}-2)", "ordered_tuple"),
    "e2h-amc-train-796": ("-1, i,-i", "\\{-1,i,-i\\}", "finite_set"),
    "e2h-amc-eval-1193": (
        "3,-\\frac{3}{2}+\\frac{3 i \\sqrt{3}}{2},-\\frac{3}{2}-\\frac{3 i \\sqrt{3}}{2}",
        "\\{3,-\\frac32+\\frac{3i\\sqrt3}{2},-\\frac32-\\frac{3i\\sqrt3}{2}\\}",
        "finite_set",
    ),
    "e2h-amc-eval-2110": ("2:1", "2", "ratio"),
    "e2h-amc-eval-1456": ("1,8,16", "1008016", "source_correction"),
    "e2h-amc-eval-2533": ("32:17", "\\frac{32}{17}", "ratio"),
    "e2h-amc-eval-616": ("(-1,-5)", "(-1,-5)", "ordered_tuple"),
    "e2h-amc-eval-1617": ("\\left(\\frac{1}{2},-\\frac{1}{12}\\right)", "(\\frac12,-\\frac1{12})", "ordered_tuple"),
    "e2h-amc-eval-2049": ("(12,2)", "(12,2)", "ordered_tuple"),
    "e2h-amc-eval-610": ("(1000,1988)", "(1000,1988)", "ordered_tuple"),
    "e2h-amc-eval-439": ("-20,28,38", "\\{-20,28,38\\}", "finite_set"),
    "e2h-amc-eval-1548": ("-2,1 \\pm \\sqrt{5}", "\\{-2,1+\\sqrt5,1-\\sqrt5\\}", "finite_set"),
    "e2h-amc-eval-853": ("(0,5,5),(4,5,1),(6,4,9)", "\\{(0,5,5),(4,5,1),(6,4,9)\\}", "set_of_ordered_tuples"),
    "e2h-amc-eval-2809": ("(3,2)", "(3,2)", "ordered_tuple"),
    "e2h-amc-eval-910": ("11: 0", "\\text{11:00 AM}", "clock_time"),
    "e2h-amc-eval-2959": ("\\left(\\frac{12}{5}, \\frac{8}{5}\\right)", "(\\frac{12}{5},\\frac85)", "ordered_tuple"),
    "e2h-amc-eval-1522": ("3:5", "\\frac35", "ratio"),
    "e2h-amc-eval-946": ("\\div", "\\div", "division_operator"),
    "e2h-amc-train-504": (
        "1, \\frac{3+\\sqrt{3} i}{2}, \\frac{1-\\sqrt{3} i}{2}, \\frac{3-\\sqrt{3} i}{2}, \\frac{1+\\sqrt{3} i}{2}",
        "\\{1,\\frac{3+i\\sqrt3}{2},\\frac{1-i\\sqrt3}{2},\\frac{3-i\\sqrt3}{2},\\frac{1+i\\sqrt3}{2}\\}",
        "finite_set",
    ),
    "e2h-amc-eval-2032": (
        (
            "\\left(\\frac{\\sqrt{30}}{3}, \\frac{\\sqrt{30}}{4}, \\frac{2 "
            "\\sqrt{30}}{5}\\right),\\left(\\frac{\\sqrt{30}}{3}, \\frac{\\sqrt{30}}{2}, "
            "\\frac{\\sqrt{30}}{5}\\right)"
        ),
        "\\{(\\frac{\\sqrt{30}}3,\\frac{\\sqrt{30}}4,\\frac{2\\sqrt{30}}5),(\\frac{\\sqrt{30}}3,\\frac{\\sqrt{30}}2,\\frac{\\sqrt{30}}5)\\}",
        "set_of_ordered_tuples",
    ),
    "e2h-amc-eval-479": (
        "1, \\frac{2^{2011}+1}{3 \\cdot 2^{2011}}",
        "\\{1,\\frac{2^{2011}+1}{3\\cdot2^{2011}}\\}",
        "finite_set",
    ),
    "e2h-amc-train-643": ("1:1", "1", "ratio"),
    "e2h-amc-eval-2670": ("2 \\cdot 15 !^{2}", "3420024505448398848000000", "official_alternatives"),
    "e2h-amc-eval-1533": ("6:\\pi", "\\frac{6}{\\pi}", "ratio"),
    "e2h-amc-eval-487": ("2,6", "\\{2,6\\}", "finite_set"),
    "e2h-amc-train-4": ("(3,-24)", "(3,-24)", "ordered_tuple"),
    "e2h-amc-eval-1380": ("(\\sqrt{3}, 2)", "(\\sqrt3,2)", "ordered_tuple"),
    "e2h-amc-eval-154": ("3:1", "3", "ratio"),
    "e2h-amc-eval-870": ("400,0", "400000", "source_correction"),
    "e2h-amc-eval-2781": ("0,1,3,4,6", "\\{0,1,3,4,6\\}", "finite_set"),
    "e2h-amc-train-539": ("-3,2,5", "\\{-3,2,5\\}", "finite_set"),
    "e2h-amc-train-965": (
        "(-1006,1006 \\sqrt{3}),(-1006,-1006 \\sqrt{3})",
        "\\{(-1006,1006\\sqrt3),(-1006,-1006\\sqrt3)\\}",
        "set_of_ordered_tuples",
    ),
    "e2h-amc-eval-1395": ("2:3", "\\frac23", "ratio"),
    "e2h-amc-train-850": ("\\left(\\frac{2}{3}, \\frac{2}{9}\\right)", "(\\frac23,\\frac29)", "ordered_tuple"),
    "e2h-amc-eval-2203": ("3:1", "3", "ratio"),
    "e2h-amc-eval-2564": (
        "\\frac{1}{2}, \\frac{-1 \\pm \\sqrt{13}}{4}",
        "\\{\\frac12,\\frac{-1+\\sqrt{13}}4,\\frac{-1-\\sqrt{13}}4\\}",
        "finite_set",
    ),
    "e2h-amc-eval-2746": ("37:1", "37", "ratio"),
    "e2h-amc-eval-1524": ("(35,90)", "(35,90)", "ordered_tuple"),
    "e2h-amc-eval-2704": ("3,5", "\\{3,5\\}", "finite_set"),
    "e2h-amc-eval-706": ("503,1509", "\\{503,1509\\}", "finite_set"),
    "e2h-amc-eval-2336": ("\\left( - \\frac {5}{3}\\sqrt {3},5\\right)", "(-\\frac{5\\sqrt3}3,5)", "ordered_tuple"),
    "e2h-amc-eval-548": ("(-4, 9)", "(-4,9)", "ordered_tuple"),
    "e2h-amc-eval-1138": ("\\text{MLS}", "\\text{MLS}", "ordered_labels"),
    "e2h-amc-eval-2777": ("\\text{Friday}", "\\text{Friday}", "weekday"),
    "e2h-amc-train-859": ("{{{0.189}}}", "0.189", "notation"),
    "e2h-amc-eval-1412": ("0.96039", "0.96040", "source_correction"),
    "e2h-amc-eval-558": ("12.5", "12.5\\%", "percentage"),
    "e2h-amc-eval-761": ("17", "17\\%", "percentage"),
    "e2h-amc-eval-2508": ("36", "36\\%", "percentage"),
    "e2h-amc-train-581": ("99", "99\\%", "percentage"),
    "e2h-amc-train-951": ("40", "40\\%", "percentage"),
    "e2h-amc-eval-79": ("60", "60\\%", "percentage"),
    "e2h-amc-eval-2321": ("4", "4\\%", "percentage"),
    "e2h-amc-eval-968": ("25", "25\\%", "percentage"),
    "e2h-amc-eval-2853": ("84", "84\\%", "percentage"),
    "e2h-amc-eval-1312": ("\\frac{10}{3}", "\\frac{10}{3}\\%", "percentage"),
    "e2h-amc-train-311": ("60", "60\\%", "percentage"),
    "e2h-amc-eval-902": ("75\\%", "75\\%", "percentage"),
    "e2h-amc-eval-191": ("93", "93\\%", "percentage"),
    "e2h-amc-train-217": ("60 \\%", "60\\%", "percentage"),
    "e2h-amc-eval-1809": ("30\\%", "30\\%", "percentage"),
    "e2h-amc-eval-1782": ("27", "27\\%", "percentage"),
    "e2h-amc-train-969": ("18", "18\\%", "percentage"),
    "e2h-amc-eval-1590": ("20\\%", "20\\%", "percentage"),
    "e2h-amc-train-767": ("25", "25\\%", "percentage"),
    "e2h-amc-eval-2474": ("23\\%", "23\\%", "percentage"),
    "e2h-amc-eval-1150": ("29", "29\\%", "percentage"),
    "e2h-amc-eval-619": ("51\\%", "51\\%", "percentage"),
}

REASONS = {
    "e2h-amc-train-774": "The maximum salary is 700000−20×15000=400000; source prose agrees and its boxed answer lost "
    "zeros.",
    "e2h-amc-train-720": "Conditional binomial probability is(90+15+1)/(270+90+15+1)=53/188; the overline is stray "
    "answer formatting, not a different probability.",
    "e2h-amc-eval-799": "Substitution gives x²−4 x+3=0, so the two ordered pairs are(1, 3) and(3, 1/3).",
    "e2h-amc-eval-2835": "The two circle configurations give the positive roots of t²+t−1=0 and t²−t−1=0; both ratios "
    "are required.",
    "e2h-amc-eval-2435": "The question explicitly orders(number of values, sum). Possible counts are 0 and 2^i for "
    "1≤i≤1008.",
    "e2h-amc-train-796": "The polynomial factors as(x+1)²(x²+1); distinct solutions are−1, i,−i.",
    "e2h-amc-eval-1193": "The two cube identities give a=ω+2ω′ for cube roots of unity; maximum magnitude 3 "
    "requiresω=ω′.",
    "e2h-amc-eval-2110": "Angle-bisector/segment relations give AF/FD=6/7+8/7=2; colon and fraction notation denote the "
    "same ratio.",
    "e2h-amc-eval-1456": "Only monomials with even total y, z degree survive; sum(2 k+1, k=0..1003)=1004²=1008016. "
    "Source prose agrees; its box lost zeros.",
    "e2h-amc-eval-2533": "The new 7×7 border adds 24 black squares to the original 8; the 17 white squares remain.",
    "e2h-amc-eval-616": "The parenthesized factor is−5 x+O(x²); a finite nonzero limit requires p=−1 and the integer "
    "c=−5.",
    "e2h-amc-eval-1617": "Comparing coefficients in H(n+1)−H(n)=1/(n+1) yields c=1/2, d=−1/12; the order is specified.",
    "e2h-amc-eval-2049": "Independent breadth-first search of all 10000 passcodes gives maximum 12 moves, attained only "
    "by 4826 and 6284.",
    "e2h-amc-eval-610": "Reflection in y=2000 maps(x, y) to(x, 4000−y).",
    "e2h-amc-eval-439": "Adding the three equations gives x+y+z=46; solving gives the three integers−20, 28, 38, with "
    "no requested order.",
    "e2h-amc-eval-1548": "The increasing inverse-function relation reduces to(x+2)(x²−2 x−4)=0; plus/minus denotes two "
    "distinct roots.",
    "e2h-amc-eval-853": "Independent enumeration of all 1000 digit triples and divisibility by 792 gives exactly these "
    "three ordered triples.",
    "e2h-amc-eval-2809": "Clockwise 90° sends(−3, 2) to(2, 3), then reflection in x=y gives(3, 2).",
    "e2h-amc-eval-910": "At 09: 00 the cyclists are 56 miles apart and close at 28 mph; they meet at 11: 00 in the "
    "morning.",
    "e2h-amc-eval-2959": "The transformed tangency point is(12/5, 16/5); inverse transformation on 0≤y≤4 gives(12/5, "
    "8/5).",
    "e2h-amc-eval-1522": "The diagram contains 6 gray and 10 white equal small squares, giving gray: white=3: 5.",
    "e2h-amc-eval-946": "The equation reduces to 6?3=2, so the missing operation is division. A choice letter alone is "
    "unsupported because choices are absent.",
    "e2h-amc-train-504": "With y=x−1 the polynomial is y(y⁴+y²+1)=y(y²+y+1)(y²−y+1), giving these five roots.",
    "e2h-amc-eval-2032": "The product of the three floors must be 2; placing the 2 in the b or c coordinate gives the "
    "two solutions, while placing it in a contradicts its floor.",
    "e2h-amc-eval-479": "The two transformations generate the six-position orbit. Generic/equivalent three-value orbits "
    "give the stated return probability, and the two-value complex orbit returns with probability "
    "1.",
    "e2h-amc-train-643": "A regular hexagon of side 1 consists of six equilateral triangles congruent to its six "
    "extensions.",
    "e2h-amc-eval-2670": "The connected desk grid has two checkerboard gender assignments; distinguishable students "
    "give 2×(15!)². The source explicitly states that 2 was also accepted because "
    "distinguishability was ambiguous.",
    "e2h-amc-eval-1533": "The rectangle area is 1350 and the combined semicircle area is 225π; their ratio is 6/π.",
    "e2h-amc-eval-487": "Independent enumeration of all closed simple 12-step orthogonal walks with 90° turns and side "
    "2 gives exactly areas 2 and 6 for the specified three vertices.",
    "e2h-amc-train-4": "The commuting cubic family can be centered at 3, with p=a(x−3)³+(9−9 a)(x−3)+3 and q=6−p. Thus "
    "p(3)=3 and q(6)=−24. One abbreviated family in the source prose omits its linear term; the "
    "final answer is unchanged.",
    "e2h-amc-eval-1380": "The enclosing circle is supported by a diameter or an acute triangle, giving√3≤D/R≤2. The "
    "extremal configurations can include additional distinct interior points.",
    "e2h-amc-eval-154": "Equating 0.75×(4πr³/3) withπr²h/3 gives h/r=3. Percentage is a premise, not the answer unit.",
    "e2h-amc-eval-870": "The maximum salary is 700000−20×15000=400000; source prose agrees and its boxed answer lost "
    "zeros.",
    "e2h-amc-eval-2781": "For n≠±2 the expression is n+4/(n−2). Checking the divisors of 4 and excluding the original "
    "zero denominator gives 0, 1, 3, 4, 6.",
    "e2h-amc-train-539": "Factorization(2 x−3)(x+2) and gcd dividing 7 gives exactly−3, 2, 5. The source explicitly "
    "corrects an earlier omission of−3.",
    "e2h-amc-train-965": "Equality in the two-step distance bound forces the two reflected extremal centric sequences, "
    "whose 2012 th points are the listed ordered pairs.",
    "e2h-amc-eval-1395": "The folded overlap has area√3/3, leaving area 2√3/3 from the original√3.",
    "e2h-amc-train-850": "The recurrence E_n=1+(E_(n−1)+E_(n−2))/2 gives E_n=2 n/3+2/9−(2/9)(−1/2)^n.",
    "e2h-amc-eval-2203": "Equating 0.75×(4πr³/3) withπr²h/3 gives h/r=3. This is the same mathematical question as "
    "eval-154.",
    "e2h-amc-eval-2564": "The increasing inverse-function identity reduces to(x−1/2)(x²+x/2−3/4)=0; expand plus/minus "
    "into both roots.",
    "e2h-amc-eval-2746": "The extended equilateral triangle has squared side ratio 4²+3²−2×4×3 cos 120°=37, equal to "
    "its area ratio.",
    "e2h-amc-eval-1524": "Independent enumeration of 1≤a≤b≤100, testing exact divisibility and maximizing(b, a), "
    "gives(35, 90).",
    "e2h-amc-eval-2704": "Explicit pairings exist for 3 and 5; odd/even row counts differ for every larger odd n, so no "
    "diagonal pairing is possible.",
    "e2h-amc-eval-706": "The source backward-induction solution identifies the two symmetric optimum first choices 503 "
    "and 2012−503=1509; no order is requested.",
    "e2h-amc-eval-2336": "The original point is(5, 5/√3); counterclockwise 90° maps(x, y) to(−y, x).",
    "e2h-amc-eval-548": "The altitude bisects the base in this isosceles triangle, so C=2 D−B=(−4, 9).",
    "e2h-amc-eval-1138": "Per-ounce costs are M=.1875, L=.195, S=.2; the requested best-to-worst order is medium, "
    "large, small. Preserve order, not a commutative product.",
    "e2h-amc-eval-2777": "Accumulated millet first exceeds 3/4 quart on the fifth addition; counting Monday as day 1 "
    "gives Friday.",
    "e2h-amc-train-859": "Exactly one approval has probability 3×0.7×0.3²=.189; nested braces in the source are "
    "answer-box formatting.",
    "e2h-amc-eval-1412": "Exact inclusion-exclusion over the four forbidden gift edges among persons 1, 2, 3, within "
    "101-person derangements gives 0.960398039803980..., which rounds to 0.96040. The source loses "
    "a sign and reports 0.96039; independent small-n brute force validates the counting method.",
    "e2h-amc-eval-558": "The square has 1/8 of the rectangle area, or 12.5 percent. A source-context percentage rule "
    "permits an omitted sign but preserves explicit percentage scaling.",
    "e2h-amc-eval-761": "The question explicitly requests the nearest integer percentage coefficient x; 100/6 rounds to "
    "17. A source-context percentage rule permits an omitted sign but preserves explicit percentage "
    "scaling.",
    "e2h-amc-eval-2508": "The height becomes 16/25 of its original value, a 36 percent decrease. A source-context "
    "percentage rule permits an omitted sign but preserves explicit percentage scaling.",
    "e2h-amc-train-581": "Area scales by 1.1×.9=.99, so the new area is 99 percent of the old. A source-context "
    "percentage rule permits an omitted sign but preserves explicit percentage scaling.",
    "e2h-amc-train-951": "Geese are 30/75 of non-swans, or 40 percent. A source-context percentage rule permits an "
    "omitted sign but preserves explicit percentage scaling.",
    "e2h-amc-eval-79": "Successive discounts leave.5×.8=.4 of the price, so 60 percent is removed. A source-context "
    "percentage rule permits an omitted sign but preserves explicit percentage scaling.",
    "e2h-amc-eval-2321": "400 euros×1.3=520 dollars, a 4 percent increase over 500 dollars. A source-context percentage "
    "rule permits an omitted sign but preserves explicit percentage scaling.",
    "e2h-amc-eval-968": "The four shaded fractions sum to 1 square out of 4, giving 25 percent. A source-context "
    "percentage rule permits an omitted sign but preserves explicit percentage scaling.",
    "e2h-amc-eval-2853": "Correct answers total 63 out of 75, giving 84 percent. A source-context percentage rule "
    "permits an omitted sign but preserves explicit percentage scaling.",
    "e2h-amc-eval-1312": "The question asks a percentage; 2=.6 x implies x=10/3 percent, not a fraction 10/3 of the "
    "milk. A source-context percentage rule permits an omitted sign but preserves explicit "
    "percentage scaling.",
    "e2h-amc-train-311": "The changed-answer percentage ranges from 20 to 80; their difference is 60 percentage points. "
    "A source-context percentage rule permits an omitted sign but preserves explicit percentage "
    "scaling.",
    "e2h-amc-eval-902": "Non-filler is 90/120 of the burger, or 75 percent. A source-context percentage rule permits an "
    "omitted sign but preserves explicit percentage scaling.",
    "e2h-amc-eval-191": "The shared-half correctness is 96 percent; averaging it with Zoe’s 90 percent gives 93 "
    "percent. A source-context percentage rule permits an omitted sign but preserves explicit "
    "percentage scaling.",
    "e2h-amc-train-217": "The final must supply 490−310=180 points out of 300, or 60 percent. A source-context "
    "percentage rule permits an omitted sign but preserves explicit percentage scaling.",
    "e2h-amc-eval-1809": "The three discounted pairs cost 105 versus 150, saving 45/150=30 percent. A source-context "
    "percentage rule permits an omitted sign but preserves explicit percentage scaling.",
    "e2h-amc-eval-1782": "The question explicitly asks the nearest whole percentage; 100×4/15 rounds to 27. A "
    "source-context percentage rule permits an omitted sign but preserves explicit percentage "
    "scaling.",
    "e2h-amc-train-969": "Removed volume is 8×27=216 out of 1200, or 18 percent. A source-context percentage rule "
    "permits an omitted sign but preserves explicit percentage scaling.",
    "e2h-amc-eval-1590": "Trip cost scales by 1.2/1.5=.8, yielding 20 percent savings. A source-context percentage rule "
    "permits an omitted sign but preserves explicit percentage scaling.",
    "e2h-amc-train-767": "The meetings occupy 135 of 540 minutes, or 25 percent. A source-context percentage rule "
    "permits an omitted sign but preserves explicit percentage scaling.",
    "e2h-amc-eval-2474": "The question explicitly asks the nearest whole percentage; 100×23/99 rounds to 23. A "
    "source-context percentage rule permits an omitted sign but preserves explicit percentage "
    "scaling.",
    "e2h-amc-eval-1150": "The integer percentage coefficient n must exceed 28.75, so its minimum is 29. A "
    "source-context percentage rule permits an omitted sign but preserves explicit percentage "
    "scaling.",
    "e2h-amc-eval-619": "The question explicitly asks the nearest whole percentage; 100×36/70 rounds to 51. A "
    "source-context percentage rule permits an omitted sign but preserves explicit percentage "
    "scaling.",
}

EXCLUSIONS = {
    "e2h-amc-eval-1674": {
        "original_gold": "24,28,32",
        "reason": "The problem allows integer die labels without a positivity restriction, but its "
        "finite reference set assumes positive labels. Opposite integer translations of the "
        "two dice preserve all sum probabilities while changing their totals. Do not "
        "silently add the missing restriction.",
        "problem_sha256": "af7d1e458770d29321d04423d75356184228b908783a53d03f794b2d7ca16ff3",
        "solution_sha256": "7f7afc9aee20100c2f4cb219815de3fa32489f3fdca7190ee5addd968dc25085",
    },
    "e2h-amc-train-779": {
        "original_gold": "\\frac{3}{1003}",
        "reason": "The prompt does not define the townspeople/goons game, random selection rules, "
        "day/night transitions, or stopping rule; these are assumed from omitted shared "
        "context.",
        "problem_sha256": "1fa465d5e3c60b1c7af3bce5665364e5fc92b663c798a286e2a8a98e6a8f743f",
        "solution_sha256": "f215b7a715d8522036b02378af71a5f67203c5438b45d2684eeb128fad7bf9e2",
    },
    "e2h-amc-train-926": {
        "original_gold": "\\frac{1}{3}",
        "reason": "The prompt defines the Jester variation but omits the underlying townspeople/goons "
        "game and random day/night selection rules.",
        "problem_sha256": "6ccc8a7362aa89cc49c639cb2a48f6c15f267faff14a3850e8af6ed69bbdaba7",
        "solution_sha256": "01cfc8d995e1403256afac0d6e56c240b8e9cc039383cf5c0b6a014f76ce2fe6",
    },
    "e2h-amc-eval-229": {
        "original_gold": "3",
        "reason": "The townspeople/goons winning probability cannot be determined without the omitted "
        "game and random elimination rules; the source solution refers to a previous "
        "question.",
        "problem_sha256": "f5310ee654c8bf1e493c2c56cfc5a527b60fb9fe261cc2abb575e9f7233889d7",
        "solution_sha256": "1af3059fce3c787695f676f47a0913a9fb5ae5503b1b3d976e65723f09039ac9",
    },
    "e2h-amc-eval-628": {
        "original_gold": "\\frac{1}{3}",
        "reason": "The prompt asks a winning probability but supplies no rules for the "
        "townspeople/goons game or random choice mechanism.",
        "problem_sha256": "ab369acd5b59ef7c0a3194482a88f5cc0cd0a867d608120c4385b8478de85873",
        "solution_sha256": "885b4cf582f91f7180ef5ea1db9dcd844227e25bfbe9ffa74f81430a440a9288",
    },
    "e2h-amc-eval-760": {
        "original_gold": "6",
        "reason": "The prompt supplies initial counts but not the underlying townspeople/goons game, "
        "random elimination rules, or winning conditions.",
        "problem_sha256": "29d6de6d10af7abb7725769d3ea0dd2cd7c6ad9191dfe9737f7a7e93c883ff20",
        "solution_sha256": "890811b067fe1d6c6d222ccbbec44219140d34cbe693ede640ed01e8df5a6ccf",
    },
    "e2h-amc-eval-955": {
        "original_gold": "[201,400]",
        "reason": "The question asks which interval contains n but the answer choices are absent. "
        "Although n=297 can be derived, many intervals contain it; the reference[201, 400] "
        "is not uniquely specified.",
        "problem_sha256": "03979d64a4fa90654d25e4cbe6858209a740beeb41cc3a215b452417efa1f68c",
        "solution_sha256": "e8b6240065558965a06aa423fb1cc17042bb3f9f075e100b92f96a7b5a51b807",
    },
    "e2h-amc-eval-1015": {
        "original_gold": "2",
        "reason": "The prompt is a fragment: the ball, trajectory, wall geometry and referenced "
        "diagram are absent. The distance is not determined by the provided input.",
        "problem_sha256": "25066540440ddac8d4deb72f94801d642ea16529ae1524d343edb43c0942aa0e",
        "solution_sha256": "0cfb7a8913a416b744c43dd0e5644fa3a6a6e5c36cc4d0cc07b91c2ae9dfa94b",
    },
    "e2h-amc-eval-800": {
        "original_gold": "422000",
        "reason": "An undated Google result-count question is time- and search-state-dependent; the "
        "historical value 422000 is not determined by the input and cannot serve as a "
        "reproducible mathematical reference.",
        "problem_sha256": "8626b57cc1e416eec8b81ba1a027f74bb6c4d9772eda7a40c728f590ecfdde5e",
        "solution_sha256": "bee832c238433483fc33a4971aa6f64efe7dcc1d45490eaf47c55ee1761faf94",
    },
    "e2h-amc-eval-2697": {
        "original_gold": "83",
        "reason": "The nearest-choice percentage question omits its choices and states no rounding "
        "precision. The exact percentage is 250/3, while 83 is only a particular "
        "approximation; no tolerance is invented.",
        "problem_sha256": "12df3a27f449e2bedd9233c3b9b34b9f0ddf2aa4a745ab75079b95013cb8eeee",
        "solution_sha256": "f5279f371181d22947d5d7dc215554c40b42e137fd2e17aa125cf7e5c4861246",
    },
    "e2h-amc-eval-1967": {
        "original_gold": "17",
        "reason": "The unfinished closest-to percentage prompt omits its answer choices and rounding "
        "precision. The exact increase is 50/3 percent; 17 is not uniquely specified by the "
        "input.",
        "problem_sha256": "23e858b639a70074c362c57002577d04e0da2fc531bad438da8fad29764241ca",
        "solution_sha256": "e6c53e229fcda5dca88a23099c99999729eae3bb8e14a4a66d6a2ad0438c77e0",
    },
    "e2h-amc-eval-258": {
        "original_gold": "6",
        "reason": "The nearest-choice area question omits choices and gives no rounding precision. The "
        "finite geometric sum is 6×(1−4^(−100)); the reference 6 is an unspecified "
        "approximation.",
        "problem_sha256": "5df08058e7e349466f1e02d251f396d28a9555b7cb2353d38e2170703c1a205f",
        "solution_sha256": "3d615d5b40b32a3c862f9601a6b6fece22cf3498ad7c55345edf2e9943949ef7",
    },
}

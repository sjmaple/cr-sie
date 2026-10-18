"""Audit exact LaTeX equivalence after the frozen mathematical grader runs.

The recorded grader rejects the valid unbraced exponent in ``e^e - 1``.
Keep its original result and apply only exact textual equality after bracing
single-token exponents and removing whitespace. No policies are refitted.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

import economics
import economics_policy
import learn
from cloud import digest


def canonical_latex(value):
    # Restrict this fallback to elementary expressions. Text commands can
    # contain meaningful spaces and require the mathematical parser instead.
    if not re.fullmatch(r"[A-Za-z0-9+\-*/^{}().= \t\r\n]+", value.strip().strip("$")):
        return None
    value = re.sub(r"\s+", "", value.strip().strip("$"))
    return re.sub(r"\^([A-Za-z0-9])", r"^{\1}", value)


def audit(directory):
    result = economics.read(directory / "economic-results.json")
    if "syntax_grading_audit" in result:
        raise ValueError("Run the frozen scorer before auditing its result")
    learn.write(directory / "economic-results-unadjusted.json", result)
    inputs = economics.read(directory / "inputs.json")
    settings = economics.read(directory / "answer-settings.json")
    gold = economics.read(directory / "gold-for-independent-scoring-only.json")
    records = economics.answer_records(directory, {r["id"]: r["request"] for r in inputs["test"]}, settings["arms"])
    plan, _ = economics.frozen_policies(directory)
    if (
        digest(plan) != result["evaluation_plan_sha256"]
        or digest([records[key] for key in sorted(records)]) != result["test_answer_records_sha256"]
        or digest(gold) != economics.read(directory / "plan.json")["gold_sha256"]
    ):
        raise ValueError("Grading audit inputs differ from the frozen evaluation")
    original_metrics = result["metrics"]
    corrections = []
    for index, row in enumerate(inputs["test"]):
        for alias in ("cheap", "strong"):
            call = records[(row["id"], settings[f"{alias}_arm"])]
            answer = economics.final_answer(call["answer"]) or ""
            if (
                not result["arrays"][f"q_{alias}"][index]
                and canonical_latex(answer)
                and canonical_latex(answer) == canonical_latex(gold[row["id"]])
            ):
                result["arrays"][f"q_{alias}"][index] = 1
                corrections.append(
                    {
                        "id": row["id"],
                        "arm": alias,
                        "answer_record_sha256": digest(call),
                        "final_answer": answer,
                        "reference": gold[row["id"]],
                        "canonical_latex": canonical_latex(answer),
                        "original_correct": False,
                        "audited_correct": True,
                    }
                )
    qc, qs, cc, cs = (np.asarray(result["arrays"][key]) for key in ("q_cheap", "q_strong", "c_cheap", "c_strong"))
    overhead = np.asarray(result["router_credits"])
    result["metrics"] = {
        name: economics_policy.evaluation_metrics(
            np.asarray(route, dtype=bool), qc, qs, cc, cs, 0 if name.startswith("always_") else overhead
        )
        for name, route in result["routes"].items()
    }
    difficulty = np.asarray([r["label"] for r in inputs["test"]])
    for label in (0, 1):
        result["accuracy_by_human_difficulty"][str(label)].update(
            cheap_accuracy=float(qc[difficulty == label].mean()),
            strong_accuracy=float(qs[difficulty == label].mean()),
            strong_advantage_pp=float((qs - qc)[difficulty == label].mean() * 100),
        )
    primary = result["metrics"][result["primary"]]
    result["primary_gate"].update(
        point_estimate_pass=primary["saving_vs_strong_pct"] >= 25 and primary["quality_loss_pp"] <= 1,
        noninferiority_supported_at_1pp=primary["harmful_switch_rate_upper95_pp"] <= 1,
        cost_saving_supported=primary["saving_vs_strong_95ci_pct"][0] > 0,
    )
    result["syntax_grading_audit"] = {
        "source_sha256": digest(Path(__file__).read_text()),
        "timing": "Exact-syntax audit added after frozen test scoring; original scores are retained separately.",
        "method": "Exact string equality after whitespace removal and bracing single-token exponents.",
        "corrections": corrections,
        "policies_refitted": False,
        "original_format_audit_retained": True,
        "costs_and_net_quality_gaps_unchanged": all(
            original_metrics[name][key] == row[key]
            for name, row in result["metrics"].items()
            for key in ("credits_per_request", "quality_loss_pp", "quality_loss_95ci_pp")
        ),
    }
    learn.write(directory / "economic-results.json", result)
    learn.write(directory / "grading-syntax-audit.json", result["syntax_grading_audit"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    result = audit(parser.parse_args().run)
    print(f"Exact-syntax corrections: {len(result['syntax_grading_audit']['corrections'])}; original scores retained.")

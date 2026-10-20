"""Present the fixed benefit-router comparisons without selecting a winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data import file_hash, write

TARGETS = (1, 0, 2, 5)
LABELS = {
    "base": "Laya, unchanged",
    "lora": "Laya + LoRA, 511 examples",
    "embedding": "Qwen embeddings + ridge, 511 examples",
    "competition": "Competition name only (control)",
}


def combine(embedding, native, control):
    documents = (embedding, native, control)
    formats = (
        "sie-solver-benefit-report-v1",
        "sie-solver-benefit-native-report-v1",
        "sie-benefit-competition-control-report-v1",
    )
    for document, expected in zip(documents, formats, strict=True):
        if document.get("format") != expected:
            raise ValueError("Unexpected experiment report format")
        if (
            document.get("n") != len(embedding["ids"])
            or document.get("ids") != embedding["ids"]
            or len(set(document["ids"])) != document["n"]
            or document.get("test_grades_sha256") != embedding["test_grades_sha256"]
            or document.get("primary_max_quality_loss_pp") != 1
            or document.get("training_rows") != 511
        ):
            raise ValueError(
                "Reports must use the same ordered problems, paired answers, training budget and primary target"
            )
    policies = {
        "base": native["models"]["base"],
        "lora": native["models"]["lora"],
        "embedding": embedding["policies"],
        "competition": control["policies"],
    }
    if any(set(values) != {str(target) for target in TARGETS} for values in policies.values()):
        raise ValueError("Every predeclared quality target must remain in the comparison")
    for values in policies.values():
        for policy in values.values():
            metric = policy["metrics"]
            if (
                metric["n"] != embedding["n"]
                or metric["always_strong_answer_accuracy"] != embedding["baselines"]["always_strong"]["answer_accuracy"]
            ):
                raise ValueError("Policy metrics disagree about the comparison population or baseline")
    return {
        "format": "sie-benefit-comparison-v1",
        "n": embedding["n"],
        "ids": embedding["ids"],
        "test_grades_sha256": embedding["test_grades_sha256"],
        "primary_target_pp": 1,
        "baselines": embedding["baselines"],
        "policies": policies,
        "native_selected_epoch": native["selected_epoch"],
        "cost_scope": "Recorded answer-model and embedding API charges. Native router hosting is unknown. Common application and small-classifier CPU costs are unpriced.",
        "interpretation": "Every target was selected on calibration and evaluated unchanged on the same test answers. A calibration loss target is not a test guarantee. Do not select a new deployment rule using this test table.",
    }


def row(label, metrics):
    net = "net_total_cost_usd_per1000" in metrics
    cost = metrics["net_total_cost_usd_per1000"] if net else metrics["total_solver_cost_usd_per1000"]
    savings = metrics["net_savings_pct"] if net else metrics["gross_solver_savings_pct"]
    price = f"${cost:.4f}" + ("" if net else " + routing hosting")
    saved = "unpriced" if savings is None else f"{savings:.2f}%" + ("" if net else " before routing hosting")
    return (
        f"| {label} | {100 * metrics['answer_accuracy']:.2f}% | {metrics['gap_vs_strong_pp']:+.2f} pp | "
        f"{price} | {saved} |"
    )


def markdown(report):
    lines = [
        "# Answer quality and money saved",
        "",
        f"The same {report['n']} test problems and paired answers are used in every row.",
        "",
        "Loss means percentage points of answer accuracy below always using Qwen 27B; negative loss is an improvement.",
        "",
    ]
    for target in TARGETS:
        lines.extend(
            [
                f"## {'Primary' if target == 1 else 'Additional'} calibration target: {target} percentage point(s)",
                "",
                "| Method | Answer accuracy | Loss vs always-strong | Cost per 1,000 | Savings |",
                "| --- | ---: | ---: | ---: | ---: |",
                row("Always Qwen 27B", report["baselines"]["always_strong"]),
                row("Always Qwen 4B", report["baselines"]["always_cheap"]),
                *[row(LABELS[name], policies[str(target)]["metrics"]) for name, policies in report["policies"].items()],
                "",
            ]
        )
    lines.extend([report["cost_scope"], "", report["interpretation"], ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", required=True, type=Path)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = (args.embedding, args.native, args.control)
    result = combine(*(json.loads(path.read_text()) for path in paths))
    result["source_reports_sha256"] = {
        name: file_hash(path) for name, path in zip(("embedding", "native", "control"), paths, strict=True)
    }
    write(args.output / "comparison.json", result)
    rendered = markdown(result)
    (args.output / "comparison.md").write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()

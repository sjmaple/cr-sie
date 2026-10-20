"""Compare the frozen outcome routers on one common set of recorded answers.

Keep both training sizes and every declared quality target. This renderer
neither chooses a winning model nor changes a deployment cutoff.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from data import file_hash, write

TARGETS = (1, 0, 2, 5)
INTERVAL_NOTE = (
    "Intervals are descriptive paired question-level 95% bootstrap intervals for the fixed policies and recorded "
    "answers. They assume independent questions and do not cover shared-paper dependence, policy selection, "
    "solver randomness, or future prices and traffic. A zero-event interval is not a risk guarantee."
)
COST_NOTE = (
    "Dollar amounts are scaled to 10,000 requests with the observed question mix and pinned prices; they are not "
    "a forecast of future traffic. API savings subtract measured embedding charges where applicable. Native "
    "threshold-router hosting is unpriced: its allowance is the most it could cost before the solver savings "
    "disappear. A static choice bypasses the router. Training, teacher labels, and local application/classifier "
    "CPU costs are separate; these tables do not measure warm serving latency or throughput."
)


def _compatible(document, reference, expected_format, training_rows):
    if (
        document.get("format") != expected_format
        or document.get("n") != reference["n"]
        or document.get("ids") != reference["ids"]
        or len(set(document["ids"])) != document["n"]
        or document.get("test_grades_sha256") != reference["test_grades_sha256"]
        or document.get("primary_max_quality_loss_pp") != 1
        or document.get("training_rows") != training_rows
    ):
        raise ValueError("Reports need the same ordered test questions, answer grades and declared training size")


def combine(heads, natives, teacher=None):
    if len(heads) != 2 or len(natives) != 2:
        raise ValueError("Include both training sizes for both designs")
    reference = heads[0]
    if (
        reference.get("n", 0) <= 0
        or len(reference.get("ids", [])) != reference["n"]
        or heads[0].get("training_rows", 0) <= 0
        or heads[0]["training_rows"] >= heads[1].get("training_rows", 0)
    ):
        raise ValueError("Need a nonempty test and increasing original/expanded training sizes")
    methods, selected_epochs = {}, {}
    for index, (head, native) in enumerate(zip(heads, natives, strict=True)):
        size = head["training_rows"]
        _compatible(head, reference, "sie-solver-outcome-report-v1", size)
        _compatible(native, reference, "sie-solver-outcome-native-report-v1", size)
        if head.get("baselines") != reference.get("baselines") or set(native.get("models", {})) != {"base", "lora"}:
            raise ValueError("Both studies must retain identical baselines and both native variants")
        suffix = "original" if index == 0 else "expanded"
        selected_epochs[suffix] = native["selected_epoch"]
        for variant in ("base", "lora"):
            methods[f"{variant}_{suffix}"] = {
                "label": f"Laya {'unchanged; cost estimates from' if variant == 'base' else '+ LoRA;'} {size:,} examples",
                "training_rows": size,
                "design": "native",
                "label_source": "reviewed benchmark answers",
                "policies": native["models"][variant],
            }
        methods[f"embedding_{suffix}"] = {
            "label": f"Qwen embeddings + classifier; {size:,} examples",
            "training_rows": size,
            "design": "embedding",
            "label_source": "reviewed benchmark answers",
            "policies": head["policies"],
        }
    if teacher is not None:
        size = teacher.get("training_rows", 0)
        _compatible(teacher, reference, "sie-solver-outcome-teacher-report-v1", size)
        if (
            teacher.get("baselines") != reference.get("baselines")
            or not teacher.get("teacher_model")
            or not 0 < size <= heads[1]["training_rows"]
        ):
            raise ValueError("Teacher comparison needs identical evaluation references and an explicit teacher model")
        methods["embedding_teacher"] = {
            "label": f"Qwen embeddings + classifier; {size:,} teacher-labeled examples",
            "training_rows": size,
            "design": "embedding",
            "label_source": teacher["teacher_model"],
            "policies": teacher["policies"],
        }
        if "matched_benchmark_control" in teacher:
            matched = teacher["matched_benchmark_control"]
            if (
                matched["training_rows"] != size
                or matched["baselines"] != reference["baselines"]
                or matched["ids"] != reference["ids"]
            ):
                raise ValueError("Teacher and matched benchmark heads must use identical training and test populations")
            methods["embedding_teacher_control"] = {
                "label": f"Qwen embeddings + classifier; same {size:,} examples with benchmark labels",
                "training_rows": size,
                "design": "embedding",
                "label_source": "reviewed benchmark answers",
                "policies": matched["policies"],
            }
    baseline = reference["baselines"]["always_strong"]
    for method in methods.values():
        if set(method["policies"]) != {str(target) for target in TARGETS}:
            raise ValueError("Retain all four declared quality targets")
        for policy in method["policies"].values():
            metric = policy["metrics"]
            if (
                metric["n"] != reference["n"]
                or metric["always_strong_answer_accuracy"] != baseline["answer_accuracy"]
                or not math.isclose(
                    metric["gap_vs_strong_pp"],
                    100 * (baseline["answer_accuracy"] - metric["answer_accuracy"]),
                    abs_tol=1e-9,
                )
                or (method["design"] == "embedding" and "net_savings_pct" not in metric)
                or (method["design"] == "native" and policy["mode"] == "threshold" and "net_savings_pct" in metric)
            ):
                raise ValueError("Quality metrics or known/unknown router costs disagree with the method")
    return {
        "format": "sie-outcome-comparison-v1",
        "n": reference["n"],
        "ids": reference["ids"],
        "test_grades_sha256": reference["test_grades_sha256"],
        "primary_target_pp": 1,
        "baselines": reference["baselines"],
        "methods": methods,
        "selected_native_epochs": selected_epochs,
        "cost_scope": "Embedding rows include recorded answer-model and embedding charges. Native threshold rows show answer-model cost plus unpriced router hosting. Training, teacher-label acquisition and common application CPU costs are separate.",
        "interpretation": "All cutoffs were selected on calibration and applied unchanged to the same test answers. A calibration quality-loss target is not a guarantee on new questions. All declared targets remain visible; this table does not select a new rule using test outcomes.",
    }


def _interval(metrics, key, *, signed=False):
    values = metrics.get("paired_bootstrap_95ci", {}).get(key)
    if values is None:
        return "interval unavailable"
    spec = "+.2f" if signed else ".2f"
    return f"[{format(values[0], spec)}, {format(values[1], spec)}]"


def policy_label(label, mode):
    """Make a calibrated static bypass visible without changing its policy."""
    if mode in ("always_strong", "always_cheap"):
        choice = "27B" if mode == "always_strong" else "4B"
        return f"{label} (always {choice}; router skipped)"
    return label


def quality_table(entries, *, estimated=False):
    """Render already computed metrics; this performs no policy selection."""
    lines = [
        "| Method | Answer accuracy | Quality gap, pp [95% interval] | Savings [95% interval] |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, metrics in entries:
        net = "net_savings_pct" in metrics
        savings_key = "net_savings_pct" if net else "gross_solver_savings_pct"
        savings = metrics[savings_key]
        scope = ("estimated API" if estimated else "API") if net else "solver only; hosting unpriced"
        saved = "unavailable" if savings is None else f"{savings:.2f}% {_interval(metrics, savings_key)} ({scope})"
        lines.append(
            f"| {label} | {100 * metrics['answer_accuracy']:.2f}% | "
            f"{metrics['gap_vs_strong_pp']:+.2f} {_interval(metrics, 'gap_vs_strong_pp', signed=True)} | {saved} |"
        )
    return lines


def _dollars_per10k(value):
    return "unavailable" if value is None else f"${10 * value:,.3f}"


def cost_table(entries, *, estimated=False):
    """Keep solver savings, paid routing, and an unknown hosting bill distinct."""
    lines = [
        "All dollar amounts below are per **10,000 requests**.",
        "",
        "| Method | Answer models | Router | Total | Solver-only savings | API savings / hosting allowance |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, metrics in entries:
        solver_cost = _dollars_per10k(metrics["total_solver_cost_usd_per1000"])
        gross = _dollars_per10k(metrics.get("gross_solver_savings_usd_per1000"))
        if "net_total_cost_usd_per1000" in metrics:
            routing = _dollars_per10k(metrics.get("router_cost_usd_per1000"))
            total = _dollars_per10k(metrics["net_total_cost_usd_per1000"])
            saved = _dollars_per10k(metrics.get("net_savings_usd_per1000")) + (
                " estimated API savings" if estimated else " API savings"
            )
        else:
            routing, total = "hosting unpriced", solver_cost + " + hosting"
            allowance = metrics.get("break_even_router_usd_per1000")
            if allowance is None:
                saved = "hosting allowance unavailable"
            elif allowance > 0:
                saved = "hosting must cost < " + _dollars_per10k(allowance)
            elif allowance == 0:
                saved = "no positive hosting allowance"
            else:
                saved = "already " + _dollars_per10k(-allowance) + " more before hosting"
        lines.append(f"| {label} | {solver_cost} | {routing} | {total} | {gross} | {saved} |")
    return lines


def markdown(report):
    lines = [
        "# Money saved and answer quality",
        "",
        f"Every row uses the same {report['n']} test questions and recorded model answers.",
        "Quality loss is the drop in answer accuracy, in percentage points, compared with always using Qwen 27B.",
        "A negative gap means higher accuracy; a positive gap means lost accuracy.",
        "",
    ]
    for target in TARGETS:
        entries = [
            ("Always Qwen 27B", report["baselines"]["always_strong"]),
            ("Always Qwen 4B", report["baselines"]["always_cheap"]),
            *[
                (
                    policy_label(method["label"], method["policies"][str(target)]["mode"]),
                    method["policies"][str(target)]["metrics"],
                )
                for method in report["methods"].values()
            ],
        ]
        lines.extend(
            [
                f"## {'Primary' if target == 1 else 'Additional'} calibration target: {target} percentage point(s)",
                "",
                *quality_table(entries),
                "",
            ]
        )
        if target == 1:
            lines.extend([*cost_table(entries), ""])
    lines.extend(
        [
            report["cost_scope"],
            "",
            COST_NOTE,
            "",
            INTERVAL_NOTE,
            "",
            "The JSON also retains harmful switches, helpful switches, and per-1,000 metrics for reuse.",
            "",
            report["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", nargs=2, type=Path, required=True, metavar=("ORIGINAL", "EXPANDED"))
    parser.add_argument("--native", nargs=2, type=Path, required=True, metavar=("ORIGINAL", "EXPANDED"))
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    def read(path):
        return json.loads(path.read_text())

    result = combine(
        [read(path) for path in args.heads],
        [read(path) for path in args.native],
        read(args.teacher) if args.teacher else None,
    )
    named = dict(
        zip(
            ("head_original", "head_expanded", "native_original", "native_expanded"),
            [*args.heads, *args.native],
            strict=True,
        )
    )
    if args.teacher:
        named["teacher"] = args.teacher
    result["source_reports_sha256"] = {name: file_hash(path) for name, path in named.items()}
    write(args.output / "comparison.json", result)
    rendered = markdown(result)
    (args.output / "comparison.md").write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()

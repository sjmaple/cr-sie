"""Draw the saved routing comparison in dollars and answer-quality loss."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

ORDER = (
    "base",
    "base_arithmetic_500",
    "base_arithmetic_2000",
    "lora_calibration_only",
    "lora_500",
    "lora_2000",
    "embedding_calibration_only",
    "embedding_500",
    "embedding_2000",
    "embedding_teacher_2000",
    "embedding_teacher_matched_reference",
)
LABELS = {
    "always_strong": "Always Qwen 27B",
    "always_cheap": "Always Qwen 4B",
    "base": "Laya, unchanged; original cost estimates",
    "base_arithmetic_500": "Laya, unchanged; costs from 500 examples",
    "base_arithmetic_2000": "Laya, unchanged; costs from 2,000 examples",
    "lora_calibration_only": "Existing LoRA, new cutoff",
    "lora_500": "Laya + LoRA, 500 examples",
    "lora_2000": "Laya + LoRA, 2,000 examples",
    "embedding_calibration_only": "Existing embedding router, new cutoff",
    "embedding_500": "Embeddings + classifiers, 500 examples",
    "embedding_2000": "Embeddings + classifiers, 2,000 examples",
    "embedding_teacher_2000": "Embeddings + teacher labels",
    "embedding_teacher_matched_reference": "Same questions, reference labels",
}


def label_for(report, name):
    label = LABELS.get(name, name)
    if name in ("embedding_teacher_2000", "embedding_teacher_matched_reference"):
        count = report["methods"][name]["training_rows"]
        return f"{label}, {count:,} examples"
    return label


def plot(report, output, *, target="1", cost_controls=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    if report.get("format") != "sie-arithmetic-adaptation-report-v1":
        raise ValueError("Expected the saved arithmetic adaptation report")
    if cost_controls is not None:
        if (
            cost_controls.get("format") != "sie-arithmetic-cost-controls-report-v1"
            or cost_controls.get("core_frozen_sha256") != report["frozen_sha256"]
            or cost_controls.get("test_grades_sha256") != report["test_grades_sha256"]
            or cost_controls.get("n") != report["n"]
            or set(cost_controls.get("methods", {})) != {"base_arithmetic_500", "base_arithmetic_2000"}
        ):
            raise ValueError("Cost controls must use the same frozen study and test answers")
        report = copy.deepcopy(report)
        report["methods"].update(cost_controls["methods"])
    rows = [(name, report["baselines"][name]) for name in ("always_strong", "always_cheap")]
    for name in ORDER:
        if name in report["methods"]:
            rows.append((name, report["methods"][name]["policies"][str(target)]["metrics"]))
    if len(rows) == 2:
        raise ValueError("No evaluated router methods found")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (costs, gaps) = plt.subplots(
        1, 2, figsize=(12.5, max(6.5, len(rows) * 0.48 + 2.4)), sharey=True,
        gridspec_kw={"width_ratios": [1.1, 1]},
    )
    for index, (name, metric) in enumerate(rows):
        known = metric.get("net_total_cost_usd_per1000") is not None
        cost = 10 * metric["net_total_cost_usd_per1000" if known else "total_solver_cost_usd_per1000"]
        color = "#325b81" if "embedding" in name or "teacher" in name else "#a76236"
        if name.startswith("always_"):
            color = "#68737d"
        costs.barh(index, cost, height=0.58, color=color, alpha=0.85, hatch=None if known else "//")
        costs.annotate(
            f"${cost:.2f}" + (" + hosting" if not known else ""),
            (cost, index), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9,
        )
        gap = metric["gap_vs_strong_pp"]
        interval = metric["paired_bootstrap_95ci"]["gap_vs_strong_pp"]
        gaps.errorbar(
            gap, index, xerr=[[max(0, gap - interval[0])], [max(0, interval[1] - gap)]],
            fmt="o", color=color, markersize=6, capsize=3,
        )
        gaps.annotate(f"{gap:.2f}", (interval[1], index), xytext=(7, 0), textcoords="offset points", va="center", fontsize=9)
    costs.set_yticks(range(len(rows)), [label_for(report, name) for name, _ in rows])
    costs.invert_yaxis()
    costs.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:g}"))
    costs.set_xlim(0, costs.get_xlim()[1] * 1.37)
    costs.set_title("Cost per 10,000 requests", loc="left", fontweight="bold", pad=15)
    costs.set_xlabel("Lower costs less")
    gaps.axvline(0, color="#68737d", linestyle="--", linewidth=1)
    gaps.axvline(float(target), color="#a1a8ae", linestyle=":", linewidth=1)
    gaps.set_xlim(min(-0.5, gaps.get_xlim()[0]), gaps.get_xlim()[1] + 0.8)
    gaps.set_title("Answer accuracy lost (percentage points)", loc="left", fontweight="bold", pad=15)
    gaps.set_xlabel("Lower preserves more quality; bars show 95% intervals")
    for axis in (costs, gaps):
        axis.grid(axis="x", alpha=0.15)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
    fig.suptitle(f"Routing on {report['n']:,} held-out arithmetic questions", fontsize=16, y=0.99)
    fig.text(
        0.02, 0.055,
        f"Cutoffs selected for a {target}-point calibration target; measured test gaps can differ. "
        "Costs use the saved Cloud token prices.\n"
        "Embedding bars include the embedding charge. Hatched bars exclude unpriced Laya hosting. "
        "Training is a separate one-time cost.",
        fontsize=9, color="#495560", va="bottom",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    for suffix in ("svg", "png"):
        fig.savefig(output / f"cost-and-quality-{target}pp.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=("0", "1", "2", "5"), default="1")
    parser.add_argument("--cost-controls", type=Path, help="Include the separate unchanged-Laya cost controls")
    args = parser.parse_args()
    plot(
        json.loads(args.report.read_text()), args.output, target=args.target,
        cost_controls=json.loads(args.cost_controls.read_text()) if args.cost_controls else None,
    )


if __name__ == "__main__":
    main()

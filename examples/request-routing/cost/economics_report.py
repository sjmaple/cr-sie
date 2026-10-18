"""Present predeclared economic comparisons without choosing winners on test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import economics


def comparisons(directory):
    plan, groups = economics.frozen_policies(directory)
    chosen = {
        "Always 27B": "always_strong",
        "Always 4B": "always_cheap",
        "Difficulty kNN · 64": "difficulty_knn64",
        "Difficulty ridge · 512": "difficulty_ridge512",
    }
    for source, label in (
        ("teacher_agreement", "Answer agreement"),
        ("teacher_review", "Teacher review"),
        ("gold_benefit", "Answer-key control"),
    ):
        if source not in groups:
            continue
        frozen = groups[source]
        chosen[label + " · 1pp budget"] = source + "_" + frozen["primary"]
        largest = max(p["requested_labels"] for p in frozen["policies"].values())
        for budget in (2, 5):
            name = min(
                (p["validation_mean_credits"], -p["validation_relative_quality"], name)
                for name, p in frozen["policies"].items()
                if p["requested_labels"] == largest and p["quality_loss_target_pp"] == budget
            )[2]
            chosen[f"{label} · {budget}pp budget"] = source + "_" + name
    return plan, chosen


def summary(directory):
    result = economics.read(directory / "economic-results.json")
    plan, chosen = comparisons(directory)
    return {
        "primary": result["primary"],
        "frozen_at_utc": plan["frozen_at_utc"],
        "primary_gate": result["primary_gate"],
        "selection": "Largest training prefix; methods and thresholds chosen using validation only, before test answers.",
        "comparisons": {label: {"policy": name, **result["metrics"][name]} for label, name in chosen.items()},
        "mean_router_usd_per_1000": sum(result["router_credits"]) / len(result["router_credits"]) * 0.01,
        "acquisition": result["acquisition"],
        "syntax_grading_audit": result.get("syntax_grading_audit"),
    }


def plot(directory, report):
    import matplotlib.pyplot as plt

    figure, (cost, quality) = plt.subplots(1, 2, figsize=(12.5, 6.2), layout="constrained", sharey=True)
    rows = {
        k: v
        for k, v in report["comparisons"].items()
        if k
        in (
            "Always 27B",
            "Always 4B",
            "Difficulty ridge · 512",
            "Teacher review · 1pp budget",
            "Teacher review · 5pp budget",
        )
    }
    labels = list(rows)
    positions = list(range(len(rows)))
    colors = ["#777777", "#b19cce", "#8dabc8", "#15665f", "#52a397"]
    costs = [r["usd_per_1000"] for r in rows.values()]
    gaps = [r["quality_loss_pp"] for r in rows.values()]
    intervals = [r["quality_loss_95ci_pp"] for r in rows.values()]
    bars = cost.barh(positions, costs, color=colors, height=0.58)
    cost.bar_label(bars, labels=[f"${v:.2f}" for v in costs], padding=5)
    cost.set(yticks=positions, yticklabels=labels, xlabel="Recorded serving cost / 1,000 requests")
    cost.set_xlim(0, max(costs) * 1.22)
    cost.invert_yaxis()
    quality.axvline(0, color="#777777", linewidth=1)
    for i, (gap, interval, color) in enumerate(zip(gaps, intervals, colors, strict=True)):
        quality.errorbar(gap, i, xerr=[[gap - interval[0]], [interval[1] - gap]], fmt="o", color=color, capsize=4)
        quality.annotate(
            f"{gap:+.2f}pp", (gap, i), xytext=(0, -16), textcoords="offset points", ha="center", fontsize=10
        )
    quality.set_xlabel("Answer-accuracy loss vs always 27B (percentage points)")
    for axis in (cost, quality):
        axis.spines[["right", "top", "bottom"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="x", alpha=0.15)
    figure.suptitle("Routing: measured cost and answer quality", fontsize=20, fontweight="bold")
    figure.supxlabel(
        "512 paired held-out questions · actual Cloud receipts including embeddings · paired 95% intervals\n"
        "Quality budgets are validation settings; the chart shows independently measured test outcomes.",
        fontsize=10,
    )
    for suffix in ("png", "svg"):
        figure.savefig(directory / f"economic-results.{suffix}", dpi=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    report = summary(args.run)
    (args.run / "economic-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("| Policy | $ / 1,000 | Saving | Answer accuracy | Quality loss (pp) |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for label, row in report["comparisons"].items():
        print(
            f"| {label} | ${row['usd_per_1000']:.2f} | {row['saving_vs_strong_pct']:.1f}% | "
            f"{row['accuracy']:.2%} | {row['quality_loss_pp']:+.2f} |"
        )
    if args.plot:
        plot(args.run, report)


if __name__ == "__main__":
    main()

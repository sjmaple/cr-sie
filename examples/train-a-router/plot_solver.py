"""Plot the measured answer-quality/cost tradeoff from a frozen solver report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = {
    "base": "Laya, unchanged",
    "lora512": "Laya + LoRA, 512 examples",
    "lora2581": "Laya + LoRA, 2,581 examples",
    "ridge64": "Qwen + classifier, 64 examples",
    "ridge512": "Qwen + classifier, 512 examples",
}


def plot(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if report.get("quality_only") or not report["baselines"]["always_strong"].get(
        "always_strong_solver_cost_usd_per1000"
    ):
        raise ValueError("A cost figure requires recorded solver prices")
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    colors = plt.get_cmap("tab10").colors
    for axis, title, calibrated in zip(
        axes,
        ("Original difficulty cutoffs", "Cutoffs selected for a 1-point calibration gap"),
        (False, True),
        strict=True,
    ):
        axis.axhline(0, color="#87929c", linewidth=1, linestyle="--")
        for name, marker, label in (("always_strong", "*", "Always Qwen 27B"), ("always_cheap", "D", "Always Qwen 4B")):
            metric = report["baselines"][name]
            axis.scatter(
                metric["gross_solver_savings_pct"],
                metric["gap_vs_strong_pp"],
                marker=marker,
                s=100,
                color="#263745",
                label=label,
                zorder=4,
            )
        for index, (name, stage) in enumerate(report["stages"].items()):
            metric = (
                stage["calibrated_cost_policies"]["1"]["metrics"] if calibrated else stage["original_difficulty_policy"]
            )
            x, y = metric["gross_solver_savings_pct"], metric["gap_vs_strong_pp"]
            low, high = metric["paired_bootstrap_95ci"]["gap_vs_strong_pp"]
            axis.errorbar(
                x,
                y,
                yerr=[[max(0, y - low)], [max(0, high - y)]],
                marker="o",
                markersize=6,
                capsize=3,
                color=colors[index],
                label=LABELS.get(name, name),
                linewidth=1.2,
            )
        axis.set_title(title, fontsize=11, pad=12)
        axis.set_xlabel("Answer-model cost saved (%) · higher saves more")
        axis.grid(alpha=0.15)
    axes[0].set_ylabel(
        "Answer accuracy lost vs always-27B (percentage points)\nLower preserves more quality; below zero improves it"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=3, frameon=False)
    fig.suptitle(f"Routing tradeoffs on the same {report['n']} problems", fontsize=15, y=0.98)
    fig.text(
        0.5,
        0.005,
        "Answer-model savings exclude router hosting. Bars: paired 95% bootstrap intervals. Calibration targets are not test guarantees.",
        ha="center",
        fontsize=9,
        color="#4e5963",
    )
    fig.tight_layout(rect=(0, 0.17, 1, 0.94))
    for suffix in ("png", "svg"):
        fig.savefig(output / f"answer-quality.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plot(json.loads(args.report.read_text()), args.output)


if __name__ == "__main__":
    main()

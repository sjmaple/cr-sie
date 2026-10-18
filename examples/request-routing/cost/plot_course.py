"""Plot the two measured learning paths from the post-training report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    import matplotlib.pyplot as plt
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.evidence / "report.json").read_text())
    labels = [
        "Qwen4B\nkNN · 64",
        "Qwen4B\nridge · 64",
        "Qwen4B\nridge · 512",
        "Laya\nbase",
        "Laya\nfull tune",
        "Laya\nLoRA + head",
    ]
    keys = ["knn64", "ridge64", "ridge512", "base", "full", "lora"]
    figure, axis = plt.subplots(figsize=(11, 5.8), layout="constrained")
    positions = np.arange(len(keys))
    for shift, split, color, title in [
        (-0.18, "reference", "#8dabc8", "Original test · 512"),
        (0.18, "confirmation", "#15665f", "Fresh confirmation · 512"),
    ]:
        values = [100 * report["splits"][split]["metrics"][key]["accuracy"] for key in keys]
        bars = axis.bar(positions + shift, values, 0.34, color=color, label=title)
        axis.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=4, fontsize=10)
    axis.set(xticks=positions, xticklabels=labels, ylim=(0, 105), ylabel="Routing accuracy (%)")
    axis.axvline(2.5, color="#cccccc", linewidth=1)
    axis.spines[["right", "top"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.18)
    axis.legend(loc="lower center", ncols=2, frameon=False)
    figure.suptitle("Two ways to learn a better routing policy", fontsize=20, fontweight="bold")
    axis.set_title(
        "Frozen embeddings + a small classifier  |  A 421M-parameter model trained on the task", fontsize=11, pad=18
    )
    figure.supxlabel(
        "Human-performance-derived competition-math difficulty. Fixed test sets; no test-based policy selection.",
        fontsize=10,
    )
    for suffix in ("png", "svg"):
        figure.savefig(args.evidence / f"course-results.{suffix}", dpi=180)


if __name__ == "__main__":
    main()

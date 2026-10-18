"""Plot a saved headroom.py diagnostic; install the optional analysis group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import headroom

COLORS = {"knn": "#0072B2", "ridge": "#D55E00", "cost_aware_ridge": "#009E73", "fixed_reranker": "#7B5BA7"}
LABELS = {
    "knn": "Growing kNN bank",
    "ridge": "Ridge predictor",
    "cost_aware_ridge": "Cost-aware ridge",
    "fixed_reranker": "Fixed reranker",
}


def plot(result, directory):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for col, (pair, data) in enumerate(result["pairs"].items()):
        top, bottom = axes[:, col]
        title = "GPT-6 Luna / GPT-6.1 Sol" if pair == "openai" else "Haiku 5.5 / Sonnet 5.5"
        levels = np.linspace(0.9, 1.0, 101)
        for name in ("knn64", "knn437", "ridge437", "fixed_reranker437", "gepa437"):
            if name not in data["frontiers"]:
                continue
            points = data["frontiers"][name]
            costs = [headroom.at_quality(points, t)["cost_per_1k"] for t in levels]
            label = {
                "knn64": "kNN: 64 feedback",
                "knn437": "kNN: 437 feedback",
                "ridge437": "Ridge: 437 feedback",
                "fixed_reranker437": "Fixed reranker",
                "gepa437": "GEPA: 437 (selected here)",
            }[name]
            color = {
                "knn64": COLORS["knn"],
                "knn437": COLORS["knn"],
                "ridge437": COLORS["ridge"],
                "fixed_reranker437": COLORS["fixed_reranker"],
                "gepa437": "#CC79A7",
            }[name]
            top.step(
                levels * 100, costs, where="post", label=label, color=color, linestyle="--" if name == "knn64" else "-"
            )
        top.plot(
            [100 * float(t) for t in data["oracle"]],
            [r["cost_per_1k"] for r in data["oracle"].values()],
            "k--",
            label="Perfect-information lower bound",
        )
        top.axhline(data["always_strong"]["cost_per_1k"], color="gray", linestyle=":", label="Always strong")
        top.axvline(98, color="gray", linewidth=0.7)
        top.set(title=title, xlabel="Required retained quality (%)", ylabel="Historical USD / 1,000 requests")
        top.legend(fontsize=8, loc="upper left")
        for method, color in COLORS.items():
            sizes = sorted({r["size"] for r in data["runs"] if r["method"] == method})
            rows = [
                [
                    r["matched_validation"]["cost_per_1k"]
                    for r in data["runs"]
                    if r["method"] == method and r["size"] == size and r["target"] == 0.98
                ]
                for size in sizes
            ]
            means = [np.mean(values) for values in rows]
            bottom.plot(sizes, means, "o-", color=color, label=LABELS[method])
            bottom.fill_between(
                sizes, [min(values) for values in rows], [max(values) for values in rows], color=color, alpha=0.09
            )
        bottom.axhline(data["oracle"]["0.98"]["cost_per_1k"], color="black", linestyle="--")
        bottom.set(xlabel="Graded training requests", ylabel="USD / 1,000 at ≥98% validation quality")
        bottom.legend(fontsize=8)
    figure.suptitle("Routing headroom and feedback curves — development diagnosis", fontsize=16)
    figure.supxlabel(
        "146 validation requests; cutoffs swept on validation (optimistic). Shading: range across 10 feedback orders, not a confidence interval.\nGEPA was already selected on this validation set. Oracle knows both answers; it is not deployable. No new test result.",
        fontsize=9,
    )
    for suffix in ("png", "svg"):
        figure.savefig(directory / f"headroom.{suffix}", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for col, (pair, data) in enumerate(result["pairs"].items()):
        for method, color in COLORS.items():
            sizes = sorted({r["size"] for r in data["runs"] if r["method"] == method})
            for row_index, field in enumerate(("cost_per_1k", "quality_vs_strong")):
                rows = [
                    [
                        r["validation"][field]
                        for r in data["runs"]
                        if r["method"] == method and r["size"] == size and r["target"] == 0.98
                    ]
                    for size in sizes
                ]
                factor = 100 if row_index else 1
                axes[row_index, col].plot(
                    sizes, [factor * np.mean(values) for values in rows], "o-", color=color, label=LABELS[method]
                )
                axes[row_index, col].fill_between(
                    sizes,
                    [factor * min(values) for values in rows],
                    [factor * max(values) for values in rows],
                    color=color,
                    alpha=0.09,
                )
        axes[0, col].set(title=pair, ylabel="Historical USD / 1,000")
        axes[0, col].legend(fontsize=8)
        axes[1, col].axhline(98, color="black", linestyle="--", label="Training quality target")
        axes[1, col].set(xlabel="Graded training requests", ylabel="Validation retained quality (%)")
    figure.suptitle("Do training-fitted cutoffs transfer to validation?", fontsize=16)
    figure.supxlabel(
        "Cutoffs fitted on training only, aiming for 98% quality; applied unchanged to 146 validation requests.\nShading is the feedback-order range, not a confidence interval. At 437 examples all orders use the same training set.",
        fontsize=9,
    )
    for suffix in ("png", "svg"):
        figure.savefig(directory / f"calibration.{suffix}", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    plot(json.loads((args.run / "results.json").read_text()), args.run)


if __name__ == "__main__":
    main()

"""Plot the frozen difficulty-policy experiment; no inference or model fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    result = json.loads((directory / "results.json").read_text())
    rows = result["test"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor("#fafbfe")
    for ax in axes:
        ax.set_facecolor("#fafbfe")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    sizes = [64, 128, 256, 512]
    for method, color, label in (
        ("knn", "#e28b31", "Nearest-neighbor bank"),
        ("ridge", "#5350c5", "Learned linear classifier"),
    ):
        values = [rows[f"{method}{n}"]["accuracy"] for n in sizes]
        axes[0].plot(sizes, values, color=color, marker="o", linewidth=2.5, label=label)
        axes[0].annotate(
            f"{100 * values[-1]:.1f}%",
            (512, values[-1]),
            xytext=(7, 0),
            textcoords="offset points",
            color=color,
            va="center",
            fontweight="bold",
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(sizes, [str(n) for n in sizes])
    axes[0].set_xlim(57, 690)
    axes[0].set_ylim(0.6, 0.96)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_xlabel("Labeled training examples")
    axes[0].set_ylabel("Held-out routing accuracy")
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    axes[0].set_title("More feedback, same embedding model", loc="left", fontsize=11)
    names = ["knn64", "knn512", "ridge64", "ridge512"]
    under = [rows[n]["false_routine"] for n in names]
    over = [rows[n]["false_advanced"] for n in names]
    axes[1].bar(range(4), under, color="#b45065", label="Advanced sent to routine")
    axes[1].bar(range(4), over, bottom=under, color="#d6a6af", label="Routine sent to advanced")
    for i, (a, b) in enumerate(zip(under, over, strict=True)):
        axes[1].text(i, a + b + 3, str(a + b), ha="center", fontweight="bold")
    axes[1].set_xticks(range(4), ["Bank\n64", "Bank\n512", "Linear\n64", "Linear\n512"])
    axes[1].set_ylabel("Errors among 512 test requests")
    axes[1].set_ylim(0, 195)
    axes[1].legend(loc="upper right", frameon=False, fontsize=8)
    axes[1].set_title("145 → 53 errors: 63% fewer", loc="left", fontsize=11)
    fig.suptitle("Learn a routing policy from feedback", x=0.08, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.08, 0.91, "Human-performance-derived math difficulty labels · fixed Qwen3-Embedding-4B", fontsize=10)
    fig.text(
        0.08,
        0.025,
        "512 reserved test problems. Gain: +18.0 percentage points (paired 95% interval: +14.1 to +22.1).",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.02, 0.06, 1, 0.88], w_pad=3)
    fig.savefig(directory / "learning-curve.png", dpi=180)
    fig.savefig(directory / "learning-curve.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    plot(parser.parse_args().run)


if __name__ == "__main__":
    main()

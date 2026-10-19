"""Plot the original comparison, or the verified LoRA training study with --lora-study."""

import argparse
import json
from pathlib import Path

from lora_report import summarize as summarize_lora
from lora_study import SEEDS


def plot_lora(plt, result, output):
    fig = plt.figure(figsize=(12.5, 6), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(5, 1))
    axes = [fig.add_subplot(grid[0, column]) for column in range(2)]
    footer = fig.add_subplot(grid[1, :])
    footer.set_axis_off()
    recipes = list(result["validation_auc"])
    seed_style = ((-0.12, "#0072b2", "o"), (0, "#d55e00", "s"), (0.12, "#009e73", "^"))
    handles, validation_values, accuracy_values = [], [], []
    for index, recipe in enumerate(recipes):
        for seed, (offset, color, marker) in zip(SEEDS, seed_style):
            value = result["validation_auc_by_seed"][recipe][str(seed)]
            validation_values.append(value)
            point = axes[0].scatter(
                index + offset, value, color=color, marker=marker, s=45, label=f"Seed {seed}", zorder=3
            )
            if index == 0:
                handles.append(point)
        mean = result["validation_auc"][recipe]
        line = axes[0].plot([index - 0.25, index + 0.25], [mean, mean], color="#333333", label="Mean", zorder=2)[0]
        if index == 0:
            mean_handle = line
    selected = recipes.index(result["recipe"])
    axes[0].axvspan(selected - 0.42, selected + 0.42, color="#009e73", alpha=0.08, zorder=0)
    names = {
        "original": "Original",
        "slow": "Slow",
        "regularized": "Regularized",
        "frozen-head": "Frozen\nhead",
        "scorer-only": "Scorer\nonly",
        "top-layers": "Top\nlayers",
    }
    axes[0].set(
        xticks=range(len(recipes)),
        xticklabels=[names[name] for name in recipes],
        ylabel="Validation ROC AUC",
        title=f"512 training examples · selected: {result['recipe']}",
        ylim=(max(0, min(validation_values) - 0.03), min(1, max(validation_values) + 0.03)),
    )
    stages = ("original_ce_512", "original_auc_512", "selected_auc_512", "selected_auc_2581")
    for index, stage in enumerate(stages):
        measured = result["stages"][stage]
        for seed, (offset, color, marker) in zip(SEEDS, seed_style):
            value = 100 * measured["seeds"][str(seed)]["accuracy"]
            accuracy_values.append(value)
            axes[1].scatter(index + offset, value, color=color, marker=marker, s=45, zorder=3)
        mean = 100 * measured["accuracy_mean"]
        axes[1].plot([index - 0.25, index + 0.25], [mean, mean], color="#333333", zorder=2)
    reference = 100 * result["ridge512_accuracy"]
    accuracy_values.append(reference)
    ridge_handle = axes[1].axhline(reference, color="#777777", linestyle="--", label="Ridge512 reference")
    axes[1].set(
        xticks=range(len(stages)),
        xticklabels=[
            "Historical\nCE · FP16\n512",
            "Original\nAUC · FP32\n512",
            "Selected\nAUC · FP32\n512",
            "Selected\nAUC · FP32\n2,581",
        ],
        ylabel="Regression accuracy (%)",
        title="Frozen stages · all three seeds",
        ylim=(max(0, min(accuracy_values) - 5), min(100, max(accuracy_values) + 5)),
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    footer.legend(handles=handles + [mean_handle, ridge_handle], loc="upper center", ncol=5, frameon=False)
    footer.text(
        0.5,
        0.15,
        "Historical CE → original AUC also changes FP16 → FP32; it is not a selection-only comparison.\n"
        "Exposed regression benchmark; seed spread is not a confidence interval.",
        ha="center",
        va="center",
        fontsize=9,
        transform=footer.transAxes,
    )
    fig.suptitle(f"LoRA training · {result['n']} exposed regression problems", fontsize=14)
    for extension in ("png", "svg"):
        fig.savefig(output / f"lora-study.{extension}", dpi=180)
    plt.close(fig)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("evidence"))
    parser.add_argument("--output", type=Path, default=Path("run-output/figures"))
    parser.add_argument("--lora-study", type=Path, help="Plot the LoRA training study using its verified records")
    args = parser.parse_args()
    followup = summarize_lora(args.lora_study, args.evidence) if args.lora_study else None
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    if followup is not None:
        plot_lora(plt, followup, args.output)
        return
    result = json.loads((args.evidence / "report.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for method, color in (("knn", "#5576a4"), ("ridge", "#008a78")):
        sizes = (64, 128, 256, 512)
        values = [100 * result["embedding"][f"{method}{n}"]["accuracy"] for n in sizes]
        axes[0].plot(sizes, values, marker="o", color=color, label=method)
    axes[0].set(
        xlabel="Training examples",
        ylabel="Test accuracy (%)",
        title="Frozen Qwen4B encoder",
        xticks=[64, 128, 256, 512],
    )
    axes[0].legend(frameon=False)
    for i, variant in enumerate(("base", "full", "lora")):
        values = [100 * result["native"][f"{variant}-{seed}"]["accuracy"] for seed in (89, 137, 211)]
        axes[1].scatter([i - 0.07, i, i + 0.07], values, s=50, color="#a75c32")
        axes[1].plot([i - 0.2, i + 0.2], [sum(values) / 3] * 2, color="#333333")
    axes[1].set(
        xticks=[0, 1, 2],
        xticklabels=["Base", "Full tuning", "LoRA + head"],
        ylabel="Test accuracy (%)",
        title="Native Laya · three seeds; line = mean",
    )
    for ax in axes:
        ax.set_ylim(40, 100)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"Contest-relative difficulty · {result['n']} chronological test problems", fontsize=14)
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"comparison.{extension}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharex=True, layout="constrained")
    for i, variant in enumerate(("full", "lora")):
        for j, seed in enumerate((89, 137, 211)):
            root = args.evidence / "native" / f"seed-{seed}"
            history = json.loads((root / f"{variant}-history.json").read_text())
            frozen = json.loads((root / "frozen.json").read_text())
            ax = axes[i, j]
            ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], label="Training loss")
            ax.plot([r["epoch"] for r in history], [r["cross_entropy"] for r in history], label="Validation CE")
            ax.axvline(frozen["selected"][variant]["epoch"], color="#444444", linestyle="--", label="Selected epoch")
            ax.set(title=f"{variant} · seed {seed}", xlabel="Epoch", ylabel="Loss")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Checkpoint selection stops before late-epoch overfitting", fontsize=14)
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"learning-curves.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

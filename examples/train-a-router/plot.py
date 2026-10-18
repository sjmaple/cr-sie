"""Regenerate the result and overfitting figures: uv run --group plots python plot.py."""

import argparse
import json
from pathlib import Path


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("evidence"))
    parser.add_argument("--output", type=Path, default=Path("run-output/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = json.loads((args.evidence / "report.json").read_text())
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
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

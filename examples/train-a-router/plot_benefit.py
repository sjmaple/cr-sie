"""Draw the frozen benefit comparison; no policy fitting or model calls.

Install the example's plots dependency group, then pass comparison.json from
benefit_summary.py. Output must be a new directory. Both figures use gross
answer-model savings, because native router hosting has not been priced.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from data import file_hash, write

METHODS = ("base", "lora", "embedding", "competition")
TARGETS = (0, 1, 2, 5)
LABELS = {
    "base": "Laya, unchanged",
    "lora": "Laya + LoRA · 511 examples",
    "embedding": "Qwen embeddings + ridge · 511 examples",
    "competition": "Competition name only · control",
}
COLORS = {"base": "#5274A0", "lora": "#D47B31", "embedding": "#008578", "competition": "#8A8F98"}
FOOTER = (
    "Gross answer-model savings exclude router costs. Embedding net charges are in the tables; native hosting is unpriced.\n"
    "Quality-loss whiskers: paired 95% problem-bootstrap intervals, conditional on fixed policies and recorded answers.\n"
    "Intervals exclude paper clustering, selection and solver randomness; zero-event intervals are not risk bounds.\n"
    "Calibration targets are not test guarantees."
)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate(report):
    """Refuse incomplete or incompatible summaries rather than invent intervals."""
    if (
        report.get("format") != "sie-benefit-comparison-v1"
        or report.get("primary_target_pp") != 1
        or not isinstance(report.get("n"), int)
        or isinstance(report["n"], bool)
        or report["n"] < 1
        or len(report.get("ids", [])) != report["n"]
        or len(set(report["ids"])) != report["n"]
        or set(report.get("policies", {})) != set(METHODS)
        or set(report.get("baselines", {})) != {"always_strong", "always_cheap"}
    ):
        raise ValueError("Expected the complete frozen benefit comparison and ordered test IDs")
    metrics = list(report["baselines"].values())
    for policies in report["policies"].values():
        if set(policies) != {str(target) for target in TARGETS}:
            raise ValueError("Retain all four predeclared calibration targets")
        metrics.extend(policy["metrics"] for policy in policies.values())
    strong = report["baselines"]["always_strong"]
    for metric in metrics:
        interval = metric.get("paired_bootstrap_95ci", {}).get("gap_vs_strong_pp")
        if (
            metric.get("n") != report["n"]
            or not all(
                _number(metric.get(name))
                for name in ("answer_accuracy", "gap_vs_strong_pp", "gross_solver_savings_pct")
            )
            or not 0 <= metric["answer_accuracy"] <= 1
            or not isinstance(interval, list)
            or len(interval) != 2
            or not all(_number(value) for value in interval)
            or not -100 <= interval[0] <= interval[1] <= 100
            or not math.isclose(
                metric["gap_vs_strong_pp"], 100 * (strong["answer_accuracy"] - metric["answer_accuracy"]), abs_tol=1e-9
            )
        ):
            raise ValueError("Need aligned finite gross savings, answer accuracy and paired quality-gap intervals")
    if strong["gap_vs_strong_pp"] != 0 or strong["gross_solver_savings_pct"] != 0:
        raise ValueError("Always-strong must remain the zero-loss, zero-savings anchor")
    return report


def _range(values, fraction=0.15, minimum=0.8):
    low, high = min(0, min(values)), max(0, max(values))
    padding = max(minimum, fraction * (high - low))
    return low - padding, high + padding


def _context(report):
    cheap = report["baselines"]["always_cheap"]
    strong = report["baselines"]["always_strong"]
    return (
        f"Always strong: {100 * strong['answer_accuracy']:.2f}% answer accuracy.  "
        f"Always-cheap context: {100 * cheap['answer_accuracy']:.2f}% accuracy, "
        f"{cheap['gap_vs_strong_pp']:+.2f} pp loss, {cheap['gross_solver_savings_pct']:.1f}% gross savings."
    )


def _title(report, title):
    return f"{title}{' · SYNTHETIC LAYOUT CHECK' if report.get('synthetic') is True else ''}"


def _save(plt, figure, output, stem):
    for extension in ("png", "svg"):
        figure.savefig(output / f"{stem}.{extension}", dpi=180, facecolor="white")
    plt.close(figure)


def primary(plt, report, output):
    figure = plt.figure(figsize=(14, 7.2), layout="constrained")
    grid = figure.add_gridspec(3, 2, height_ratios=(0.85, 4.7, 1.25), width_ratios=(1, 1.05))
    header = figure.add_subplot(grid[0, :])
    header.set_axis_off()
    header.text(0, 0.7, _title(report, "Answer quality and answer-model savings"), fontsize=17, weight="bold")
    header.text(
        0,
        0.22,
        f"Primary calibration target: at most 1 pp quality loss · {report['n']} fixed test problems",
        fontsize=11,
    )
    savings, gap = figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])
    names = ("strong", *METHODS)
    rows = [report["baselines"]["always_strong"]] + [report["policies"][name]["1"]["metrics"] for name in METHODS]
    labels = ["Always strong · baseline"] + [LABELS[name] for name in METHODS]
    saved = [row["gross_solver_savings_pct"] for row in rows]
    bounds = [value for row in rows for value in row["paired_bootstrap_95ci"]["gap_vs_strong_pp"]]
    savings.set_xlim(_range(saved, fraction=0.25, minimum=2))
    gap.set_xlim(_range(bounds + [row["gap_vs_strong_pp"] for row in rows] + [1], fraction=0.2))
    for index, (name, row) in enumerate(zip(names, rows, strict=True)):
        color = COLORS.get(name, "#303740")
        value = row["gross_solver_savings_pct"]
        savings.barh(index, value, color=color, height=0.52, alpha=0.85)
        savings.scatter(value, index, color=color, s=25, zorder=3)
        savings.annotate(
            f"{value:.1f}%",
            (value, index),
            xytext=(6 if value >= 0 else -6, 0),
            textcoords="offset points",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=10,
        )
        low, high = row["paired_bootstrap_95ci"]["gap_vs_strong_pp"]
        gap.hlines(index, low, high, color=color, linewidth=2)
        gap.vlines([low, high], index - 0.07, index + 0.07, color=color, linewidth=1)
        gap.scatter(
            row["gap_vs_strong_pp"], index, color=color, marker="D" if name == "competition" else "o", s=52, zorder=3
        )
        gap.text(
            1.015,
            index,
            f"{row['gap_vs_strong_pp']:+.2f} pp",
            transform=gap.get_yaxis_transform(),
            va="center",
            fontsize=10,
        )
    for axis in (savings, gap):
        axis.set_ylim(len(rows) - 0.45, -0.55)
        axis.set_yticks(range(len(rows)))
        axis.grid(axis="x", color="#E2E6EB", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.axvline(0, color="#67717E", linewidth=0.8)
        axis.tick_params(axis="y", length=0)
    savings.set_yticklabels(labels)
    savings.set(title="Gross savings vs always strong", xlabel="Answer-model cost saved (%)")
    gap.set_yticklabels([])
    gap.axvline(1, color="#A17D53", linewidth=0.9, linestyle="--", zorder=0)
    gap.set(
        title="Observed answer-quality loss · paired 95% interval",
        xlabel="Accuracy loss (percentage points) · negative is better",
    )
    footer = figure.add_subplot(grid[2, :])
    footer.set_axis_off()
    footer.text(0, 0.92, _context(report), fontsize=10, va="top")
    footer.text(
        0,
        0.59,
        FOOTER + "\nDashed line: 1 pp reference. The control uses competition metadata only.",
        fontsize=9,
        va="top",
        color="#46515E",
    )
    _save(plt, figure, output, "primary-target")


def alternatives(plt, report, output):
    figure = plt.figure(figsize=(13, 9.2), layout="constrained")
    grid = figure.add_gridspec(4, 2, height_ratios=(0.65, 3.15, 3.15, 1.35))
    header = figure.add_subplot(grid[0, :])
    header.set_axis_off()
    header.text(0, 0.7, _title(report, "Four predeclared cost / quality choices"), fontsize=17, weight="bold")
    header.text(
        0,
        0.12,
        "Point labels are calibration targets (0, 1, 2, 5 pp); filled points include the primary 1 pp target.",
        fontsize=10.5,
    )
    all_rows = [report["policies"][name][str(target)]["metrics"] for name in METHODS for target in TARGETS]
    xvalues = [value for row in all_rows for value in row["paired_bootstrap_95ci"]["gap_vs_strong_pp"]]
    xvalues += [row["gap_vs_strong_pp"] for row in all_rows]
    yvalues = [row["gross_solver_savings_pct"] for row in all_rows]
    xlim, ylim = _range(xvalues, fraction=0.24), _range(yvalues, fraction=0.2, minimum=3)
    for index, name in enumerate(METHODS):
        axis = figure.add_subplot(grid[1 + index // 2, index % 2])
        color = COLORS[name]
        # Identical policies can share coordinates. Retain every target label.
        grouped = {}
        for target in TARGETS:
            row = report["policies"][name][str(target)]["metrics"]
            point = (row["gap_vs_strong_pp"], row["gross_solver_savings_pct"])
            grouped.setdefault(point, []).append((target, row))
        for (loss, savings), entries in grouped.items():
            for _, row in entries:
                low, high = row["paired_bootstrap_95ci"]["gap_vs_strong_pp"]
                axis.hlines(savings, low, high, color=color, linewidth=1.4, alpha=0.75)
            targets = [target for target, _ in entries]
            primary_target = 1 in targets
            axis.scatter(
                loss,
                savings,
                s=70 if primary_target else 52,
                marker="D" if name == "competition" else "o",
                facecolor=color if primary_target else "white",
                edgecolor=color,
                linewidth=1.6,
                zorder=4,
            )
            label = "/".join(map(str, targets)) + " pp"
            axis.annotate(
                label,
                (loss, savings),
                xytext=(7, 8 if min(targets) != 0 else -16),
                textcoords="offset points",
                fontsize=9,
                color=color,
                weight="bold" if primary_target else "normal",
            )
        axis.scatter(0, 0, marker="+", s=100, linewidth=1.4, color="#303740", zorder=5)
        axis.axvline(0, color="#A9AFB8", linewidth=0.8)
        axis.axhline(0, color="#A9AFB8", linewidth=0.8)
        axis.set(
            xlim=xlim,
            ylim=ylim,
            title=LABELS[name],
            xlabel="Answer accuracy loss vs always strong (pp)",
            ylabel="Gross answer-model savings (%)",
        )
        axis.grid(color="#E2E6EB", linewidth=0.7)
        axis.set_axisbelow(True)
    footer = figure.add_subplot(grid[3, :])
    footer.set_axis_off()
    footer.text(
        0,
        0.96,
        "+ is always strong. Negative loss means higher answer accuracy. Each point is a frozen policy; coincident targets share a label.",
        fontsize=9.5,
        va="top",
    )
    footer.text(0, 0.75, _context(report), fontsize=9.5, va="top")
    footer.text(0, 0.51, FOOTER, fontsize=9, va="top", color="#46515E")
    _save(plt, figure, output, "calibration-targets")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True, help="Frozen comparison.json from benefit_summary.py")
    parser.add_argument("--output", type=Path, required=True, help="New figure directory")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = validate(json.loads(args.comparison.read_text()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    args.output.mkdir(parents=True, exist_ok=False)
    primary(plt, report, args.output)
    alternatives(plt, report, args.output)
    write(
        args.output / "manifest.json",
        {
            "format": "sie-benefit-figures-v1",
            "comparison_sha256": file_hash(args.comparison),
            "plot_source_sha256": file_hash(Path(__file__)),
            "synthetic": report.get("synthetic") is True,
            "cost_measure": "gross answer-model savings; excludes router hosting and embedding charges",
            "intervals": "copied paired 95% quality-gap intervals; no recomputation or policy selection",
            "files": {path.name: file_hash(path) for path in sorted(args.output.iterdir()) if path.is_file()},
        },
    )
    print(json.dumps({"output": str(args.output), "files": sorted(path.name for path in args.output.iterdir())}))


if __name__ == "__main__":
    main()

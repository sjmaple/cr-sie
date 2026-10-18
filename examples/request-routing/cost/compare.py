"""Compare kNN and Qwen reranker routing offline on the same 728 held-out requests.

Run fetch.py --reranker first. Recomputes calibration thresholds and held-out
point estimates from hash-verified evidence; makes no inference calls.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import reranker
import score
import study


def check_reranker(s: study.Study, pair: str, recording: reranker.Recording) -> dict:
    o = s.pairs[pair]
    tau = score.choose_tau(o, recording.scores, recording.usd, s.calibration)
    expected = recording.pairs[pair]
    expected_tau = math.inf if expected["tau"] is None else expected["tau"]
    row = score.summarize(o, *score.routed(o, recording.scores, recording.usd, tau), ~s.calibration)
    row["router_cost_per_1k"] = float(1000 * recording.usd[~s.calibration].mean())
    if not score.close(tau, expected_tau) or any(not score.close(v, expected[k]) for k, v in row.items()):
        raise study.EvidenceError(f"{pair}: reranker threshold or held-out metrics differ from the addendum")
    return row | {"tau": tau}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=study.EVIDENCE)
    args = parser.parse_args()
    try:
        s = study.load_study(args.evidence)
        recordings = {q: reranker.load(args.evidence / "reranker", s.ids, q) for q in reranker.ARMS}
        failures: list[str] = []
        for pair, names in study.PAIRS.items():
            baseline = score.check_pair(s, pair, failures)
            if failures:
                raise study.EvidenceError("; ".join(failures))
            rows = {
                "Always strong": baseline["rows"]["always_strong"],
                "bge-m3 kNN20": baseline["rows"]["router"] | {"tau": baseline["tau"]},
                **{f"Qwen reranker {q.upper()}": check_reranker(s, pair, r) for q, r in recordings.items()},
                "Always cheap": baseline["rows"]["always_cheap"],
            }
            print(f"\n{names.cheap_name} / {names.strong_name}: {baseline['held']} held-out requests")
            print(f"{'Router':<23} {'$/1,000':>8} {'Saving':>8} {'Quality':>8} {'Cheap':>8} {'Threshold':>10}")
            for label, row in rows.items():
                threshold = f"{row['tau']:.6f}" if "tau" in row else "-"
                print(
                    f"{label:<23} {score.money(row['cost_per_1k']):>8} "
                    f"{score.percent(row['saving_vs_strong']):>8} {score.percent(row['quality_vs_strong']):>8} "
                    f"{score.percent(row['cheap_share']):>8} {threshold:>10}"
                )
        print(
            "\nAll thresholds and point estimates match the pinned studies. No API calls.\n"
            "Quality is relative to always-strong; thresholds target 98% on calibration only.\n"
            "Q3 was written with knowledge of the earlier study's results; treat it as exploratory.\n"
            "Costs include routing: recorded billed credits for kNN; dated content-token prices for Qwen.\n"
            "No current price, latency, or superiority claim; confidence intervals are in the source results."
        )
    except (OSError, ValueError, KeyError) as error:
        print(f"Comparison failed: {error}. Run fetch.py --reranker first.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

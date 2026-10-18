"""Export disjoint prompt-training, validation and threshold-calibration sets.

Only the original 729 calibration requests are exported. The original 728
held-out requests stay in the verified evidence for the final evaluation.
No API calls, optimizer calls or GEPA dependency. See README.md for the loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

import study


def partitions(s: study.Study) -> dict[str, list[int]]:
    """Deterministic 60/20/20 split within each stratum, independent of outcome labels."""
    result: dict[str, list[int]] = {"train": [], "validation": [], "threshold": []}
    for stratum in sorted(set(s.strata)):
        indices = np.where(s.calibration & (s.strata == stratum))[0].tolist()
        indices.sort(key=lambda i: hashlib.sha256(f"odsc-routing-v1:{s.ids[i]}".encode()).hexdigest())
        first, second = 3 * len(indices) // 5, 4 * len(indices) // 5
        for name, selected in zip(result, (indices[:first], indices[first:second], indices[second:]), strict=True):
            result[name].extend(selected)
    return result


def export(s: study.Study, pair: str, destination: Path) -> dict[str, int]:
    if destination.exists():
        raise FileExistsError("output already exists; choose another --output directory")
    groups = partitions(s)
    o = s.pairs[pair]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gepa-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "export"
        staging.mkdir()
        for name, indices in groups.items():
            with (staging / f"{name}.jsonl").open("w") as handle:
                for i in indices:
                    row = {
                        "id": s.ids[i],
                        "stratum": str(s.strata[i]),
                        "request": s.prompts[i],
                        "q_cheap": float(o.q_cheap[i]),
                        "q_strong": float(o.q_strong[i]),
                        "usd_cheap": float(o.usd_cheap[i]),
                        "usd_strong": float(o.usd_strong[i]),
                    }
                    handle.write(json.dumps(row) + "\n")
        metadata = {
            "pair": pair,
            "study_revision": study.REVISION,
            "split_seed": "odsc-routing-v1",
            "counts": {name: len(indices) for name, indices in groups.items()},
            "heldout_exported": False,
            "note": "Only request goes to the router; quality and cost are evaluator feedback, never prompt input.",
        }
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        staging.rename(destination)
    return metadata["counts"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=study.EVIDENCE)
    parser.add_argument("--pair", choices=sorted(study.PAIRS), default="openai")
    parser.add_argument("--output", type=Path, default=study.HERE / "run-output/gepa")
    args = parser.parse_args()
    try:
        counts = export(study.load_study(args.evidence), args.pair, args.output)
    except (OSError, ValueError) as error:
        print(f"Export failed: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {counts} to {args.output}; held-out test requests were not exported. No API calls.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

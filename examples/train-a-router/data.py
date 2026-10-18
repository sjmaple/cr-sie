"""Pinned public inputs, duplicate-family isolation and chronological splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SOURCE_REVISION = "55bc0d2fb10954151e669d2026b87fa896f2fa26"
SOURCE_URL = (
    "https://huggingface.co/datasets/superlinked/sie-task-evidence/resolve/"
    "002087323da76e925235e0c6acb12f402bf8fd9e/evaluations/2026-10-17/"
    "request-routing-difficulty-policy-v1/source/"
)
SOURCE_HASHES = {
    "train.parquet": "1686c7df11b60848f05142e5b2215a96a9ab6952b3867827542f2c4d92621baa",
    "eval.parquet": "2a04f50a80a16bfc5c1ff5667f9403e238184bbe7c325eff99f684d89f8f97fe",
}
LABELS = ("routine", "advanced")
PROTOCOL = {
    "name": "contest-relative-grouped-v2",
    "target": "Above versus below the training-population median human difficulty for the same contest family",
    "input": "Contest family plus problem text; no year, problem index, ratings, answer or solution",
    "training_years": "through 2017",
    "validation_years": [2018],
    "calibration_years": [2019],
    "evaluation_years": "2020 onward, excluding every previously used duplicate family",
    "duplicates": "Connected components of canonical text or five-token Jaccard>=0.70; masked-number Jaccard>=0.85 with >=12 tokens. Drop components crossing chronological partitions.",
    "label_thresholds": "Median rating per contest tag from eligible training population only; fixed across every learning budget",
    "training_order": "Round-robin tag and label strata, each hash ordered; nested 64/128/256/512 prefixes",
    "embedding_candidates": {
        "methods": ["knn", "ridge"],
        "sizes": [64, 128, 256, 512],
        "penalties": [0.01, 0.1, 1.0, 10.0],
        "neighbors": 5,
    },
    "selection": "Ridge penalty on validation; all cutoffs on separate calibration; no evaluation-driven choices",
    "native_training": {
        "seeds": [89, 137, 211],
        "epochs": 8,
        "selection": "minimum validation cross entropy; calibration fits cutoff after checkpoint selection",
    },
    "reporting": "All declared families/seeds, paired case intervals and per-seed spread; no automatic winner or cost claim",
    "primary": "ridge512 versus knn64 on frozen temporal evaluation; report ridge64 and knn512 controls",
}
QUESTION = {
    "difficulty": {
        "type": "choice",
        "instructions": "Classify the problem relative to other problems in the named competition. Routine is the easier half and advanced the harder half of that competition's historical human-rated difficulty. Judge reasoning and insight needed; do not solve it.",
        "criteria": {
            "routine": "Relatively routine for this competition: familiar techniques and fewer demanding reasoning steps.",
            "advanced": "Relatively advanced for this competition: more demanding insight or multi-step reasoning.",
        },
    }
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def download_source(output):
    output.mkdir(parents=True, exist_ok=True)
    for name, expected in SOURCE_HASHES.items():
        path = output / name
        if not path.exists():
            with urllib.request.urlopen(SOURCE_URL + name, timeout=60) as response:
                body = response.read(10_000_001)
            if len(body) > 10_000_000 or hashlib.sha256(body).hexdigest() != expected:
                raise ValueError("Source download failed its size/hash check")
            with path.open("xb") as handle:
                handle.write(body)
        if file_hash(path) != expected:
            raise ValueError(f"Pinned source differs: {name}")


def canonical(text):
    text = text.casefold().replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return "".join(re.findall(r"[a-z0-9]+", text))


def fingerprint(text, masked=False):
    tokens = re.findall(r"[a-z]+|\d+", text.casefold())
    if masked:
        if len(tokens) < 12:
            return set()
        tokens = ["#" if token.isdigit() else token for token in tokens]
    return {tuple(tokens[i : i + 5]) for i in range(len(tokens) - 4)}


def duplicate_components(rows):
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        parent[max(a, b)] = min(a, b)

    seen = {}
    for i, row in enumerate(rows):
        key = canonical(row["problem"])
        if key in seen:
            union(i, seen[key])
        seen[key] = i
    # An inverted shingle index avoids comparing all pairs without a common token run.
    for masked, threshold in ((False, 0.70), (True, 0.85)):
        prints = [fingerprint(row["problem"], masked) for row in rows]
        index = defaultdict(list)
        for i, current in enumerate(prints):
            candidates = Counter(j for token in current for j in index[token])
            for j, common in candidates.items():
                if common / (len(current) + len(prints[j]) - common) >= threshold:
                    union(i, j)
            for token in current:
                index[token].append(i)
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    return list(groups.values())


def partition(year):
    return "train" if year <= 2017 else "validation" if year == 2018 else "calibration" if year == 2019 else "test"


def render_request(tag, text):
    return f"Competition: {tag}\nProblem: {text.strip()}"


def interleave(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["contest"], row["label"])].append(row)
    for group in groups.values():
        group.sort(key=lambda r: digest(["training-order-v2", r["id"]]))
    result = []
    for i in range(max(map(len, groups.values()))):
        result.extend(groups[key][i] for key in sorted(groups) if i < len(groups[key]))
    return result


def prepare(source, output, previously_used):
    import pyarrow.parquet as pq

    if output.exists():
        raise FileExistsError(output)
    rows = []
    for name, expected in SOURCE_HASHES.items():
        if file_hash(source / name) != expected:
            raise ValueError("Source hash differs")
        split = name.split(".")[0]
        rows.extend(dict(r, id=f"e2h-amc-{split}-{i}") for i, r in enumerate(pq.read_table(source / name).to_pylist()))
    used = set(json.loads(previously_used.read_text()))
    known_ids = {r["id"] for r in rows}
    if not used or not used <= known_ids:
        raise ValueError("Previously used IDs must be a nonempty subset of the pinned source")
    components = duplicate_components(rows)
    selected, removed, families = [], Counter(), {}
    for component in components:
        members = [rows[i] for i in component]
        ids = sorted(r["id"] for r in members)
        family = digest(ids)
        splits = {partition(r["year"]) for r in members}
        if len(splits) != 1:
            removed["cross_partition_family_rows"] += len(members)
            continue
        split = next(iter(splits))
        if split == "test" and any(identity in used for identity in ids):
            removed["previously_used_test_family_rows"] += len(members)
            continue
        # Retain one representative, chosen without ratings or model outputs.
        row = min(members, key=lambda r: digest(["representative-v2", r["id"]]))
        selected.append(dict(row, split=split, family=family))
        families[family] = ids
        removed["within_family_duplicates"] += len(members) - 1
    training_pool = [r for r in selected if r["split"] == "train"]
    medians = {
        tag: float(np.median([r["rating"] for r in training_pool if r["tag"] == tag]))
        for tag in sorted({r["tag"] for r in training_pool})
    }
    splits = {s: [] for s in ("train", "validation", "calibration", "test")}
    for row in selected:
        tag = row["tag"]
        splits[row["split"]].append(
            {
                "id": row["id"],
                "request": render_request(tag, row["problem"]),
                "label": int(row["rating"] >= medians[tag]),
                "contest": tag,
                "year": row["year"],
                "family": row["family"],
                "paper_group": f"{row['contest']}:{row['year']}",
            }
        )
    splits["train"] = interleave(splits["train"])[:512]
    for split in ("validation", "calibration", "test"):
        splits[split].sort(key=lambda r: digest(["input-order-v2", r["id"]]))
    if len(splits["train"]) != 512 or any(len(r) < 50 for r in splits.values()):
        raise ValueError("Insufficient examples for the declared protocol")
    dataset = {"protocol": PROTOCOL, "source_revision": SOURCE_REVISION, "label_thresholds": medians, "splits": splits}
    audit(dataset)
    output.mkdir(parents=True)
    write(output / "data.json", dataset)
    write(output / "questions.json", QUESTION)
    write(output / "protocol.json", PROTOCOL)
    write(output / "prior-used-ids.json", sorted(used))
    write(
        output / "split-audit.json",
        {
            "data_sha256": digest(dataset),
            "protocol_sha256": digest(PROTOCOL),
            "source_hashes": SOURCE_HASHES,
            "previously_used_ids_sha256": file_hash(previously_used),
            "counts": {
                s: {
                    "n": len(r),
                    "labels": dict(Counter(x["label"] for x in r)),
                    "contests": dict(Counter(x["contest"] for x in r)),
                }
                for s, r in splits.items()
            },
            "removed": dict(removed),
            "duplicate_families": {k: v for k, v in families.items() if len(v) > 1},
            "label_threshold_fit_rows": len(training_pool),
            "label_budget_note": "64/128/256/512 counts classifier feedback; the target definition uses the eligible pre-2018 source population",
        },
    )
    for split, values in splits.items():
        write(output / f"{split}.json", values)
        with (output / f"{split}.jsonl").open("x") as handle:
            for r in values:
                handle.write(
                    json.dumps(
                        {"state": r["request"], "questions": QUESTION, "expected": {"difficulty": LABELS[r["label"]]}}
                    )
                    + "\n"
                )
    print(
        json.dumps(
            {
                "output": str(output),
                "counts": {s: len(r) for s, r in splits.items()},
                "removed": dict(removed),
                "data_sha256": digest(dataset),
            },
            indent=2,
        )
    )


def audit(dataset):
    seen_ids, seen_text, seen_families, seen_groups = set(), set(), set(), set()
    if set(dataset["splits"]) != {"train", "validation", "calibration", "test"}:
        raise ValueError("Expected four disjoint splits")
    for split, rows in dataset["splits"].items():
        ids = {r["id"] for r in rows}
        texts = {canonical(r["request"]) for r in rows}
        families = {r["family"] for r in rows}
        groups = {r["paper_group"] for r in rows}
        if len(ids) != len(rows) or len(texts) != len(rows) or len(families) != len(rows):
            raise ValueError("Duplicate row/family within a split")
        if seen_ids & ids or seen_text & texts or seen_families & families or seen_groups & groups:
            raise ValueError("Cross-split identity, text, family or paper leakage")
        if any(partition(r["year"]) != split for r in rows) or {r["label"] for r in rows} != {0, 1}:
            raise ValueError("Invalid year or label distribution")
        seen_ids |= ids
        seen_text |= texts
        seen_families |= families
        seen_groups |= groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("evidence/source"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previously-used", type=Path, required=True)
    args = parser.parse_args()
    download_source(args.source)
    prepare(args.source, args.output, args.previously_used)


if __name__ == "__main__":
    main()

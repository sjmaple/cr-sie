"""Apply frozen routers to the separate arithmetic inputs without grading them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import benefit_outcomes as embedding
import benefit_outcomes_native as native
import solver
from data import digest, file_hash, write
from serving import vectors


def read(path):
    return json.loads(Path(path).read_text())


def inputs(study):
    study = Path(study)
    plan, rows = read(study / "source-plan.json"), read(study / "inputs.json")
    if (
        plan.get("format") != "sie-arithmetic-transfer-source-v1"
        or len(rows) != plan["n"]
        or len({row["id"] for row in rows}) != len(rows)
        or any(set(row) != {"id", "request"} for row in rows)
        or digest(rows) != plan["inputs_sha256"]
        or file_hash(study / "inputs.json") != plan["files_sha256"]["inputs.json"]
        or digest(solver.SETTINGS) != plan["settings_sha256"]
    ):
        raise ValueError("Arithmetic transfer inputs or solver settings changed")
    return plan, rows


def _record(study, rows, run, design, models):
    return {
        "format": "sie-arithmetic-router-scores-v1",
        "design": design,
        "n": len(rows),
        "ids": [row["id"] for row in rows],
        "inputs_sha256": file_hash(Path(study) / "inputs.json"),
        "source_plan_sha256": file_hash(Path(study) / "source-plan.json"),
        "frozen_sha256": file_hash(Path(run) / "frozen.json"),
        "models": models,
        "label_access": "Only request inputs and existing frozen router state were read. No arithmetic references or solver answers were opened.",
        "scoring_source_sha256": file_hash(Path(__file__)),
    }


def score_embedding(run, study, output, *, matched_control=False):
    run, study, output = Path(run), Path(study), Path(output)
    if output.exists():
        raise FileExistsError(output)
    _, rows = inputs(study)
    if read(run / "frozen.json").get("format") == embedding.FROZEN_FORMAT:
        if matched_control:
            raise ValueError("A matched control belongs to the teacher study")
        frozen, selected = embedding.checked_frozen(run)
        design = "embedding"
    else:
        import teacher_outcomes

        frozen, selected = teacher_outcomes.checked_frozen(run)
        design = "embedding_teacher_control" if matched_control else "embedding_teacher"
    head = selected["matched_benchmark_control"] if matched_control else selected["head"]
    selections = frozen["matched_benchmark_control"]["selections"] if matched_control else frozen["selections"]
    matrix = vectors(rows, study / "embeddings")
    probabilities = np.column_stack([embedding.probability(matrix, head["models"][arm]) for arm in embedding.ARMS])
    scores, expected = embedding.normalized_scores(probabilities, head["conditional_cost_statistics"])
    result = _record(
        study,
        rows,
        run,
        design,
        {
            design: {
                "probabilities": probabilities.tolist(),
                "scores": scores.tolist(),
                "expected_costs_proxy": expected.tolist(),
                "policies": selections,
            }
        },
    )
    result["embedding_files_sha256"] = {
        name: file_hash(study / "embeddings" / name) for name in ("plan.json", "vectors.jsonl.gz", "collection.json")
    }
    write(output, result)
    return result


def score_native(run, head_run, study, output, device="cuda"):
    run, head_run, study, output = map(Path, (run, head_run, study, output))
    if output.exists():
        raise FileExistsError(output)
    _, rows = inputs(study)
    _, trained, frozen = native.checked_frozen(run, head_run, weights=True)
    models = {}
    for variant in ("base", "lora"):
        margins = native.checkpoint_margins(run / variant, rows, device)
        probabilities = native.probabilities(margins)
        scores, expected = embedding.normalized_scores(probabilities, trained["conditional_cost_statistics"])
        models[variant] = {
            "margins": np.asarray(margins).tolist(),
            "probabilities": probabilities.tolist(),
            "scores": scores.tolist(),
            "expected_costs_proxy": expected.tolist(),
            "policies": frozen["models"][variant]["policies"],
        }
        print(f"Scored {len(rows)} arithmetic inputs with frozen {variant}", flush=True)
    result = _record(study, rows, run, "native", models)
    result["head_frozen_sha256"] = file_hash(head_run / "frozen.json")
    result["checkpoint_manifest_sha256"] = digest(trained["checkpoints"])
    result["device"] = device
    write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("embedding", "native"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-run", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument(
        "--matched-control", action="store_true", help="Score the teacher study's matched benchmark head"
    )
    args = parser.parse_args()
    if args.method == "native":
        if args.head_run is None or args.matched_control:
            parser.error("Native scoring requires --head-run")
        result = score_native(args.run, args.head_run, args.study, args.output, args.device)
    else:
        result = score_embedding(args.run, args.study, args.output, matched_control=args.matched_control)
    print(json.dumps({"n": result["n"], "design": result["design"], "models": list(result["models"])}))


if __name__ == "__main__":
    main()

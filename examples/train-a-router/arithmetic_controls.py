"""Separate cost-estimate controls for unchanged Laya weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import arithmetic_study as study
from data import digest, file_hash, write

FORMAT = "sie-arithmetic-cost-controls-frozen-v1"
PROTOCOL = {
    "purpose": "Isolate weight adaptation from changing expected solver costs.",
    "weights": "Reuse the unchanged reference Laya probabilities; no training or new model calls.",
    "costs": "Use the same arithmetic500/2000 conditional cost means as each corresponding LoRA model.",
    "selection": "Apply the unchanged calibration cutoff rule at targets1/0/2/5.",
    "scope": "Separate pre-test controls; all original core models, predictions and cutoffs remain immutable.",
}


def names():
    return {f"base_arithmetic_{size}" for size in (500, 2000)}


def _bindings(run):
    run = Path(run)
    return {
        "core_frozen_sha256": file_hash(run / "frozen.json"),
        "study_plan_sha256": file_hash(run / "plan.json"),
        "calibration_grades_sha256": file_hash(run / "grades-calibration.json"),
        "native_scores_sha256": file_hash(run / "native-scores.json"),
        "native_trained_sha256": {str(size): file_hash(run / f"native-{size}/trained.json") for size in (500, 2000)},
        "source_sha256": file_hash(Path(__file__)),
    }


def scores(probabilities, statistics, count):
    values = np.asarray(probabilities, dtype=float)
    study.require(
        values.shape == (count, 2) and np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(),
        "Need finite, unchanged cheap/strong correctness probabilities",
    )
    return study.outcomes.normalized_scores(values, statistics)[0].tolist()


def freeze(source, run, controls):
    run, controls = Path(run), Path(controls)
    if controls.exists():
        raise FileExistsError(controls)
    study.unopened(run)
    _, inputs, _ = study.checked_plan(source, run)
    core = study.read(run / "frozen.json")
    bindings = _bindings(run)
    study.require(
        core["study_plan_sha256"] == bindings["study_plan_sha256"]
        and core["calibration_grades_sha256"] == bindings["calibration_grades_sha256"]
        and core["model_artifacts_sha256"]["native-scores.json"] == bindings["native_scores_sha256"]
        and core["source_code_sha256"] == study.code_hashes(),
        "The existing core freeze or executed sources changed",
    )
    native = study.read(run / "native-scores.json")["models"]["reference_base"]
    calibration = study.read(run / "grades-calibration.json")["rows"]
    ids = {split: [row["id"] for row in inputs[split]] for split in ("calibration", "test")}
    study.require(
        [row["id"] for row in calibration] == ids["calibration"]
        and all(native["splits"][split]["ids"] == ordered == core["ids"][split] for split, ordered in ids.items()),
        "Cached probabilities or calibration rows differ from the frozen question order",
    )
    methods = {}
    for size in (500, 2000):
        trained = study.checked_native(source, run, size)
        statistics = trained["conditional_cost_statistics"]
        values = {
            split: scores(native["splits"][split]["probabilities"], statistics, len(ordered))
            for split, ordered in ids.items()
        }
        methods[f"base_arithmetic_{size}"] = {
            "design": "native",
            "training_rows": 0,
            "cost_training_rows": size,
            "weights_changed": False,
            "cost_model_source": f"arithmetic {size} training outcomes",
            "conditional_cost_statistics": statistics,
            "scores": values,
            "policies": {
                str(target): study.solver_report.select_cutoff(
                    calibration, values["calibration"], target, router_credits=None
                )
                for target in study.TARGETS
            },
        }
    result = {
        "format": FORMAT,
        "protocol": PROTOCOL,
        **bindings,
        "ids": ids,
        "base_trained_sha256": native["trained_sha256"],
        "base_probabilities_sha256": {split: digest(native["splits"][split]["probabilities"]) for split in ids},
        "methods": methods,
        "fresh_test_answers_opened": False,
    }
    write(controls / "frozen.json", result)
    return result


def checked_frozen(source, run, controls):
    run, controls = Path(run), Path(controls)
    _, inputs, _ = study.checked_plan(source, run)
    result = study.read(controls / "frozen.json")
    study.require(
        result["format"] == FORMAT
        and result["protocol"] == PROTOCOL
        and all(result[key] == value for key, value in _bindings(run).items())
        and set(result["methods"]) == names(),
        "Cost-control protocol, source or original study bindings changed",
    )
    native = study.read(run / "native-scores.json")["models"]["reference_base"]
    calibration = study.read(run / "grades-calibration.json")["rows"]
    study.require(result["base_trained_sha256"] == native["trained_sha256"], "Base checkpoint binding changed")
    for size in (500, 2000):
        item = result["methods"][f"base_arithmetic_{size}"]
        statistics = study.checked_native(source, run, size)["conditional_cost_statistics"]
        study.require(
            item["training_rows"] == 0
            and item["cost_training_rows"] == size
            and item["weights_changed"] is False
            and item["conditional_cost_statistics"] == statistics
            and set(item["policies"]) == {str(target) for target in study.TARGETS},
            "Unchanged-weight control or matched cost means changed",
        )
        for split in ("calibration", "test"):
            ordered = [row["id"] for row in inputs[split]]
            probabilities = native["splits"][split]["probabilities"]
            expected = scores(probabilities, statistics, len(ordered))
            actual = np.asarray(item["scores"][split], dtype=float)
            study.require(
                result["ids"][split] == ordered == native["splits"][split]["ids"]
                and digest(probabilities) == result["base_probabilities_sha256"][split]
                and actual.shape == (len(ordered),)
                and np.isfinite(actual).all()
                and np.allclose(actual, expected, rtol=1e-12, atol=1e-15),
                "Cost-control input probabilities or score formula changed",
            )
        for target in study.TARGETS:
            expected = study.solver_report.select_cutoff(
                calibration, item["scores"]["calibration"], target, router_credits=None
            )
            actual = item["policies"][str(target)]
            study.require(
                actual["mode"] == expected["mode"] and actual["cutoff"] == expected["cutoff"],
                "Cost-control cutoff differs from the declared calibration selection",
            )
    return result


def authorize_test(source, run, controls):
    run, controls = Path(run), Path(controls)
    for path in (run / "grades-test.json", controls / "test-access.json", controls / "report.json"):
        if path.exists():
            raise FileExistsError(path)
    frozen = checked_frozen(source, run, controls)
    access = study.read(run / "test-access.json")
    study.require(
        access["frozen_sha256"] == frozen["core_frozen_sha256"],
        "Core access must bind the unchanged pre-test model freeze",
    )
    result = {
        "format": "sie-arithmetic-cost-controls-test-access-v1",
        "frozen_sha256": file_hash(controls / "frozen.json"),
        "core_test_access_sha256": file_hash(run / "test-access.json"),
        "core_frozen_sha256": frozen["core_frozen_sha256"],
        "statement": "Both cost-only controls were frozen before reading fresh test answers.",
    }
    write(controls / "test-access.json", result)
    return result


def report(source, run, controls):
    run, controls = Path(run), Path(controls)
    if (controls / "report.json").exists():
        raise FileExistsError(controls / "report.json")
    frozen = checked_frozen(source, run, controls)
    access = study.read(controls / "test-access.json")
    grades = study.read(run / "grades-test.json")
    study.require(
        access["frozen_sha256"] == file_hash(controls / "frozen.json")
        and access["core_test_access_sha256"] == file_hash(run / "test-access.json")
        and grades["test_access_sha256"] == access["core_test_access_sha256"]
        and [row["id"] for row in grades["rows"]] == frozen["ids"]["test"],
        "Cost-control access or shared held-out grades changed",
    )
    methods = {}
    for name, item in frozen["methods"].items():
        policies = {}
        values = np.asarray(item["scores"]["test"])
        for target, policy in item["policies"].items():
            decisions = (values >= policy["cutoff"]).tolist() if policy["mode"] == "threshold" else policy["mode"]
            policies[target] = {
                "mode": policy["mode"],
                "cutoff": policy["cutoff"],
                "metrics": study.solver_report.summarize(
                    grades["rows"], decisions, router_credits=None if policy["mode"] == "threshold" else 0
                ),
            }
        methods[name] = {
            "label": f"Unchanged Laya + arithmetic {item['cost_training_rows']} cost estimates",
            **{key: item[key] for key in ("design", "training_rows", "cost_training_rows", "weights_changed")},
            "cost_model_source": item["cost_model_source"],
            "show_by_default": False,
            "policies": policies,
        }
    result = {
        "format": "sie-arithmetic-cost-controls-report-v1",
        "n": len(grades["rows"]),
        "ids": frozen["ids"]["test"],
        "primary_target_pp": 1,
        "frozen_sha256": file_hash(controls / "frozen.json"),
        "core_frozen_sha256": frozen["core_frozen_sha256"],
        "test_grades_sha256": file_hash(run / "grades-test.json"),
        "methods": methods,
        "cost_scope": "Gross solver-token savings; native hosting is unpriced. Each cost estimate matches its LoRA training population.",
    }
    write(controls / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "authorize-test", "report"))
    for name in ("source", "run", "controls"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = {"freeze": freeze, "authorize-test": authorize_test, "report": report}[args.command](
        args.source, args.run, args.controls
    )
    print(result["format"])


if __name__ == "__main__":
    main()

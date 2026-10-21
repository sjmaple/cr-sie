"""Keep published-data reproduction separate from the original blind experiment."""

from pathlib import Path
import shutil

import paired_calibration
import paired_report
import paired_training as core


FORMAT = "train-a-router-reproduction-v1"
MODE = "post_publication_reproduction"
INTERNAL_ROUTES = "internal/frozen-final-routes.json"


def reuse_controls(data, run):
    """Explicitly reuse fixed controls while fitting only the embedding classifier."""
    data, run = Path(data), Path(run)
    reference = data / "reference"
    core.checked_plan(run)
    _, rows = core.checked_plan(reference)
    core.require(core.sha(run / "plan.json") == core.sha(reference / "plan.json"), "Reference controls require the identical study plan")
    bundle = core.read(reference / "frozen/frozen-final-routes.json")
    paired_calibration.checked_frozen_bundle(bundle, rows)
    names = ("topic-model.json", "topic-scores.json", "native-scores.json", "native/trained.json")
    core.require(not (run / "comparison-reuse.json").exists() and all(not (run / name).exists() for name in names),
                 "Comparison artifacts already exist; preserve them or use a new run")
    for method in ("native", "topic"):
        scores = core.read(reference / f"{method}-scores.json")
        core.require(core.sha(reference / f"{method}-scores.json") == bundle["scores_sha256"][method], "Reference scores changed")
        model_name = "native/trained.json" if method == "native" else "topic-model.json"
        key = "trained_sha256" if method == "native" else "model_sha256"
        core.require(core.sha(reference / model_name) == scores[key], "Reference comparison model changed")
    for name in names:
        target = run / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / name, target)
    result = {
        "format": "train-a-router-reused-comparisons-v1",
        "plan_sha256": core.sha(run / "plan.json"),
        "copied_sha256": {name: core.sha(run / name) for name in names},
        "fitting_performed": False,
        "weights_copied": False,
        "interpretation": "Only the embedding classifier is refitted in this path; Laya, adapted Laya and topic comparisons are the published reference controls.",
    }
    core.write(run / "comparison-reuse.json", result)
    return result


def checked_routes(path, source_rows):
    """Load either original routes or an explicitly retrospective public envelope."""
    path = Path(path)
    envelope = core.read(path)
    if envelope["format"] != FORMAT:
        paired_calibration.checked_frozen_bundle(envelope, source_rows)
        return envelope, None
    core.require(
        envelope["evaluation_mode"] == MODE
        and envelope["frozen_before_final_model_outcomes"] is False
        and envelope["final_outcomes_accessed_during_calibration"] is False
        and envelope["generated_routes_path"] == INTERNAL_ROUTES,
        "Public reproduction must remain explicitly retrospective",
    )
    generated = path.parent / INTERNAL_ROUTES
    core.require(core.sha(generated) == envelope["generated_routes_sha256"], "Refitted route artifact changed")
    bundle = core.read(generated)
    # The unchanged scientific function records its local input-only boundary.
    # Only the outer envelope describes this published-data run's timing claim.
    paired_calibration.checked_frozen_bundle(bundle, source_rows)
    core.require(bundle["plan_sha256"] == envelope["plan_sha256"], "Reproduction plan binding changed")
    return bundle, envelope


def calibrate(run, data, output):
    run, data, output = Path(run), Path(data), Path(output)
    core.require(not output.exists(), "Preserve previous calibration")
    _, rows = core.checked_plan(run)
    original = data / "reference/frozen/frozen-final-routes.json"
    reference = core.read(original)
    paired_calibration.checked_frozen_bundle(reference, rows)
    result = paired_calibration.freeze(
        run,
        data / "recordings/grades-calibration.json",
        run / "embedding-scores.json",
        run / "native-scores.json",
        run / "topic-scores.json",
        data / "fees/projection.json",
        output / "internal",
    )
    envelope = {
        "format": FORMAT,
        "evaluation_mode": MODE,
        "frozen_before_final_model_outcomes": False,
        "final_outcomes_accessed_during_calibration": False,
        "generated_routes_path": INTERNAL_ROUTES,
        "generated_routes_sha256": core.sha(output / INTERNAL_ROUTES),
        "original_frozen_routing_sha256": core.sha(original),
        "plan_sha256": result["plan_sha256"],
        "interpretation": "The test answers are already published. This repeats the recipe; it is not a new blind evaluation.",
    }
    core.write(output / "frozen-final-routes.json", envelope)
    return envelope


def evaluate(data, run, output):
    """Validate original acquisition before applying refitted routes to those answers."""
    data, run, output = Path(data), Path(run), Path(output)
    core.require(not output.exists(), "Preserve previous analysis")
    plan, rows = core.checked_plan(run)
    routing = run / "frozen/frozen-final-routes.json"
    bundle, envelope = checked_routes(routing, rows)
    core.require(envelope is not None, "Use replay for the original recorded experiment")
    core.require(bundle["plan_sha256"] == core.sha(run / "plan.json"), "Routes belong to a different run")

    original_path = data / "reference/frozen/frozen-final-routes.json"
    final_path = data / "recordings/grades-final.json"
    original, grades = core.read(original_path), core.read(final_path)
    paired_report.check_lineage(original, grades, core.sha(original_path))
    core.require(
        envelope["original_frozen_routing_sha256"] == core.sha(original_path),
        "Original acquisition reference changed",
    )
    for key in ("source_plan_sha256", "pair_spec_sha256", "acquisition_protocol_sha256", "grader_sha256", "cost_scope"):
        core.require(bundle[key] == original[key], "Refit and original experiment use different " + key)
    core.require(bundle["source_rows"] == original["source_rows"], "Refit final population changed")
    core.require(bundle["pair_spec_sha256"] == plan["pair_spec_sha256"], "Refit model pair changed")
    for key, relative in (
        ("training_grades_sha256", "recordings/grades-train.json"),
        ("calibration_grades_sha256", "recordings/grades-calibration.json"),
        ("fee_projection_sha256", "fees/projection.json"),
    ):
        core.require(bundle[key] == core.sha(data / relative), "Refit input changed: " + relative)
    for method, digest in bundle["scores_sha256"].items():
        core.require(digest == core.sha(run / f"{method}-scores.json"), "Refit input scores changed")

    # Evaluate the generated routes; never alter the original grades or their
    # frozen-routing binding. The public result explicitly removes a blind-test claim.
    result = paired_report.evaluate(grades["rows"], bundle)
    result.update(
        format="train-a-router-reproduction-report-v1",
        evaluation_mode=MODE,
        new_blind_evaluation=False,
        test_outcomes_already_published=True,
    )
    result["provenance"] = {
        "adapter_sha256": core.sha(__file__),
        "reporter_sha256": core.sha(paired_report.__file__),
        "reproduction_routing_sha256": core.sha(routing),
        "generated_routes_sha256": envelope["generated_routes_sha256"],
        "original_frozen_routing_sha256": core.sha(original_path),
        "final_grades_sha256": core.sha(final_path),
        "cost_scope": grades["cost_scope"],
    }
    if (run / "comparison-reuse.json").exists():
        result["reused_reference_comparisons"] = core.read(run / "comparison-reuse.json")
    core.write(output, result)
    return {"output": str(output), "sha256": core.sha(output), "questions": result["questions"], "evaluation_mode": MODE}

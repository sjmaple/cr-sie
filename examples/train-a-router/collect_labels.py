"""Regenerate public Sol/Luna paired labels with the recorded request protocol.

This is a new public reproduction. It never replaces the original experiment.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SOURCES = HERE / "_collection"
EXPECTED = {
    "collect.py": "f6cb7fe51cc67561cf8d5457c03eb80754e85a99c0fbc48c25126815b7429100",
    "openai_population.py": "b94d77b28686230d6bdb3469e0a22577998b0a1181b152d02a82489790d87e0c",
    "grade_openai_population.py": "a7b271d76ecfed7e0e3a1876ed77f0c6b2949262b7283a1f572f9437bef8373c",
    "pair-spec.json": "4b4cc5f936fe8a8018b6845bfa9bd29b7d18ec428489d15ebab62554c4cd11b4",
}
COUNTS = {"train": 9111, "calibration": 3056, "final": 3043}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


# Preserve original import names and bytes. No private checkout or import aliases.
for _name, _expected in EXPECTED.items():
    require(sha(SOURCES / _name) == _expected, "Recorded collection source changed: " + _name)
sys.path.insert(0, str(SOURCES))
import collect  # noqa: E402
import grade_openai_population as original_grade  # noqa: E402
import openai_population as acquisition  # noqa: E402


def read(path):
    return json.loads(Path(path).read_bytes())


def checked_source(prepared):
    """Read the pinned public question/role metadata only, never gold answers."""
    prepared = Path(prepared)
    pair = read(SOURCES / "pair-spec.json")
    require(sha(prepared / "source-plan.json") == pair["source_plan_sha256"], "Use the pinned public source plan")
    source = read(prepared / "source-plan.json")
    for name in ("question-only/inputs.json", "rows.json"):
        require(sha(prepared / name) == source["files_sha256"][name], "Public source bytes changed")
    require(
        sha(prepared / "question-only/inputs.json") == pair["source_all_inputs_sha256"], "Wrong question population"
    )
    inputs, rows = read(prepared / "question-only/inputs.json"), read(prepared / "rows.json")
    require(
        [r["id"] for r in inputs] == [r["id"] for r in rows] and len({r["id"] for r in rows}) == len(rows),
        "Source order or uniqueness changed",
    )
    require(
        all(i["request"] == r["request"] and i["role"] == r["role"] for i, r in zip(inputs, rows, strict=True)),
        "Source role/request changed",
    )
    return source, pair, inputs, rows


def checked_routes(path, expected_sha, source, pair, rows):
    require(
        path is not None and expected_sha is not None, "Final collection requires a frozen route file and exact SHA"
    )
    require(sha(path) == expected_sha, "Frozen route bytes changed")
    # Optional dependency: this module is included in the public example's CPU workflow.
    import paired_calibration

    frozen = read(path)
    require(
        frozen["source_plan_sha256"] == pair["source_plan_sha256"]
        and frozen["pair_spec_sha256"] == EXPECTED["pair-spec.json"]
        and frozen["cost_scope"] == pair["cost_scope"],
        "Frozen routes use another source/pair/cost contract",
    )
    summary = paired_calibration.checked_frozen_bundle(frozen, rows)
    require(summary["n_final"] == COUNTS["final"], "Wrong final population")
    summary["validator_source_sha256"] = sha(paired_calibration.__file__)
    return frozen, summary


def plan_run(prepared, splits, budget, frozen_routes=None, frozen_sha=None):
    require(splits and len(splits) == len(set(splits)) and all(s in COUNTS for s in splits), "Invalid split selection")
    require("final" not in splits or splits == ["final"], "Final must be a separate gated collection")
    require(splits != ["calibration", "train"], "Collect training before calibration")
    cap = Decimal(str(budget))
    require(cap.is_finite() and cap > 0, "Supply a positive all-physical-attempt USD cap")
    source, pair, inputs, rows = checked_source(prepared)
    items = {
        split: [{k: i[k] for k in ("id", "request", "response_contract")} for i in inputs if i["role"] == split]
        for split in splits
    }
    require(all(len(values) == COUNTS[split] for split, values in items.items()), "Do not sample a partial split")
    for values in items.values():
        for item in values:
            collect.body_for(item, "luna")
    plan = {
        "format": "routerbench-public-paired-acquisition-v1",
        "reproduction": True,
        "original_evidence_replaced": False,
        "public_outcomes_previously_available": True,
        "new_study_preregistration_claimed": False,
        "splits": splits,
        "counts": {s: len(items[s]) for s in splits},
        "source_plan_sha256": pair["source_plan_sha256"],
        "pair_spec_sha256": EXPECTED["pair-spec.json"],
        "all_inputs_sha256": pair["source_all_inputs_sha256"],
        "cost_scope": pair["cost_scope"],
        "budget_usd": str(cap),
        "workers_per_arm": 16,
        "max_physical_attempts_per_id": 3,
        "first_terminal_response_only": True,
        "no_quality_retries": True,
        "final_dispatched": "final" in splits,
        "files_sha256": {k: v for k, v in EXPECTED.items() if k.endswith(".py")},
        "adapter_sha256": sha(__file__),
        "durability": "POSIX filesystem sync before dispatch; atomic original writes",
        "dispatch_seconds": 20900,
        "historical_pair_scope": "The pair file's no-final flag describes the original train/calibration phase.",
    }
    if "final" in splits:
        frozen, gate = checked_routes(frozen_routes, frozen_sha, source, pair, rows)
        plan.update(
            frozen_final_routing_sha256=frozen_sha,
            parent_acquisition_protocol_sha256=frozen["acquisition_protocol_sha256"],
            parent_grader_sha256=frozen["grader_sha256"],
            routing_gate=gate,
        )
    else:
        require(frozen_routes is None and frozen_sha is None, "Frozen final routes apply only to the final phase")
    return plan, items


@contextmanager
def output_lock(root):
    """A process-wide local lock prevents two writers from dispatching the same run."""
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / ("." + root.name + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another collector owns this output directory") from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def collect_run(prepared, output, splits, budget, *, resume=False, frozen_routes=None, frozen_sha=None, call=None):
    plan, items = plan_run(prepared, splits, budget, frozen_routes, frozen_sha)
    root = Path(output)
    with output_lock(root):
        if (root / "plan.json").exists():
            require(resume, "Use --resume only after the previous collector has stopped")
            require(
                (root / "plan.json").read_bytes() == acquisition.encoded(plan),
                "Resume cannot change source, splits, budget or settings",
            )
        else:
            require(not resume, "No prior run exists to resume")
            require(not root.exists() or not any(root.iterdir()), "Use an empty output directory")
            root.mkdir(parents=True, exist_ok=True)
            acquisition.dump(root / "plan.json", plan)
            (root / "pair-spec.json").write_bytes((SOURCES / "pair-spec.json").read_bytes())
            population = {
                "source_plan_sha256": plan["source_plan_sha256"],
                "pair_spec_sha256": plan["pair_spec_sha256"],
                "splits": {},
            }
            for split, values in items.items():
                acquisition.dump(root / split / "inputs.json", values)
                population["splits"][split] = {
                    "ordered_ids": [i["id"] for i in values],
                    "inputs_sha256": sha(root / split / "inputs.json"),
                    "count": len(values),
                }
            acquisition.dump(root / "population.json", population)
            if "final" in splits:
                (root / "frozen-final-routes.json").write_bytes(Path(frozen_routes).read_bytes())
                acquisition.dump(
                    root / "final-access.json",
                    {
                        "frozen_final_routing_sha256": frozen_sha,
                        "acquisition_protocol_sha256": sha(root / "plan.json"),
                        "gate": plan["routing_gate"],
                        "reproduction": True,
                        "public_outcomes_previously_available": True,
                    },
                )
        require(sha(root / "pair-spec.json") == EXPECTED["pair-spec.json"], "Saved pair changed")
        saved_population = read(root / "population.json")
        require(
            saved_population["source_plan_sha256"] == plan["source_plan_sha256"]
            and saved_population["pair_spec_sha256"] == plan["pair_spec_sha256"]
            and set(saved_population["splits"]) == set(splits),
            "Saved population changed",
        )
        for split, values in items.items():
            require(
                read(root / split / "inputs.json") == values
                and saved_population["splits"][split]["ordered_ids"] == [i["id"] for i in values]
                and saved_population["splits"][split]["inputs_sha256"] == sha(root / split / "inputs.json"),
                "Saved input order/body changed",
            )
        if "final" in splits:
            require(
                sha(root / "frozen-final-routes.json") == frozen_sha
                and read(root / "final-access.json")
                == {
                    "frozen_final_routing_sha256": frozen_sha,
                    "acquisition_protocol_sha256": sha(root / "plan.json"),
                    "gate": plan["routing_gate"],
                    "reproduction": True,
                    "public_outcomes_previously_available": True,
                },
                "Saved final gate changed",
            )
        if call is None:
            require(
                bool(os.environ.get("OPENAI_API_KEY")),
                "Set OPENAI_API_KEY for paid collection; never put it in a file here",
            )
        # The unchanged scheduler calls this after durable intents and bounded response groups.
        result = acquisition.run(
            items, root, plan, os.sync, call=call or acquisition.acquire, dispatch_seconds=plan["dispatch_seconds"]
        )
        acquisition.dump(
            root / "reproduction.json",
            {
                "original_evidence_replaced": False,
                "new_model_calls": True,
                "original_outcomes_may_be_public": True,
                "adapter_sha256": sha(__file__),
                "plan_sha256": sha(root / "plan.json"),
            },
        )
        os.sync()
        return result


def grade_run(prepared, collection, output, split):
    """Reuse original physical-attempt/usage/parser checks; never make API calls."""
    root, output = Path(collection), Path(output)
    require(not output.exists(), "Preserve prior grade bytes")
    plan, population = read(root / "plan.json"), read(root / "population.json")
    require(
        plan["format"] == "routerbench-public-paired-acquisition-v1" and plan["reproduction"] is True,
        "Grade only this adapter's distinct public reproductions",
    )
    require(plan["adapter_sha256"] == sha(__file__), "Reproduction adapter source changed")
    require(
        plan["files_sha256"] == {k: v for k, v in EXPECTED.items() if k.endswith(".py")}, "Scientific source changed"
    )
    require(split in plan["splits"], "Split was not collected")
    require(
        plan["final_dispatched"] is (split == "final") and (split != "final" or plan["splits"] == ["final"]),
        "Final phase cannot share a training collection",
    )
    source, pair, all_inputs, metadata = checked_source(prepared)
    require(sha(root / "pair-spec.json") == EXPECTED["pair-spec.json"] == plan["pair_spec_sha256"], "Pair changed")
    require(
        plan["source_plan_sha256"] == pair["source_plan_sha256"] == population["source_plan_sha256"]
        and plan["cost_scope"] == pair["cost_scope"],
        "Source/cost provenance changed",
    )
    items = [{k: i[k] for k in ("id", "request", "response_contract")} for i in all_inputs if i["role"] == split]
    require(
        read(root / split / "inputs.json") == items and len(items) == plan["counts"][split], "Input order/body changed"
    )
    ids = [i["id"] for i in items]
    require(
        population["splits"][split]["ordered_ids"] == ids
        and population["splits"][split]["inputs_sha256"] == sha(root / split / "inputs.json"),
        "Population changed",
    )
    protocol = sha(root / "plan.json")
    receipt = read(root / split / "collection.json")
    require(
        receipt["status"] == "complete"
        and receipt["split"] == split
        and receipt["expected_pairs"] == len(items)
        and receipt["plan_sha256"] == receipt["acquisition_protocol_sha256"] == protocol
        and receipt["pair_spec_sha256"] == plan["pair_spec_sha256"]
        and receipt["source_plan_sha256"] == plan["source_plan_sha256"],
        "Complete paired population required before grading",
    )
    if split == "final":
        frozen, gate = checked_routes(
            root / "frozen-final-routes.json", plan["frozen_final_routing_sha256"], source, pair, metadata
        )
        require(
            plan["parent_acquisition_protocol_sha256"] == frozen["acquisition_protocol_sha256"]
            and plan["parent_grader_sha256"] == frozen["grader_sha256"],
            "Final parent lineage changed",
        )
        require(
            read(root / "final-access.json")
            == {
                "frozen_final_routing_sha256": plan["frozen_final_routing_sha256"],
                "acquisition_protocol_sha256": protocol,
                "gate": gate,
                "reproduction": True,
                "public_outcomes_previously_available": True,
            },
            "Pre-dispatch final gate changed",
        )
    selected, selection_hashes = {}, {}
    physical = {
        "physical_attempts": 0,
        "known_list_usd": Decimal(0),
        "known_budget_usd": Decimal(0),
        "unknown_reservation_usd": Decimal(0),
    }
    for arm in ("luna", "sol"):
        path = root / split / arm / "selection.json"
        require(sha(path) == receipt["selections"][arm]["sha256"], "Selection bytes changed")
        selection = read(path)
        require(
            selection["split"] == split
            and selection["arm"] == arm
            and selection["model"] == collect.ARMS[arm]["model"]
            and selection["plan_sha256"] == protocol
            and selection["ordered_ids"] == ids
            and [e["id"] for e in selection["entries"]] == ids,
            "Selection order/model changed",
        )
        selected[arm], declared = {}, set()
        for item, entry in zip(items, selection["entries"], strict=True):
            value, ledger = original_grade.checked_attempts(root, protocol, split, arm, item, entry)
            selected[arm][item["id"]] = value
            for attempt in entry["attempts"]:
                declared.add(attempt["intent_path"])
                if attempt["record_path"] is not None:
                    declared.add(attempt["record_path"])
            for key in physical:
                physical[key] += ledger[key]
        actual = {p.relative_to(root).as_posix() for p in (root / split / arm / "attempts").rglob("*.json")}
        require(actual == declared, "Undeclared/missing physical attempt")
        selection_hashes[arm] = sha(path)
    require(sha(Path(prepared) / "gold.json") == source["files_sha256"]["gold.json"], "Reference bytes changed")
    gold, by_id = read(Path(prepared) / "gold.json"), {r["id"]: r for r in metadata}
    rows = []
    for item in items:
        identity = item["id"]
        require(gold[identity]["response_contract"] == item["response_contract"], "Reference contract changed")
        reference = gold[identity]["answer"].strip()
        if item["response_contract"]["type"] == "integer":
            reference = str(int(reference))
        row = {
            "id": identity,
            "family": by_id[identity]["family"],
            "dataset": by_id[identity]["source"],
            "task": by_id[identity]["task"],
            "role": split,
        }
        if "fold" in by_id[identity]:
            row["fold"] = by_id[identity]["fold"]
        for label, arm in (("cheap", "luna"), ("strong", "sol")):
            value = selected[arm][identity]
            row.update(
                {
                    f"{label}_correct": int(value["answer"] is not None and value["answer"] == reference),
                    f"{label}_cost_usd": value["cost_usd"],
                    f"{label}_valid": value["valid"],
                    f"{label}_record_sha256": value["record_sha256"],
                    f"{label}_intent_sha256": value["intent_sha256"],
                    f"{label}_usage_sha256": value["usage_sha256"],
                    f"{label}_selected_attempt": value["attempt"],
                    f"{label}_provider_status": value["provider_status"],
                    f"{label}_cache_write_detail_present": value["cache_write_detail_present"],
                }
            )
        rows.append(row)
    artifact = {
        "format": "routerbench-paired-grades-v1",
        "pair": "sol_luna",
        "split": split,
        "source_plan_sha256": plan["source_plan_sha256"],
        "pair_spec_sha256": plan["pair_spec_sha256"],
        "acquisition_protocol_sha256": protocol,
        "grader_sha256": sha(__file__),
        "cost_scope": pair["cost_scope"],
        "arms": pair["arms"],
        "rows": rows,
        "reproduction": True,
        "original_evidence_replaced": False,
        "base_grading_source_sha256": EXPECTED["grade_openai_population.py"],
        "provenance": {
            "split_collection_sha256": sha(root / split / "collection.json"),
            "selections_sha256": selection_hashes,
            "source_rows_sha256": source["files_sha256"]["rows.json"],
            "source_gold_sha256": source["files_sha256"]["gold.json"],
            "source_inputs_sha256": source["files_sha256"]["question-only/inputs.json"],
            "physical_acquisition": {k: str(v) if isinstance(v, Decimal) else v for k, v in physical.items()},
        },
    }
    if split == "final":
        artifact.update(
            {
                key: plan[key]
                for key in ("parent_acquisition_protocol_sha256", "parent_grader_sha256", "frozen_final_routing_sha256")
            }
        )
    acquisition.dump(output, artifact)
    return {"status": "complete", "split": split, "count": len(rows), "grades_sha256": sha(output), "new_inference": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect", help="Paid new collection; full fixed split(s), explicit cap")
    collect_parser.add_argument("--prepared", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--split", nargs="+", choices=tuple(COUNTS), required=True)
    collect_parser.add_argument("--budget-usd", required=True)
    collect_parser.add_argument("--resume", action="store_true")
    collect_parser.add_argument("--frozen-routes", type=Path)
    collect_parser.add_argument("--frozen-sha256")
    grade_parser = sub.add_parser("grade", help="CPU-only verification and grading; no API credential needed")
    grade_parser.add_argument("--prepared", type=Path, required=True)
    grade_parser.add_argument("--collection", type=Path, required=True)
    grade_parser.add_argument("--output", type=Path, required=True)
    grade_parser.add_argument("--split", choices=tuple(COUNTS), required=True)
    args = parser.parse_args()
    if args.command == "collect":
        result = collect_run(
            args.prepared,
            args.output,
            args.split,
            args.budget_usd,
            resume=args.resume,
            frozen_routes=args.frozen_routes,
            frozen_sha=args.frozen_sha256,
        )
    else:
        result = grade_run(args.prepared, args.collection, args.output, args.split)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

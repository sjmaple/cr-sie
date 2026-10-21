"""Replay one complete train/calibration split; no inference or final access."""

import argparse
from decimal import Decimal
import json
from pathlib import Path

import collect
import openai_population
from collect import ARMS, OPENAI_SETTINGS, body_for, openai_cost, parse_answer, reservation
from openai_population import decision, digest, dump, file_hash, numeric_usage, require


def read(path):
    return json.loads(Path(path).read_bytes())


def bound_path(root, relative, expected):
    require(isinstance(relative, str) and relative == expected, "Attempt path crosses its exact split/arm/ID")
    path = root / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), "Attempt path escapes collection")
    return path


def finite_usd(value):
    amount = Decimal(str(value))
    require(amount.is_finite() and amount >= 0, "Missing, negative, or nonfinite cost is not free")
    return amount


def checked_attempts(root, plan_sha, split, arm, item, entry):
    """Check every physical attempt; choose only the first terminal response."""
    identity = item["id"]
    require(entry["id"] == identity and entry["arm"] == arm and entry["split"] == split,
            "Selection entry identity/role mismatch")
    attempts = entry["attempts"]
    require(1 <= len(attempts) <= 3, "Unexpected physical attempt count")
    body = body_for(item, arm)
    body_sha = digest(json.dumps(body, sort_keys=True).encode())
    prefix = f"{split}/{arm}/attempts/{digest(identity.encode())}"
    known, budget, unknown = Decimal(0), Decimal(0), Decimal(0)
    first_terminal = None
    selected = None
    for number, attempt in enumerate(attempts, 1):
        require(first_terminal is None, "Physical attempt after a terminal provider response")
        require(attempt["attempt"] == number, "Physical attempts are not consecutive")
        intent = bound_path(root, attempt["intent_path"], f"{prefix}/{number:03d}-intent.json")
        require(file_hash(intent) == attempt["intent_sha256"], "Intent bytes changed")
        intent_row = read(intent)
        expected = {"id": identity, "arm": arm, "split": split, "attempt": number,
                    "body_sha256": body_sha, "plan_sha256": plan_sha}
        require(all(intent_row.get(key) == value for key,value in expected.items()), "Intent request/source binding mismatch")
        reserved = finite_usd(intent_row["reserved_usd"])
        require(reserved == reservation(body, arm), "Reservation differs from frozen request")
        expected_response = f"{prefix}/{number:03d}-response.json"
        if attempt["record_path"] is None:
            require(attempt["record_sha256"] is None and not (root / expected_response).exists(),
                    "Unknown attempt conceals a durable response")
            row = {"id": identity, "arm": arm, "model": body["model"], "status": "interrupted_unknown",
                   "terminal_response": False, "error_type": "ProcessInterrupted"}
        else:
            response = bound_path(root, attempt["record_path"], expected_response)
            require(file_hash(response) == attempt["record_sha256"], "Response bytes changed")
            row = read(response)
            expected_response_fields = {"id": identity, "arm": arm, "model": body["model"], "attempt": number,
                                        "body_sha256": body_sha, "intent_sha256": file_hash(intent), "binding_valid": True}
            require(all(row.get(key) == value for key,value in expected_response_fields.items()), "Response request/model binding mismatch")
        action = decision(row, number)
        require(action == attempt["action"] and row["status"] == attempt["status"], "Attempt action/status cannot be replayed")
        require(action not in {"fatal", "incomplete", "exhausted", "defer_server_delay"}, "Incomplete acquisition cannot silently enter fitting")
        if "cost" in row:
            require(isinstance(row.get("usage"), dict) and numeric_usage(row["usage"]), "Unreadable token usage")
            cost = openai_cost(row["usage"], body)
            require(cost == row["cost"], "Physical cost cannot be replayed from exact token usage")
            amount, conservative = finite_usd(cost["usd_estimate"]), finite_usd(cost["budget_usd"])
            require(conservative <= reserved and amount <= conservative, "Charge exceeds frozen bound")
            require(finite_usd(attempt["known_list_usd"]) == amount
                    and finite_usd(attempt["known_budget_usd"]) == conservative
                    and finite_usd(attempt["unknown_reservation_usd"]) == 0, "Attempt cost ledger mismatch")
            known += amount
            budget += conservative
        else:
            require(attempt["known_list_usd"] is None and attempt["known_budget_usd"] is None
                    and finite_usd(attempt["unknown_reservation_usd"]) == reserved, "Unknown execution was priced as free")
            unknown += reserved
        if row.get("terminal_response"):
            first_terminal = number
            require(action == "terminal" and row["status"] == "complete" and "cost" in row,
                    "Selected terminal response needs verified usage; no free missing receipt")
            require(row.get("resolved_model") == body["model"] and row.get("service_tier") == "default",
                    "Returned model/tier differs from frozen pair")
            require(isinstance(row.get("messages"), list), "Missing readable provider output")
            for part in row["messages"]:
                require(isinstance(part, dict) and part.get("type") in {"output_text", "refusal"}, "Unexpected output field")
                key = "text" if part["type"] == "output_text" else "refusal"
                require(set(part) == {"type", key} and isinstance(part[key], str), "Non-readable output content")
            content = "".join(part["text"] for part in row["messages"] if part["type"] == "output_text")
            answer = parse_answer(content, item["response_contract"]) if row["provider_status"] == "completed" else None
            require(answer == row.get("answer") and (answer is not None) == row.get("valid_answer"), "Saved answer cannot be replayed")
            selected = {"answer": answer, "cost_usd": float(amount), "valid": answer is not None,
                        "record_sha256": attempt["record_sha256"], "intent_sha256": attempt["intent_sha256"],
                        "usage_sha256": digest(json.dumps(row["usage"], sort_keys=True).encode()),
                        "provider_status": row["provider_status"], "attempt": number,
                        "cache_write_detail_present": row["cost"]["cache_write_detail_present"]}
    require(selected is not None and entry["selected_attempt"] == first_terminal and entry["status"] == "complete",
            "Selection must name the first terminal response")
    return selected, {"physical_attempts": len(attempts), "known_list_usd": known,
                      "known_budget_usd": budget, "unknown_reservation_usd": unknown}


def grade(prepared, collection, output, split, protocol_sha):
    require(split in {"train", "calibration"}, "Final outcomes are not authorized for this grader")
    prepared, root, output = Path(prepared), Path(collection), Path(output)
    require(not output.exists(), "Preserve previous grading outputs")
    require(file_hash(root / "plan.json") == protocol_sha, "Acquisition protocol changed")
    plan, pair = read(root / "plan.json"), read(root / "pair-spec.json")
    pair_sha = file_hash(root / "pair-spec.json")
    require(file_hash(collect.__file__) == plan["files_sha256"]["collect.py"]
            and file_hash(openai_population.__file__) == plan["files_sha256"]["openai_population.py"],
            "Frozen request/cost/selection implementation changed")
    source_plan, population = read(prepared / "source-plan.json"), read(root / "population.json")
    source_sha = file_hash(prepared / "source-plan.json")
    require(pair_sha == plan["pair_spec_sha256"] == population["pair_spec_sha256"], "Pair specification changed")
    require(source_sha == plan["source_plan_sha256"] == pair["source_plan_sha256"] == population["source_plan_sha256"], "Source plan changed")
    require(pair["pair"] == "sol_luna" and pair["arms"] == {
        "cheap": {"arm": "luna", "model": ARMS["luna"]["model"], "settings": OPENAI_SETTINGS},
        "strong": {"arm": "sol", "model": ARMS["sol"]["model"], "settings": OPENAI_SETTINGS}}, "Pair/model/settings changed")
    require(plan["cost_scope"] == pair["cost_scope"] and split in plan["splits"] and plan["final_dispatched"] is False,
            "Cost or split scope changed")
    for name in ("question-only/inputs.json", "rows.json", "gold.json"):
        require(file_hash(prepared / name) == source_plan["files_sha256"][name], "Source input/reference bytes changed")
    require(file_hash(prepared / "question-only/inputs.json") == pair["source_all_inputs_sha256"], "Pair uses another source population")
    all_inputs, metadata = read(prepared / "question-only/inputs.json"), read(prepared / "rows.json")
    require([row["id"] for row in all_inputs] == [row["id"] for row in metadata]
            and len({row["id"] for row in all_inputs}) == len(all_inputs), "Source ID order/uniqueness changed")
    expected_items = [{key:item[key] for key in ("id", "request", "response_contract")} for item in all_inputs if item["role"] == split]
    items = read(root / split / "inputs.json")
    require(items == expected_items and len(items) == plan["counts"][split], "Requested split/order/body differs from source")
    ids = [item["id"] for item in items]
    require(population["splits"][split]["ordered_ids"] == ids
            and population["splits"][split]["inputs_sha256"] == file_hash(root / split / "inputs.json"), "Population receipt changed")
    by_id = {row["id"]:row for row in metadata}
    require(all(by_id[item["id"]]["role"] == split and by_id[item["id"]]["request"] == item["request"] for item in items),
            "Source metadata role/request mismatch")
    split_receipt = read(root / split / "collection.json")
    require(split_receipt["status"] == "complete" and split_receipt["split"] == split and split_receipt["expected_pairs"] == len(items)
            and split_receipt["plan_sha256"] == protocol_sha and split_receipt["pair_spec_sha256"] == pair_sha
            and split_receipt["acquisition_protocol_sha256"] == protocol_sha and split_receipt["source_plan_sha256"] == source_sha,
            "Need a complete, bound paired split before grading")
    gold = read(prepared / "gold.json")
    selected, physical = {}, {"physical_attempts":0, "known_list_usd":Decimal(0), "known_budget_usd":Decimal(0), "unknown_reservation_usd":Decimal(0)}
    selection_hashes = {}
    for arm in ("luna", "sol"):
        path = root / split / arm / "selection.json"
        require(file_hash(path) == split_receipt["selections"][arm]["sha256"], "Selected attempt map changed")
        selection = read(path)
        require(selection["split"] == split and selection["arm"] == arm and selection["model"] == ARMS[arm]["model"]
                and selection["plan_sha256"] == protocol_sha and selection["ordered_ids"] == ids
                and [entry["id"] for entry in selection["entries"]] == ids, "Selection population/order changed")
        selection_hashes[arm] = file_hash(path)
        selected[arm] = {}
        declared_files = set()
        for item, entry in zip(items, selection["entries"], strict=True):
            value, ledger = checked_attempts(root, protocol_sha, split, arm, item, entry)
            selected[arm][item["id"]] = value
            for attempt in entry["attempts"]:
                declared_files.add(attempt["intent_path"])
                if attempt["record_path"] is not None:
                    declared_files.add(attempt["record_path"])
            for key in physical:
                physical[key] += ledger[key]
        actual_files = {path.relative_to(root).as_posix() for path in (root / split / arm / "attempts").rglob("*.json")}
        require(actual_files == declared_files, "Undeclared/missing physical attempt file in split acquisition inventory")
    rows = []
    for item in items:
        identity = item["id"]
        require(gold[identity]["response_contract"] == item["response_contract"], "Reference transport contract changed")
        reference = gold[identity]["answer"].strip()
        if item["response_contract"]["type"] == "integer":
            reference = str(int(reference))
        row = {"id": identity, "family": by_id[identity]["family"], "dataset": by_id[identity]["source"],
               "task": by_id[identity]["task"], "role": split}
        if "fold" in by_id[identity]:
            row["fold"] = by_id[identity]["fold"]
        for label, arm in (("cheap", "luna"), ("strong", "sol")):
            value = selected[arm][identity]
            row.update({f"{label}_correct": int(value["answer"] is not None and value["answer"] == reference),
                        f"{label}_cost_usd": value["cost_usd"], f"{label}_valid": value["valid"],
                        f"{label}_record_sha256": value["record_sha256"], f"{label}_intent_sha256": value["intent_sha256"],
                        f"{label}_usage_sha256": value["usage_sha256"], f"{label}_selected_attempt": value["attempt"],
                        f"{label}_provider_status": value["provider_status"],
                        f"{label}_cache_write_detail_present": value["cache_write_detail_present"]})
        rows.append(row)
    artifact = {"format": "routerbench-paired-grades-v1", "pair": "sol_luna", "split": split,
        "source_plan_sha256": source_sha, "pair_spec_sha256": pair_sha, "acquisition_protocol_sha256": protocol_sha,
        "grader_sha256": file_hash(Path(__file__)), "cost_scope": pair["cost_scope"], "arms": pair["arms"], "rows": rows,
        "grading": "Frozen development semantics: strict declared label/string equality with outer whitespace trim; canonical signed integers. Invalid/incomplete finals remain incorrect and charged.",
        "provenance": {"split_collection_sha256": file_hash(root / split / "collection.json"), "selections_sha256": selection_hashes,
             "source_rows_sha256": source_plan["files_sha256"]["rows.json"], "source_gold_sha256": source_plan["files_sha256"]["gold.json"],
             "source_inputs_sha256": source_plan["files_sha256"]["question-only/inputs.json"], "source_order_and_folds_preserved": True,
             "request_builder_sha256": plan["files_sha256"]["collect.py"], "acquisition_source_sha256": plan["files_sha256"]["openai_population.py"],
             "final_model_outcomes_read": False, "other_split_model_outcomes_read": False,
             "physical_acquisition": {key:str(value) if isinstance(value,Decimal) else value for key,value in physical.items()}}}
    dump(output, artifact)
    return {"status": "complete", "split": split, "count": len(rows), "grades_sha256": file_hash(output),
            "physical_attempts": physical["physical_attempts"], "final_model_outcomes_read": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "calibration"), required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(grade(args.prepared, args.collection, args.output, args.split, args.protocol_sha256)))


if __name__ == "__main__":
    main()

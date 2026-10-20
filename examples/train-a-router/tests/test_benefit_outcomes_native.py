"""Two-question target mapping and native training/calibration/test boundaries."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import test_benefit as fixtures
import test_benefit_outcomes as outcome_fixtures

import benefit
import benefit_outcomes as outcomes
import benefit_outcomes_native as native
from data import file_hash


def fake_laya():
    def state(_tokenizer, text, _max_len):
        return list(text.encode())

    def items(_tokenizer, rows, _max_len, _head_len, label_smoothing):
        if label_smoothing != 0:
            raise AssertionError("Outcome targets must remain hard labels")
        result = []
        for row in rows:
            for name, question in row["questions"].items():
                result.append(
                    {
                        "state_ids": list(row["state"].encode()),
                        "q": {"ins": question["instructions"], "crit": question["criteria"]},
                        "k": 2,
                        "target": [row["gold"][name]["probabilities"][option] for option in native.OPTIONS],
                    }
                )
        return result, {}

    return SimpleNamespace(encode_state=state, items_from_rows=items)


class NativeOutcomeMathTests(unittest.TestCase):
    def test_float64_replay_rejects_broadcast_nonfinite_and_meaningful_drift(self):
        values = np.array([[1e-5, 500.0], [0.0, 0.7]])
        self.assertTrue(native._matches_recorded(values, np.nextafter(values, np.inf)))
        for changed in (values[0], np.full_like(values, np.nan), np.full_like(values, np.inf), values + 1e-6):
            with self.subTest(changed=changed.tolist()):
                self.assertFalse(native._matches_recorded(values, changed))
        self.assertFalse(native._matches_recorded([], []))

    def test_known_replay_source_does_not_relax_any_other_dependency(self):
        current = native.code_hashes()
        historical = current | {"benefit_outcomes_native.py": next(iter(native.REPLAY_COMPATIBLE_NATIVE_SOURCES))}
        self.assertTrue(native._source_matches(historical, replay=True))
        self.assertFalse(native._source_matches(historical))
        for changed in (
            historical | {"benefit_outcomes_native.py": "f" * 64},
            historical | {"finetune.py": "f" * 64},
            historical | {"extra.py": "f" * 64},
            {key: value for key, value in historical.items() if key != "benefit_outcomes.py"},
        ):
            self.assertFalse(native._source_matches(changed, replay=True))

    def test_two_questions_preserve_request_arm_and_incorrect_correct_order(self):
        rows = [{"id": "a", "request": "first"}, {"id": "b", "request": "second"}]
        targets = np.array([[0, 1], [1, 0]])
        items = native.encode_rows(fake_laya(), None, rows, targets)
        self.assertEqual([item["target"] for item in items], [[1, 0], [0, 1], [0, 1], [1, 0]])
        records = [(0, [0, value], item["target"], 2) for value, item in zip((1.0, 2.0, 3.0, 4.0), items, strict=True)]
        np.testing.assert_array_equal(native.records_to_margins(records, targets), [[1, 2], [3, 4]])
        records[0], records[1] = records[1], records[0]
        with self.assertRaisesRegex(ValueError, "order"):
            native.records_to_margins(records, targets)

    def test_encoder_rejects_dropped_reordered_or_reversed_option_items(self):
        rows, targets = [{"request": "a"}, {"request": "b"}], [[0, 1], [1, 0]]
        original = fake_laya()
        for change in ("drop", "questions", "options", "requests"):

            def changed(*args, change=change, **kwargs):
                items, skipped = original.items_from_rows(*args, **kwargs)
                if change == "drop":
                    return items[:-1], {"options_collapsed": 1}
                if change == "questions":
                    items[0], items[1] = items[1], items[0]
                elif change == "options":
                    items[0]["q"]["crit"] = dict(reversed(list(items[0]["q"]["crit"].items())))
                else:
                    items[0]["state_ids"] = items[2]["state_ids"]
                return items, skipped

            with self.subTest(change=change), self.assertRaises(ValueError):
                native.encode_rows(
                    SimpleNamespace(encode_state=original.encode_state, items_from_rows=changed), None, rows, targets
                )

    def test_epoch_selection_uses_both_outcome_losses_and_includes_zero(self):
        targets = np.array([[0, 1], [1, 0]])
        with patch.dict(native.PROTOCOL, {"selection_epochs": 2}):
            # Epoch1 improves one arm while badly harming the other: choose0.
            values = [np.zeros((2, 2)), np.array([[2, 1], [-2, -1]]), np.zeros((2, 2))]
            history = [
                {"epoch": epoch, "margins": margins.tolist(), "summed_binary_nll": native.outcome_nll(margins, targets)}
                for epoch, margins in enumerate(values)
            ]
            self.assertEqual(native.selected_epoch(history, targets), 0)
            self.assertAlmostEqual(history[0]["summed_binary_nll"], 2 * np.log(2))
            history[0]["summed_binary_nll"] = 0
            with self.assertRaisesRegex(ValueError, "NLL"):
                native.selected_epoch(history, targets)
        self.assertTrue(np.isfinite(native.outcome_nll([[1000, -1000]], [[1, 0]])))

    def test_epoch_tie_uses_recorded_losses_after_numeric_validation(self):
        targets = [[0, 1]]
        loss = native.outcome_nll([[0, 0]], targets)
        history = [{"epoch": epoch, "margins": [[0, 0]], "summed_binary_nll": loss} for epoch in range(3)]
        with (
            patch.dict(native.PROTOCOL, {"selection_epochs": 2}),
            patch.object(
                native, "outcome_nll", side_effect=[np.nextafter(loss, np.inf), loss, np.nextafter(loss, -np.inf)]
            ),
        ):
            self.assertEqual(native.selected_epoch(history, targets), 0)


class NativeOutcomeLifecycleTests(unittest.TestCase):
    def setUp(self):
        fixture = outcome_fixtures.OutcomeLifecycleTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.head_run, self.run = fixture.run, fixture.root / "native-outcomes"
        protocol = patch.dict(native.PROTOCOL, {"selection_epochs": 2})
        protocol.start()
        self.addCleanup(protocol.stop)
        fixtures.save(self.run / "plan.json", native.prepare(self.head_run))

    def trained(self):
        fixture = self.fixture
        fixture.freeze()
        embedding = outcomes.checked_model(self.head_run)
        plan = native.checked_plan(self.run, self.head_run)
        targets = np.array(embedding["head"]["targets"])
        held = np.array(plan["head_plan"]["training_folds"]) == 0
        history = []
        for epoch, strength in enumerate((0, 0.8, 0.4)):
            margins = strength * (2 * targets[held] - 1)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": None if epoch == 0 else 0.5,
                    "margins": margins.tolist(),
                    "summed_binary_nll": native.outcome_nll(margins, targets[held]),
                }
            )
        artifacts = {
            **{key: plan["head_plan"]["training_artifacts_sha256"][key] for key in ("plan", "inputs", "rows")},
            "grades": embedding["training_grades_sha256"],
            "cost_proxy": embedding["cost_proxy_sha256"],
            "price_snapshot": embedding["price_snapshot_sha256"],
        }
        fixtures.save(
            self.run / "training-start.json",
            {
                "plan_sha256": file_hash(self.run / "plan.json"),
                "versions": native.PROTOCOL["versions"],
                "device": "cuda",
                "runtime": native.PROTOCOL["runtime"],
                "eager_flags": native.PROTOCOL["eager_flags"],
                "training_artifacts_sha256": artifacts,
            },
        )
        fixtures.save(self.run / "selection-config.json", native._config_values(2))
        fixtures.save(self.run / "refit-config.json", native._config_values(1))
        fixtures.save(self.run / "selection-history.json", history)
        fixtures.save(
            self.run / "epoch-selected.json",
            {
                "epoch": 1,
                "history_sha256": file_hash(self.run / "selection-history.json"),
                "validation_ids": plan["selection_validation_ids"],
                "targets": targets[held].tolist(),
                "criterion": native.PROTOCOL["selection"],
            },
        )
        checkpoints = {}
        for variant in ("base", "lora"):
            path = self.run / variant
            fixtures.save(path / "rl_agent_config.json", {})
            (path / "model.safetensors").write_bytes(b"synthetic weight bytes; never loaded")
            margins = np.zeros((int(held.sum()), 2)).tolist()
            checkpoints[variant] = {
                "files_sha256": {
                    name: file_hash(path / name) for name in ("rl_agent_config.json", "model.safetensors")
                },
                "parameters": 10,
                "reference_margins": margins,
                "merged_margins": margins,
                "exported_margins": margins,
                "max_merge_margin_error": 0.0,
                "max_export_margin_error": 0.0,
            }
        trained = benefit.seal(
            {
                "format": native.TRAINED_FORMAT,
                "plan_sha256": file_hash(self.run / "plan.json"),
                "head_plan_sha256": plan["head_plan_sha256"],
                "training_start_sha256": file_hash(self.run / "training-start.json"),
                "training_artifacts_sha256": artifacts,
                "training_grade_provenance": embedding["training_grade_provenance"],
                "training_targets": targets.tolist(),
                "training_cost_proxies": embedding["head"]["training_cost_proxies"],
                "conditional_cost_statistics": embedding["head"]["conditional_cost_statistics"],
                "cost_proxy_provenance": embedding["cost_proxy_provenance"],
                "selection_history": history,
                "selected_epoch": 1,
                "refit_losses": [0.5],
                "checkpoints": checkpoints,
                "receipts_sha256": {
                    name: file_hash(self.run / name)
                    for name in (
                        "selection-config.json",
                        "selection-history.json",
                        "epoch-selected.json",
                        "refit-config.json",
                    )
                },
            }
        )
        fixtures.save(self.run / "trained.json", trained)
        return trained

    def freeze(self):
        trained = self.trained()

        def infer(_checkpoint, rows, _device):
            # The first request prefers cheap; all others prefer strong. These
            # synthetic features never consult calibration or test correctness.
            result = np.tile([-2.0, 2.0], (len(rows), 1))
            result[0] = [2.0, -2.0]
            return result

        with patch.object(native, "checkpoint_margins", side_effect=infer):
            frozen = native.freeze(self.run, self.head_run, self.fixture.baseline, self.fixture.original)
        fixtures.save(self.run / "frozen.json", frozen)
        return trained, frozen

    def test_preflight_uses_training_only_and_rejects_non_cuda_before_gpu_imports(self):
        fixture, data = self.fixture, self.fixture.fixture
        context = native.preflight(
            self.run,
            self.head_run,
            fixture.baseline,
            data.training_plan,
            data.training_inputs,
            fixture.rows_path,
            data.training_grades,
            fixture.cost_path,
            fixture.price_path,
        )
        self.assertEqual(context["targets"].shape, (19, 2))
        self.assertFalse(set(context["plan"]["selection_fit_ids"]) & set(context["plan"]["selection_validation_ids"]))
        self.assertNotIn("calibration", context)
        with self.assertRaisesRegex(ValueError, "CUDA only"):
            native.train(None, None, None, None, None, None, None, None, None, device="mps")

    def test_replay_is_metadata_only_and_weight_bytes_can_be_verified_separately(self):
        self.freeze()
        native.checked_frozen(self.run, self.head_run, weights=True)
        (self.run / "base/model.safetensors").unlink()
        (self.run / "lora/model.safetensors").unlink()
        native.checked_frozen(self.run, self.head_run)
        with self.assertRaises(FileNotFoundError):
            native.checked_frozen(self.run, self.head_run, weights=True)
        receipt = native.authorize_test(self.run, self.head_run)
        self.assertEqual(receipt["checkpoint_verification"], "manifest-only")

    def test_historical_training_replays_but_cannot_train_or_freeze_under_new_source(self):
        historical = native.code_hashes() | {
            "benefit_outcomes_native.py": next(iter(native.REPLAY_COMPATIBLE_NATIVE_SOURCES))
        }
        with patch.object(native, "code_hashes", return_value=historical):
            fixtures.save(self.run / "plan.json", native.prepare(self.head_run))
            self.freeze()
        native.checked_frozen(self.run, self.head_run)
        receipt = native.authorize_test(self.run, self.head_run)
        self.assertEqual(receipt["replay_validation"]["training_source_code_sha256"], historical)
        self.assertEqual(receipt["replay_validation"]["validator_source_code_sha256"], native.code_hashes())
        (self.run / "frozen.json").unlink()
        (self.run / "scoring-start.json").unlink()
        for action in (
            lambda: native.checked_plan(self.run, self.head_run),
            lambda: native.freeze(self.run, self.head_run, self.fixture.baseline, self.fixture.original),
        ):
            with self.assertRaisesRegex(ValueError, "source"):
                action()

    def test_numeric_replay_preserves_stored_scores_and_cutoff_ties(self):
        _, frozen = self.freeze()
        original = outcomes.normalized_scores

        def rounded(*args):
            scores, costs = original(*args)
            return np.nextafter(scores, -np.inf), np.nextafter(costs, np.inf)

        with patch.object(outcomes, "normalized_scores", side_effect=rounded):
            _, _, replayed = native.checked_frozen(self.run, self.head_run)
        self.assertEqual(replayed, frozen)
        thresholds = 0
        for model in replayed["models"].values():
            for policy in model["policies"].values():
                if policy["mode"] == "threshold":
                    thresholds += 1
                    saved = np.asarray(model["scores"]["calibration"])
                    self.assertTrue((saved == policy["cutoff"]).any())
                    self.assertEqual((saved >= policy["cutoff"]).tolist(), policy["calibration_routes"])
        self.assertGreater(thresholds, 0)
        for field in ("scores", "costs"):

            def changed(*args, field=field):
                scores, costs = original(*args)
                return (scores + 1e-9, costs) if field == "scores" else (scores, costs + 1e-6)

            with (
                self.subTest(field=field),
                patch.object(outcomes, "normalized_scores", side_effect=changed),
                self.assertRaisesRegex(ValueError, "ratio scores"),
            ):
                native.checked_frozen(self.run, self.head_run)

    def test_resealed_export_parity_and_epoch_selection_tampering_are_rejected(self):
        trained = self.trained()
        for field in ("epoch", "parity", "costs"):
            changed = json.loads(json.dumps(trained))
            changed.pop("sha256")
            if field == "epoch":
                changed["selected_epoch"] = 2
            elif field == "parity":
                changed["checkpoints"]["lora"]["exported_margins"][0][0] = 1
            else:
                changed["conditional_cost_statistics"]["cheap_correct"]["mean_credits_proxy"] += 1
            fixtures.save(self.run / "trained.json", benefit.seal(changed))
            with self.subTest(field=field), self.assertRaises(ValueError):
                native.checked_trained(self.run, self.head_run)

    def test_test_access_is_required_and_native_router_overhead_stays_unknown(self):
        _, frozen = self.freeze()
        test = self.fixture.root / "held-test-grades.json"
        with self.assertRaises(FileNotFoundError):
            native.report(self.run, self.head_run, test)
        fixtures.save(self.run / "test-access.json", native.authorize_test(self.run, self.head_run))
        fixtures.save(self.head_run / "test-access.json", outcomes.authorize_test(self.head_run))
        rows = fixtures.outcomes("test", 8)
        for row in rows:
            row["cheap_correct"] = False
        fixture = self.fixture.fixture
        fixtures.save(
            test,
            {
                "split": "test",
                "plan_sha256": file_hash(fixture.original_plan),
                "policies_frozen_sha256": file_hash(self.fixture.original),
                **fixture.grading,
                "rows": rows,
            },
        )
        measured = native.report(self.run, self.head_run, test)
        self.assertEqual(measured["ids"], [row["id"] for row in rows])
        for variant in ("base", "lora"):
            policy = measured["models"][variant]["1"]
            self.assertEqual(policy["cutoff"], frozen["models"][variant]["policies"]["1"]["cutoff"])
            self.assertGreater(policy["metrics"]["gap_vs_strong_pp"], 1)
            self.assertFalse(policy["metrics"]["router_overhead_known"])
            self.assertNotIn("net_total_cost_usd_per1000", policy["metrics"])
        with self.assertRaises(FileExistsError):
            native.prepare(self.head_run)
        grades = outcomes.read(test)
        grades["outcome_grader_protocol_sha256"] = "e" * 64
        fixtures.save(test, grades)
        with self.assertRaisesRegex(ValueError, "grading study"):
            native.report(self.run, self.head_run, test)


if __name__ == "__main__":
    unittest.main()

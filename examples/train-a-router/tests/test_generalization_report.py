"""Exercise policy immutability and arithmetic transfer cost accounting."""

import json
import tempfile
import unittest
from pathlib import Path

import generalization_report as report
import generalization_score
import solver
import solver_report
from data import digest, file_hash, write


class TransferReportTests(unittest.TestCase):
    def test_fixed_policy_and_input_bindings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [{"id": "one", "request": "What is1+1?"}, {"id": "two", "request": "What is2+2?"}]
            write(root / "inputs.json", rows)
            write(
                root / "source-plan.json",
                {
                    "format": "sie-arithmetic-transfer-source-v1",
                    "n": 2,
                    "inputs_sha256": digest(rows),
                    "settings_sha256": digest(solver.SETTINGS),
                    "files_sha256": {"inputs.json": file_hash(root / "inputs.json")},
                },
            )
            policies = {str(i): {"mode": "threshold", "cutoff": 0.5} for i in (0, 1, 2, 5)}
            model = {"scores": [0.2, 0.8], "probabilities": [[0.5, 0.5]] * 2, "policies": policies}
            bindings = {}
            for name in report.REQUIRED:
                native = name.startswith("native-")
                models = {variant: model for variant in ("base", "lora")} if native else {"embedding": model}
                frozen = (
                    {"models": {key: {"policies": policies} for key in models}} if native else {"selections": policies}
                )
                frozen_path, score_path = root / (name + "-frozen.json"), root / (name + "-scores.json")
                write(frozen_path, frozen)
                write(
                    score_path,
                    {
                        "format": "sie-arithmetic-router-scores-v1",
                        "n": 2,
                        "ids": ["one", "two"],
                        "design": "native" if native else "embedding",
                        "models": models,
                        "inputs_sha256": file_hash(root / "inputs.json"),
                        "source_plan_sha256": file_hash(root / "source-plan.json"),
                        "frozen_sha256": file_hash(frozen_path),
                        "scoring_source_sha256": file_hash(Path(generalization_score.__file__)),
                    },
                )
                bindings[name] = (score_path, frozen_path)
            self.assertEqual(len(report.validate_scores(root, bindings)[2]), 5)
            changed_path = bindings["embedding-original"][0]
            changed = json.loads(changed_path.read_text())
            changed["models"]["embedding"]["policies"]["1"]["cutoff"] = 0.7
            changed_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "policy set"):
                report.validate_scores(root, bindings)

    def test_observed_embedding_charge_allocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root / "embeddings/collection.json", {"calls": [{"ids": ["a", "b"], "credits": 3}]})
            self.assertEqual(report.embedding_overhead(root, ["a", "b"]), {"a": 1.5, "b": 1.5})
            with self.assertRaises(ValueError):
                report.embedding_overhead(root, ["a"])

    def test_native_does_not_claim_unknown_net_cost(self):
        rows = [
            {"id": "a", "cheap_correct": True, "strong_correct": True, "cheap_credits": 10, "strong_credits": 100},
            {"id": "b", "cheap_correct": False, "strong_correct": True, "cheap_credits": 10, "strong_credits": 100},
        ]
        model = {"scores": [0.1, 0.9], "policies": {"1": {"mode": "threshold", "cutoff": 0.5}}}
        predictions = {
            "embedding-original": {"models": {"embedding": model}},
            "native-original": {"models": {"lora": model}},
        }
        result = report.summarize(rows, predictions, {"a": 1, "b": 1})
        embedding = result["embedding-original"]["1"]["metrics"]
        native = result["native-original/lora"]["1"]["metrics"]
        self.assertAlmostEqual(embedding["net_savings_pct"], 44)
        self.assertAlmostEqual(native["gross_solver_savings_pct"], 45)
        self.assertNotIn("net_savings_pct", native)
        self.assertEqual(native["gap_vs_strong_pp"], 0)

    def test_native_static_bypass_has_zero_router_cost(self):
        rows = [
            {"id": "a", "cheap_correct": True, "strong_correct": True, "cheap_credits": 10, "strong_credits": 100},
            {"id": "b", "cheap_correct": False, "strong_correct": True, "cheap_credits": 10, "strong_credits": 100},
        ]
        model = {
            "scores": [0.1, 0.9],
            "policies": {
                "0": {"mode": "always_strong", "cutoff": None},
                "1": {"mode": "threshold", "cutoff": 0.5},
                "5": {"mode": "always_cheap", "cutoff": None},
            },
        }
        result = report.summarize(rows, {"native-original": {"models": {"base": model}}}, {"a": 1, "b": 1})
        native = result["native-original/base"]
        for target in ("0", "5"):
            metrics = native[target]["metrics"]
            self.assertEqual(metrics["router_cost_usd_per1000"], 0)
            self.assertEqual(metrics["net_savings_pct"], metrics["gross_solver_savings_pct"])
            self.assertFalse(metrics["router_overhead_applied"])
        self.assertNotIn("net_savings_pct", native["1"]["metrics"])

    def test_transfer_markdown_retains_estimated_cost_scope_and_visible_intervals(self):
        rows = [
            {"id": "a", "cheap_correct": True, "strong_correct": True, "cheap_credits": 10, "strong_credits": 100},
            {"id": "b", "cheap_correct": False, "strong_correct": True, "cheap_credits": 10, "strong_credits": 100},
        ]
        model = {"scores": [0.1, 0.9], "policies": {"1": {"mode": "always_strong", "cutoff": None}}}
        result = {
            "population": "Synthetic transfer fixture",
            "baselines": {
                name: solver_report.summarize(rows, name, router_credits=0) for name in solver_report.BASELINES
            },
            "methods": report.summarize(rows, {"native-original": {"models": {"base": model}}}, {"a": 1, "b": 1}),
            "cost_scope": "Synthetic list-price estimates, not a paid invoice.",
            "selection": "Fixed competition cutoffs.",
            "intervals": "Descriptive question-level intervals.",
        }
        rendered = report.markdown(result)
        self.assertIn("estimates from observed tokens", rendered)
        self.assertIn("per **10,000 requests**", rendered)
        self.assertIn("+0.00 [+0.00, +0.00]", rendered)
        self.assertIn("Laya unchanged; original training set (always 27B; router skipped)", rendered)
        self.assertIn("$10.000 | $0.000 | $10.000 | $0.000 | $0.000 estimated API savings", rendered)
        self.assertIn("(estimated API)", rendered)


if __name__ == "__main__":
    unittest.main()

"""The live path must obey its frozen cutoff and call only the selected solver."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import benefit_route
from data import digest, file_hash, write


class BenefitRouteTests(unittest.TestCase):
    def setUp(self):
        self.frozen = {
            "head": {"weights": [0.4, 0.6], "bias": 0.1},
            "selections": {"1": {"mode": "threshold", "cutoff": 0.5}},
            "plan": {"solver_settings_sha256": digest(benefit_route.solver.SETTINGS)},
        }

    def test_cutoff_equality_selects_strong_and_static_rules_skip_embedding(self):
        with patch.object(benefit_route, "embed", return_value=(np.array([1, 0]), {"credits": 2})) as encode:
            result = benefit_route.choose(self.frozen, "problem", 1)
            self.assertEqual(result["arm"], "strong")
            self.assertEqual(result["score"], 0.5)
            encode.assert_called_once_with("problem", None)
        for arm in ("cheap", "strong"):
            self.frozen["selections"]["1"] = {"mode": f"always_{arm}", "cutoff": None}
            with patch.object(benefit_route, "embed") as encode:
                self.assertEqual(benefit_route.choose(self.frozen, "problem", 1)["arm"], arm)
                encode.assert_not_called()

    def test_cloud_requires_receipt_and_self_hosted_keeps_unknown_cost(self):
        with patch.object(benefit_route, "embed", return_value=(np.array([0, 1]), {"credits": None})):
            with self.assertRaisesRegex(ValueError, "charge receipt"):
                benefit_route.choose(self.frozen, "problem", 1)
            self.assertIsNone(
                benefit_route.choose(self.frozen, "problem", 1, self_hosted=True)["embedding_receipt"]["credits"]
            )

    def test_outcome_route_uses_both_probabilities_and_expected_cost(self):
        head = {
            "models": {
                "cheap": {"weights": [0, 0], "bias": -float(np.log(9))},
                "strong": {"weights": [0, 0], "bias": float(np.log(9))},
            },
            "conditional_cost_statistics": {
                f"{arm}_{state}": {"mean_credits_proxy": cost}
                for arm, cost in (("cheap", 10), ("strong", 100))
                for state in ("correct", "wrong")
            },
        }
        frozen = {"selections": {"1": {"mode": "threshold", "cutoff": 0.008}}}
        with patch.object(benefit_route, "embed", return_value=(np.array([1, 0]), {"credits": 2})) as encode:
            result = benefit_route.choose(frozen, "question", 1, outcome_head=head)
            self.assertAlmostEqual(result["score"], 0.8 / 90)
            self.assertEqual(result["arm"], "strong")
            self.assertEqual(result["router"], "qwen_embedding_outcomes")
            encode.assert_called_once()
        frozen["selections"]["1"] = {"mode": "always_cheap", "cutoff": None}
        with patch.object(benefit_route, "embed") as encode:
            self.assertEqual(benefit_route.choose(frozen, "question", 1, outcome_head=head)["arm"], "cheap")
            encode.assert_not_called()

    def test_live_path_preserves_receipts_and_never_runs_both_solvers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root / "model/frozen.json", self.frozen)
            selected = benefit_route.solver.SETTINGS["arms"]["cheap"]

            def solve(row, arm, settings, **kwargs):
                self.assertEqual(arm, "cheap")
                result = benefit_route.solver.initial_record(row, arm, settings)
                return result | {
                    "status": "complete",
                    "answer": "\\boxed{2}",
                    "credits": 8,
                    "observed_credits": 8,
                    "has_unknown_charges": False,
                    "calls": [{"billing": "cloud", "receipt": {"credits": 8}}],
                }

            with (
                patch.object(benefit_route.benefit, "_checked_frozen", return_value=self.frozen),
                patch.object(benefit_route, "embed", return_value=(np.array([0, 0]), {"credits": 2})),
                patch.object(benefit_route, "SingleSendClient") as client,
                patch.object(benefit_route.solver, "solve", side_effect=solve) as generate,
            ):
                client.return_value.__enter__.return_value.get_model.return_value = {
                    "name": selected["model"],
                    "revision": selected["revision"],
                }
                summary = benefit_route.answer(root / "model", root / "request", "AMC8", "What is 1+1?")
                generate.assert_called_once()
                self.assertEqual(summary["total_api_credits"], 10)
                self.assertEqual(summary["embedding_calls"], 1)
                self.assertTrue((root / "request/intent.json").exists())
                self.assertTrue((root / "request/answer.json").exists())
                with self.assertRaises(FileExistsError):
                    benefit_route.answer(root / "model", root / "request", "AMC8", "What is 1+1?")
                generate.assert_called_once()

    def test_invalid_paid_embedding_preserves_its_receipt_before_raising(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(benefit_route, "SingleSendClient") as factory:
            client = factory.return_value.__enter__.return_value
            client.get_model.return_value = {"name": benefit_route.MODEL, "revision": benefit_route.REVISION}
            client.encode.return_value = {
                "dense": [0, 0],
                "request": {"id": "paid-embedding-request", "credits_debited": 3},
            }
            path = Path(temporary) / "embedding.json"
            with self.assertRaises(ValueError):
                benefit_route.embed("question", path)
            self.assertEqual(benefit_route.benefit.read(path)["credits"], 3)
            self.assertEqual(benefit_route.benefit.read(path)["request_id"], "paid-embedding-request")

    def test_native_uses_frozen_cutoff_and_verifies_weights_without_embedding_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "lora").mkdir()
            weights = root / "lora/model.safetensors"
            weights.write_bytes(b"only the selected checkpoint is downloaded")
            frozen = {
                "models": {
                    "base": {"checkpoint_files_sha256": {"model.safetensors": "0" * 64}},
                    "lora": {
                        "policies": {"1": {"mode": "threshold", "cutoff": 0.2}},
                        "checkpoint_files_sha256": {"model.safetensors": file_hash(weights)},
                    },
                }
            }
            with (
                patch.object(benefit_route.benefit_native, "checked_frozen", return_value=({}, frozen)) as check,
                patch.object(benefit_route.benefit_native, "score_checkpoint", return_value=[0.2]) as score,
                patch.object(benefit_route, "embed") as embed,
            ):
                result = benefit_route.choose_native(root, Path("head"), "question", 1, "lora", "mps")
                self.assertEqual(result["arm"], "strong")
                self.assertEqual(result["native_model_calls"], 1)
                check.assert_called_once_with(root, Path("head"))
                score.assert_called_once_with(root / "lora", [{"request": "question"}], "mps")
                embed.assert_not_called()
                self.assertGreaterEqual(result["local_checkpoint_verification_seconds"], 0)
                self.assertGreaterEqual(
                    result["local_router_end_to_end_seconds"],
                    result["local_checkpoint_verification_seconds"] + result["local_model_load_and_score_seconds"],
                )
                weights.write_bytes(b"tampered selected weights")
                with self.assertRaisesRegex(ValueError, "checkpoint bytes"):
                    benefit_route.choose_native(root, Path("head"), "question", 1, "lora", "mps")
                score.assert_called_once()
                weights.unlink()
                with self.assertRaises(FileNotFoundError):
                    benefit_route.choose_native(root, Path("head"), "question", 1, "lora", "mps")
                score.assert_called_once()

    def test_native_static_bypass_needs_no_checkpoint_files(self):
        frozen = {
            "models": {
                "lora": {
                    "policies": {"1": {"mode": "always_cheap", "cutoff": None}},
                    "checkpoint_files_sha256": {"model.safetensors": "0" * 64},
                }
            }
        }
        with (
            patch.object(benefit_route.benefit_native, "checked_frozen", return_value=({}, frozen)),
            patch.object(benefit_route.benefit_native, "score_checkpoint") as score,
            patch.object(benefit_route, "file_hash") as hash_file,
        ):
            result = benefit_route.choose_native(Path("native"), Path("head"), "question", 1, "lora", "cpu")
            self.assertEqual(result["arm"], "cheap")
            self.assertEqual(result["native_model_calls"], 0)
            self.assertIsNone(result["local_checkpoint_verification_seconds"])
            hash_file.assert_not_called()
            score.assert_not_called()


if __name__ == "__main__":
    unittest.main()

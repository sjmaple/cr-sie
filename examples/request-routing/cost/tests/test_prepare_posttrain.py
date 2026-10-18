"""Guard the next experiment's held-out boundary and training input fields."""

import unittest

import prepare_posttrain as prep


class PosttrainPreparationTests(unittest.TestCase):
    def test_confirmation_excludes_prior_ids_and_normalized_text(self):
        previous = {
            "splits": {
                "train": [{"id": "old-train", "request": "  SEEN problem  ", "label": 0}],
                "validation": [],
                "test": [{"id": "e2h-amc-eval-1", "request": "old test", "label": 1}],
            }
        }
        raw = [
            {"problem": "seen PROBLEM", "rating_quantile": 0.1},
            {"problem": "changed text but previous ID", "rating_quantile": 0.9},
            {"problem": "new routine", "rating_quantile": 0.1},
            {"problem": "new advanced", "rating_quantile": 0.9},
            {"problem": " NEW routine ", "rating_quantile": 0.1},
        ]
        rows = prep.select_confirmation(previous, raw, count=2)
        self.assertEqual({r["id"] for r in rows}, {"e2h-amc-eval-2", "e2h-amc-eval-3"})
        self.assertEqual(rows, prep.select_confirmation(previous, raw, count=2))
        with self.assertRaises(ValueError):
            prep.select_confirmation(previous, raw, count=4)

    def test_training_export_excludes_side_channel_metadata(self):
        row = {
            "id": "secret-id",
            "request": "public problem",
            "label": 1,
            "solution": "answer",
            "rating_quantile": 0.9,
            "source_tag": "AIME",
        }
        result = prep.training_case(row)
        self.assertEqual(set(result), {"state", "questions", "expected"})
        self.assertEqual(result["state"], "public problem")
        self.assertEqual(result["expected"], {"difficulty": "advanced"})
        self.assertEqual(set(result["questions"]["difficulty"]["criteria"]), {"routine", "advanced"})


if __name__ == "__main__":
    unittest.main()

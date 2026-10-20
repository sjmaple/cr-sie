import unittest

from arithmetic_data import assign, grouped_sources
from data import render_request


def row(identity, text, source="train"):
    return {"id": identity, "problem": text, "gold": "1", "dataset": "arithmetic", "original_split": source}


class ArithmeticDataTests(unittest.TestCase):
    def test_families_do_not_cross_exposed_and_fresh_splits(self):
        rows = [row(str(i), f"Distinct question number {i}.") for i in range(9)]
        exposed = [{"id": str(i), "request": render_request("GSM8K", rows[i]["problem"])} for i in range(3)]
        components = [[0, 3], [1, 2], [4], [5], [6], [7], [8]]
        counts = {"train": 3, "train_small": 2, "calibration": 2, "test": 2}
        selected, families, small, audit = assign(rows, components, exposed, counts)
        self.assertEqual(set(small), {"1", "2"})
        self.assertEqual(audit["available_unused_families"], 5)
        self.assertNotIn("3", {r["id"] for split in ("calibration", "test") for r in selected[split]})
        groups = [{families[r["id"]] for r in selected[split]} for split in ("train", "calibration", "test")]
        self.assertTrue(all(not groups[i] & groups[j] for i in range(3) for j in range(i)))

    def test_rejects_changed_exposed_request_and_insufficient_population(self):
        rows = [row(str(i), f"A question {i}") for i in range(3)]
        counts = {"train": 1, "train_small": 1, "calibration": 1, "test": 2}
        exposed = [{"id": "0", "request": render_request("GSM8K", rows[0]["problem"])}]
        with self.assertRaisesRegex(ValueError, "Not enough"):
            assign(rows, [[0], [1], [2]], exposed, counts)
        with self.assertRaisesRegex(ValueError, "match the pinned source"):
            assign(rows, [[0], [1], [2]], [{"id": "0", "request": "Changed"}], counts)

    def test_numeric_templates_are_grouped_without_answers(self):
        rows = [
            row(
                "a",
                "Mira buys 12 apples for 8 dollars and shares them equally with her 4 friends. How many apples remain?",
            ),
            row(
                "b",
                "Mira buys 24 apples for 16 dollars and shares them equally with her 8 friends. How many apples remain?",
            ),
            row("c", "A triangular garden has side lengths five seven and nine meters. What is its perimeter?"),
        ]
        components, _ = grouped_sources(rows)
        self.assertIn([0, 1], components)
        self.assertIn([2], components)

    def test_exposed_competition_overlap_is_explicit(self):
        rows = [row("known", "Same question"), {"id": "old", "problem": "Same question", "dataset": "competition"}]
        exposed = [{"id": "known", "request": render_request("GSM8K", "Same question")}]
        with self.assertRaisesRegex(ValueError, "overlaps competition"):
            assign(rows, [[0, 1]], exposed, {"train": 1, "train_small": 1, "calibration": 0, "test": 0})


if __name__ == "__main__":
    unittest.main()

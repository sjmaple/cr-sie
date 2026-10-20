"""Check source binding and answer grading without generation calls."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data import SOURCE_REVISION, file_hash, render_request
from solver_answers import extract_final_answer, grade_answer, preflight_gold, prepare_inputs


class FinalAnswerTests(unittest.TestCase):
    def test_nested_box_and_last_explicit_answer(self):
        self.assertEqual(extract_final_answer(r"Work: 5. Final answer: \boxed{\frac{1}{2}}"), r"\frac{1}{2}")
        self.assertEqual(extract_final_answer("First \\boxed{4}.\nFinal answer: 5."), "5")
        self.assertEqual(extract_final_answer(r"Answer: \fbox{\{1,2\}}"), r"\{1,2\}")
        self.assertEqual(extract_final_answer(r"Final answer: \boxed{1+\boxed{2}}"), r"1+\boxed{2}")

    def test_unfinished_final_box_cannot_reuse_an_intermediate(self):
        self.assertEqual(extract_final_answer(r"First \boxed{4}. Finally \boxed{\frac{1}{"), "")

    def test_later_final_clause_overrides_an_earlier_box(self):
        for text in (
            r"An intermediate result is \boxed{42}. Therefore, the final answer is 43.",
            r"An intermediate result is \boxed{42}. Thus the answer is 43.",
            r"An intermediate result is \boxed{42}, the final answer is 43.",
            r"An intermediate result is \boxed{42}; hence the final answer is 43.",
            r"Answer: \boxed{42}. Final answer: 43.",
            "Final answer: 42. Final answer: 43.",
        ):
            with self.subTest(text=text):
                self.assertEqual(extract_final_answer(text), "43")
        self.assertEqual(extract_final_answer(r"The result is \boxed{42}. I am confident in this result."), "42")

    def test_unmarked_reasoning_is_not_an_answer(self):
        self.assertEqual(extract_final_answer("We first find 42.\nThe calculation is unfinished."), "")
        self.assertEqual(extract_final_answer("We first find 42."), "")
        self.assertEqual(extract_final_answer("42"), "42")
        self.assertEqual(extract_final_answer(r"$\frac12$"), r"\frac12")


@unittest.skipUnless(importlib.util.find_spec("math_verify"), "Install the evaluation dependency group")
class GradingTests(unittest.TestCase):
    def test_equivalent_fractions_radicals_and_cosmetic_boxing(self):
        for gold, answer in [
            (r"\frac12", r"\boxed{\frac{2}{4}}"),
            (r"2\sqrt{3}", r"\boxed{\sqrt{12}}"),
            ("240", "Final answer: 240."),
            (r"\sqrt{113}-\sqrt{65}.", r"\boxed{\sqrt{113}-\sqrt{65}}"),
            (r"2 \frac{265}{357}", r"\boxed{\frac{979}{357}}"),
            (r"320^{\circ}", r"\boxed{320}"),
        ]:
            with self.subTest(gold=gold):
                result = grade_answer(gold, answer)
                self.assertEqual(result["status"], "graded")
                self.assertTrue(result["correct"])

    def test_tuple_order_and_cardinality_are_preserved(self):
        self.assertTrue(grade_answer("(7,9)", r"\boxed{(7,9)}")["correct"])
        self.assertFalse(grade_answer("(7,9)", r"\boxed{(9,7)}")["correct"])
        self.assertFalse(grade_answer("(7,9)", r"\boxed{7}")["correct"])
        self.assertTrue(grade_answer(r"(\frac12,9)", r"\boxed{(0.5,9)}")["correct"])

    def test_named_text_is_exact_and_not_an_intermediate(self):
        self.assertTrue(grade_answer(r"\text{Aaron}", r"\boxed{\text{Aaron}}")["correct"])
        self.assertTrue(grade_answer(r"\text{Aaron}", "Final answer: aaron")["correct"])
        self.assertTrue(grade_answer("Yes", "Final answer: yes")["correct"])
        self.assertFalse(grade_answer("Yes", "Final answer: no")["correct"])
        self.assertFalse(grade_answer(r"\text{Aaron}", "Aaron was considered.\nFinal answer: Dan")["correct"])
        self.assertEqual(grade_answer("42", "We first find 42.")["status"], "no_final_answer")

    def test_multiple_tuples_preserve_coordinates_but_not_set_order(self):
        gold = "(2,2),(12,17),(17,12)"
        self.assertTrue(grade_answer(gold, r"\boxed{(17,12),(2,2),(12,17)}")["correct"])
        self.assertTrue(grade_answer(gold, r"\boxed{\{(17,12),(2,2),(12,17)\}}")["correct"])
        self.assertFalse(grade_answer(gold, r"\boxed{(17,12),(2,2),(17,12)}")["correct"])

    def test_math_verify_timeouts_are_recorded(self):
        from math_verify.errors import TimeoutException

        with patch("solver_answers._parse", side_effect=TimeoutException("bounded")):
            self.assertEqual(grade_answer("42", "42")["status"], "gold_unparseable")
            self.assertFalse(preflight_gold({"problem": "42"})["ready"])

    def test_wrong_and_unparseable_answers_remain_distinct(self):
        self.assertFalse(grade_answer("42", r"\boxed{43}")["correct"])
        self.assertEqual(grade_answer("42", r"\boxed{\frac{}{}}")["status"], "prediction_unparseable")
        self.assertEqual(grade_answer(r"\frac{}{A}", r"\boxed{42}")["status"], "gold_unparseable")

    def test_percentages_have_mathematical_scale(self):
        self.assertFalse(grade_answer("42", r"\boxed{42\%}")["correct"])
        self.assertFalse(grade_answer(r"42\%", r"\boxed{42}")["correct"])
        self.assertTrue(grade_answer(r"\frac12", r"\boxed{50\%}")["correct"])
        self.assertTrue(grade_answer("7", r"\boxed{6+100\%}")["correct"])
        self.assertTrue(grade_answer(r"80\%", r"\boxed{0.8}")["correct"])

    def test_false_equations_cannot_supply_only_the_correct_rhs(self):
        for answer in ("8=7", "x=8=7", "x+1=7", "7=8"):
            self.assertFalse(grade_answer("7", "\\boxed{" + answer + "}")["correct"], answer)
        for answer in ("x=7", "x=6+1=7", "6+1=7"):
            self.assertTrue(grade_answer("7", "\\boxed{" + answer + "}")["correct"], answer)

    def test_preflight_keeps_ambiguous_rows_visible(self):
        gold = {"integer": "42", "ellipsis": r"0,3,6,\ldots,6048", "pair": "(7,9)", "ratio": "6:0"}
        audit = preflight_gold(gold)
        self.assertFalse(audit["ready"])
        self.assertEqual(audit["n"], 4)
        self.assertEqual(audit["counts"], {"ready": 1, "needs_review": 3})
        self.assertEqual(set(audit["needs_review"]), {"ellipsis", "pair", "ratio"})
        self.assertFalse(grade_answer("6:0", r"\boxed{3:0}")["correct"])


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Install the evaluation dependency group")
class SourceBindingTests(unittest.TestCase):
    def setUp(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name)
        self.dataset = {"source_revision": SOURCE_REVISION, "splits": {}}
        originals = []
        for split, year in zip(("train", "validation", "calibration", "test"), (2017, 2018, 2019, 2020)):
            self.dataset["splits"][split] = []
            for label in (0, 1):
                index = len(originals)
                original = {"tag": "AMC10", "problem": f"Problem {index}", "answer": str(index), "year": year}
                originals.append(original)
                self.dataset["splits"][split].append(
                    {
                        "id": f"e2h-amc-train-{index}",
                        "request": render_request(original["tag"], original["problem"]),
                        "contest": "AMC10",
                        "year": year,
                        "label": label,
                        "family": str(index),
                        "paper_group": str(year),
                    }
                )
        table = pa.Table.from_pylist(originals)
        pq.write_table(table, self.source / "train.parquet")
        pq.write_table(table.slice(0, 0), self.source / "eval.parquet")
        pins = {name: file_hash(self.source / name) for name in ("train.parquet", "eval.parquet")}
        self.pins = patch("solver_answers.SOURCE_HASHES", pins)
        self.pins.start()
        self.addCleanup(self.pins.stop)

    def test_solver_inputs_do_not_contain_gold_or_difficulty_metadata(self):
        inputs, gold, provenance = prepare_inputs(self.source, self.dataset)
        self.assertEqual(set(inputs), {"validation", "calibration", "test"})
        self.assertEqual(len(gold), 6)
        self.assertTrue(all(set(row) == {"id", "request"} for rows in inputs.values() for row in rows))
        self.assertEqual(gold["e2h-amc-train-2"], "2")
        self.assertEqual(provenance["counts"], {"validation": 2, "calibration": 2, "test": 2})

    def test_changed_source_or_request_is_rejected(self):
        altered = copy.deepcopy(self.dataset)
        altered["splits"]["test"][0]["request"] += " changed"
        with self.assertRaisesRegex(ValueError, "join differs"):
            prepare_inputs(self.source, altered)
        with (self.source / "train.parquet").open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            prepare_inputs(self.source, self.dataset)


if __name__ == "__main__":
    unittest.main()

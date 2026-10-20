import unittest

from arithmetic_answers import clock_question, grade_reference


class ArithmeticAnswerTests(unittest.TestCase):
    def test_clock_hour_reference_accepts_cosmetic_period_and_minutes(self):
        question = "The party starts at 5:00 pm. What is the latest time of day she can begin?"
        for prediction in ("2:00 pm", r"2{:}00\ \mathrm{p.m.}", "14:00"):
            with self.subTest(prediction=prediction):
                result = grade_reference(question, "2", r"\boxed{" + prediction + "}")
                self.assertTrue(result["correct"])

    def test_clock_different_hour_or_minutes_is_wrong(self):
        question = "What is the latest time they can leave home?"
        for prediction in ("4:00 pm", "5:30 pm"):
            self.assertFalse(grade_reference(question, "5", r"\boxed{" + prediction + "}")["correct"])

    def test_duration_is_not_clock_time(self):
        question = "A train leaves at 2 pm. How many hours does its trip take?"
        self.assertFalse(clock_question(question))
        self.assertFalse(grade_reference(question, "2", r"\boxed{2:00 pm}")["correct"])

    def test_explicit_reference_period_is_preserved(self):
        question = "What time does the train leave?"
        self.assertFalse(grade_reference(question, "2:00 pm", r"\boxed{2:00 am}")["correct"])
        self.assertTrue(grade_reference(question, "2:00 pm", r"\boxed{14:00}")["correct"])

    def test_ordinary_count_keeps_existing_grading(self):
        question = "How many apples are left?"
        self.assertTrue(grade_reference(question, "6", r"\boxed{6}")["correct"])
        self.assertFalse(grade_reference(question, "6", r"\boxed{7}")["correct"])


if __name__ == "__main__":
    unittest.main()

import unittest

from teacher_format import _same_base_integer_power, grade_reference, normalize_final


class TeacherFormatTests(unittest.TestCase):
    def test_huge_positive_integer_powers_use_exact_bounded_exponent_comparison(self):
        result = grade_reference("Find the exact boundary.", "2001^{2002}", r"\boxed{2001^{2002^{2003^{2004}}}}")
        self.assertFalse(result["correct"])
        self.assertEqual(result["status"], "graded")
        self.assertIn("integer_power_guard", result)
        self.assertTrue(_same_base_integer_power("2^{1024}", "2^{2^{10}}"))
        self.assertFalse(_same_base_integer_power("2^{3^2}", "2^{(3^2)^2}"))
        self.assertTrue(_same_base_integer_power("2^{3^2}", "2^{9}"))
        self.assertIsNone(_same_base_integer_power("4^3", "2^6"))
        self.assertIsNone(_same_base_integer_power("(-2)^2", "(-2)^3"))
        self.assertIsNone(_same_base_integer_power("1^2", "1^3"))
        self.assertIsNone(_same_base_integer_power("2^0", "2^1"))
        self.assertIsNone(_same_base_integer_power("2^{1 2}", "2^{12}"))
        self.assertIsNone(_same_base_integer_power("2^{10^{10}}", "2^{10^{11}}"))

    def test_real_odd_radicals_preserve_arbitrary_complex_powers(self):
        question = "Find the real root of the cubic equation."
        original = r"\sqrt[3]{(1-\sqrt{33})/2}"
        equivalent = r"-\sqrt[3]{(\sqrt{33}-1)/2}"
        self.assertTrue(grade_reference(question, original, r"\boxed{" + equivalent + "}")["correct"])
        self.assertTrue(grade_reference(question, r"\sqrt[3]{-8}", r"\boxed{-2}")["correct"])
        self.assertFalse(grade_reference(question, r"\sqrt[3]{-8}", r"\boxed{2}")["correct"])
        self.assertEqual(normalize_final("Find the complex principal value.", original)["normalized"], original)
        self.assertEqual(normalize_final(question, "(-8)^{1/3}")["normalized"], "(-8)^{1/3}")
        self.assertEqual(normalize_final(question, r"\sqrt[4]{-16}")["normalized"], r"\sqrt[4]{-16}")

    def test_year_prefix_requires_requested_year(self):
        self.assertTrue(
            grade_reference("What is the first year in which this happens?", "13", r"\boxed{Year13}")["correct"]
        )
        self.assertFalse(
            grade_reference("What is the first year in which this happens?", "13", r"\boxed{Year14}")["correct"]
        )
        self.assertEqual(normalize_final("How many coins?", "Year13")["normalized"], "Year13")

    def test_currency_and_unit_conversion(self):
        question = "Patty has nickels and dimes. How much are her coins worth?"
        self.assertTrue(grade_reference(question, r"115\text{ cents}", r"\boxed{1.15}")["correct"])
        self.assertFalse(grade_reference(question, r"115\text{ cents}", r"\boxed{115}")["correct"])
        self.assertTrue(grade_reference("How many dollars must he pay?", r"\$36.00", r"\boxed{36}")["correct"])
        self.assertTrue(grade_reference("How many cents does it cost?", r"1.15 dollars", r"\boxed{115}")["correct"])
        self.assertTrue(
            grade_reference("How many kilometers does it represent?", "102 kilometers", r"\boxed{102}")["correct"]
        )
        self.assertEqual(normalize_final("Solve x+1=2", "2 dollars")["normalized"], "2 dollars")
        self.assertTrue(
            grade_reference("What speed in miles per hour?", r"22\text{ miles per hour}", r"\boxed{22}")["correct"]
        )
        self.assertEqual(
            normalize_final("Find the number of hours.", r"22\text{ miles per hour}")["normalized"],
            r"22\text{ miles per hour}",
        )

    def test_named_set_assignment_and_tuple(self):
        reference = r"(a,b)\in\left\{(1,2),\ (4,2),\ \left(\frac52,2+\frac32 i\right),\ \left(\frac52,2-\frac32 i\right)\right\}"
        expected = r"\{(1,2),(4,2),(5/2,2+3i/2),(5/2,2-3i/2)\}"
        self.assertTrue(
            grade_reference("Find all ordered pairs (a,b) of complex numbers.", reference, r"\boxed{" + expected + "}")[
                "correct"
            ]
        )
        self.assertTrue(grade_reference("Solve for x.", "x=(1,2)", r"\boxed{(1,2)}")["correct"])
        self.assertFalse(grade_reference("Find a number.", "7", r"\boxed{8=7}")["correct"])
        self.assertFalse(grade_reference("Find a value for complex numbers.", "1/2^1000", r"\boxed{0}")["correct"])

    def test_approximation_does_not_add_tolerance(self):
        self.assertEqual(normalize_final("Find x.", r"\approx1129.42")["normalized"], "1129.42")
        self.assertTrue(grade_reference("Find x.", r"\approx1129.42", r"\boxed{1129.42}")["correct"])
        self.assertFalse(grade_reference("Find x.", r"\approx1129.42", r"\boxed{1129}")["correct"])

    def test_ratio_and_abstention(self):
        self.assertTrue(
            grade_reference("What is the resulting ratio of cream amounts?", "7:6", r"\boxed{7/6}")["correct"]
        )
        abstention = "Insufficient information. Please provide the diagram."
        self.assertFalse(
            grade_reference("Find the circle areas.", abstention, r"\boxed{" + abstention + "}")["correct"]
        )

    def test_thousands_only_for_complete_numeric_literals(self):
        self.assertEqual(normalize_final("How many?", "1,234.50")["normalized"], "1234.50")
        self.assertEqual(normalize_final("Give a pair.", "(1,234)")["normalized"], "(1,234)")
        self.assertEqual(normalize_final("Give a list.", "1,2,3")["normalized"], "1,2,3")
        self.assertTrue(grade_reference("How many dollars?", r"\$1,234.50", r"\boxed{1234.5}")["correct"])

    def test_yes_no_prefix_requires_binary_question(self):
        question = "Letf be a bijection. Must there exist such an integer?"
        explanation = r"\text{No. A counterexample is } f(n)=n+2."
        self.assertTrue(grade_reference(question, "No", r"\boxed{" + explanation + "}")["correct"])
        self.assertFalse(grade_reference(question, "Yes", r"\boxed{" + explanation + "}")["correct"])
        self.assertEqual(
            normalize_final("What is its value?", explanation)["normalized"], explanation.removesuffix(".")
        )

    def test_tuple_set_tex_space_is_cosmetic(self):
        self.assertTrue(
            grade_reference(
                "Find all ordered pairs.", "(2,2),(12,17),(17,12)", r"\boxed{\{(2,2),\ (12,17),\ (17,12)\}}"
            )["correct"]
        )
        self.assertFalse(
            grade_reference("Find all ordered pairs.", "(2,2),(12,17),(17,12)", r"\boxed{\{(2,2),\ (12,17)\}}")[
                "correct"
            ]
        )

    def test_incidental_units_do_not_determine_answer_units(self):
        question = "Walk12 miles in3 hours. How many hours?"
        self.assertFalse(grade_reference(question, "3", r"\boxed{3 miles}")["correct"])
        self.assertTrue(grade_reference(question, "3", r"\boxed{3 hours}")["correct"])
        self.assertFalse(grade_reference("The coins cost6 dollars. How many coins?", "6", r"\boxed{\$6}")["correct"])
        self.assertTrue(grade_reference("How many coins?", "6", r"\boxed{6 coins}")["correct"])
        self.assertFalse(grade_reference("What is x?", "3", r"\boxed{3 miles}")["correct"])
        self.assertFalse(
            grade_reference("Tickets cost3 dollars. How much time did he wait?", "3", r"\boxed{\$3}")["correct"]
        )
        self.assertFalse(
            grade_reference("The mixture has ratio2:1. What is its mass?", "1.5", r"\boxed{3:2}")["correct"]
        )
        self.assertTrue(
            grade_reference("How many conference games are scheduled?", "96", r"\boxed{96 games}")["correct"]
        )
        self.assertTrue(
            grade_reference(
                "On how many days during the next year did she receive no call?", "146", r"\boxed{146 days}"
            )["correct"]
        )

    def test_angle_units_are_exact_and_question_conditioned(self):
        question = r"A triangle has an angle of10^\circ. What is\angle ABC?"
        self.assertTrue(grade_reference(question, r"\pi/12", r"\boxed{15^\circ}")["correct"])
        self.assertTrue(grade_reference(question, "15", r"\boxed{15^\circ}")["correct"])
        self.assertFalse(grade_reference(question, r"\pi/12", r"\boxed{16^\circ}")["correct"])
        self.assertTrue(grade_reference("What is the angle in radians?", r"\pi/12", r"\boxed{15^\circ}")["correct"])
        self.assertFalse(grade_reference("What is the angle in radians?", "15", r"\boxed{15^\circ}")["correct"])
        self.assertFalse(grade_reference("What is the angle in degrees?", r"\pi/12", r"\boxed{15^\circ}")["correct"])
        self.assertFalse(grade_reference("How many apples?", "15", r"\boxed{15^\circ}")["correct"])

    def test_requested_variable_uses_only_its_explicit_quantity_declaration(self):
        question = (
            r"Let $x$ be the angle measure in degrees of the larger interior angle. What is $x$? [asy] draw(p); [/asy]"
        )
        self.assertTrue(grade_reference(question, "100", r"\boxed{100^\circ}")["correct"])
        result = grade_reference(question, "100", r"\boxed{160^\circ}")
        self.assertFalse(result["correct"])
        self.assertEqual(result["status"], "graded")
        self.assertTrue(
            grade_reference("Let n denote the number of apples. Find n.", "7", r"\boxed{7 apples}")["correct"]
        )
        self.assertFalse(
            grade_reference("Let x be the angle in degrees. What is y?", "100", r"\boxed{100^\circ}")["correct"]
        )
        self.assertFalse(
            grade_reference("Let x be the number of coins. An angle is30 degrees. What is x?", "6", r"\boxed{6^\circ}")[
                "correct"
            ]
        )
        self.assertFalse(
            grade_reference("Let x be the angle in degrees. What is x+1?", "100", r"\boxed{100^\circ}")["correct"]
        )
        self.assertFalse(
            grade_reference(
                "Let x be the number of coins. Let X be the angle in degrees. What is x?", "6", r"\boxed{6^\circ}"
            )["correct"]
        )
        self.assertTrue(
            grade_reference(
                "Let x be the angle in degrees and y be the percentage of students present. What is x?",
                "30",
                r"\boxed{30^\circ}",
            )["correct"]
        )
        for question in (
            "Let x be the ratio of angle A in degrees to angle B in degrees. What is x?",
            "What is the ratio of angle A in degrees to angle B in degrees?",
        ):
            self.assertFalse(grade_reference(question, "1", r"\boxed{1^\circ}")["correct"])
            self.assertTrue(grade_reference(question, "1", r"\boxed{1:1}")["correct"])
        self.assertFalse(grade_reference("How many angles measure30 degrees?", "3", r"\boxed{3^\circ}")["correct"])
        for question in (
            "Let x be the number of angle bisectors. What is x?",
            "How many angle bisectors are there?",
        ):
            self.assertTrue(grade_reference(question, "6", r"\boxed{6 bisectors}")["correct"])

    def test_height_inherits_only_one_consistent_length_unit(self):
        question = "Two towers have integer heights in meters. What are the possibilities for the height?"
        self.assertTrue(grade_reference(question, "7,15", r"\boxed{\{7,15\}\text{ meters}}")["correct"])
        self.assertFalse(grade_reference(question, "7,15", r"\boxed{\{7,15\}\text{ miles}}")["correct"])
        self.assertFalse(
            grade_reference("A tower is2 meters and3 feet away. What is its height?", "7", r"\boxed{7 meters}")[
                "correct"
            ]
        )

    def test_inner_verbs_and_relative_clauses_do_not_replace_the_request(self):
        self.assertTrue(
            grade_reference("How many dollars must Pat give to Lee?", "7", r"\boxed{7\text{ dollars}}")["correct"]
        )
        self.assertTrue(
            grade_reference("How many games will be played to determine the winner?", "7", r"\boxed{7 games}")[
                "correct"
            ]
        )
        self.assertTrue(
            grade_reference("Find the smallest number of moves after which it has ended.", "7", r"\boxed{7 moves}")[
                "correct"
            ]
        )

    def test_explicit_variable_unit_adjacency_preserves_variable_identity(self):
        question = r"The solid rotates by $\theta$ degrees. What is the smallest positive value of $\theta$?"
        self.assertTrue(grade_reference(question, r"72^\circ", r"\boxed{72}")["correct"])
        self.assertFalse(grade_reference(question, r"72^\circ", r"\boxed{73}")["correct"])
        self.assertTrue(
            grade_reference(r"The stump is $x$ meters tall. What is $x$?", "2", r"\boxed{2 meters}")["correct"]
        )
        self.assertTrue(
            grade_reference(
                r"Angles are $30^\circ$ and $x^\circ$. What is the sum of possible values of $x$?",
                "70",
                r"\boxed{70^\circ}",
            )["correct"]
        )
        self.assertFalse(
            grade_reference(r"The object is $X$ meters tall. What is $x$?", "2", r"\boxed{2 meters}")["correct"]
        )


if __name__ == "__main__":
    unittest.main()

"""Sampling, warm timing and resident-model behavior without model downloads."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import test_benefit_outcomes_native as native_fixtures

import benchmark_outcomes_native as benchmark
import benefit_outcomes as outcomes


def source_rows():
    return [{"id": f"id-{index}", "request": f"problem{index}"} for index in range(512)]


def result():
    return {"score": 0.01, "margins": [1.0, 2.0], "route": "strong", "state_tokens": 3}


class NativeWarmBenchmarkTests(unittest.TestCase):
    def test_hash_sample_is_shared_between_training_sizes_and_source_orders(self):
        source = source_rows()
        eligible = [row["id"] for row in source[2:]]
        timed, warmup = benchmark.select_inputs(source, eligible)
        expanded, expanded_warmup = benchmark.select_inputs(list(reversed(source)), eligible + ["extra-training"])
        self.assertEqual(timed, expanded)
        self.assertEqual(warmup, expanded_warmup)
        self.assertEqual(len({row["id"] for row in [*timed, warmup]}), 21)
        self.assertTrue(all(row["id"] in eligible for row in [*timed, warmup]))
        self.assertNotEqual(timed, source[2:22])

    def test_wrong_population_or_duplicate_source_ids_are_rejected(self):
        source = source_rows()
        for pool, eligible in (
            (source[:-1], [row["id"] for row in source[2:]]),
            (source, [row["id"] for row in source[:20]]),
            (source, [row["id"] for row in source[2:]] + [source[2]["id"]]),
            ([source[0], *source[:-1]], [row["id"] for row in source[2:]]),
        ):
            with self.subTest(n=len(pool)), self.assertRaises(ValueError):
                benchmark.select_inputs(pool, eligible)

    def test_reviewed_exclusions_are_applied_before_shared_hash_sampling(self):
        source = source_rows()
        earlier, _ = benchmark.select_inputs(source, [row["id"] for row in source[2:]])
        excluded = {earlier[0]["id"], earlier[3]["id"], *[row["id"] for row in source[:12]]}
        eligible = [row["id"] for row in source if row["id"] not in excluded]
        original = benchmark.select_inputs(source, eligible)
        expanded = benchmark.select_inputs(source, eligible + ["expanded-only"])
        self.assertEqual(original, expanded)
        timed, warmup = original
        self.assertFalse(excluded & {row["id"] for row in [*timed, warmup]})
        self.assertEqual(len(timed), 20)

    def test_only_one_warmup_is_excluded_and_all_requests_are_synchronized(self):
        rows, warmup = source_rows()[:20], source_rows()[20]
        clock_values, now = [0.0, 3.0], 3.0
        for index in range(20):
            clock_values.extend((now, now + (index + 1) / 100))
            now = clock_values[-1]
        clock = iter(clock_values)
        score, synchronize = Mock(side_effect=lambda row: result()), Mock()
        measured = benchmark.measure_requests(score, rows, warmup, synchronize, clock=lambda: next(clock))
        self.assertEqual(
            [call.args[0]["id"] for call in score.call_args_list], [warmup["id"], *[row["id"] for row in rows]]
        )
        self.assertEqual(synchronize.call_count, 42)
        self.assertEqual(measured["warmup_seconds_excluded"], 3.0)
        self.assertEqual(measured["n"], 20)
        self.assertAlmostEqual(measured["p50_ms"], 105.0)
        self.assertAlmostEqual(measured["p95_ms"], 190.5)
        self.assertEqual([row["id"] for row in measured["requests"]], [row["id"] for row in rows])

    def test_nonfinite_result_cannot_become_a_latency_measurement(self):
        score = Mock(return_value={**result(), "score": float("nan")})
        with self.assertRaisesRegex(ValueError, "finite"):
            benchmark.measure_requests(score, source_rows()[:20], source_rows()[20], lambda: None)
        self.assertEqual(score.call_count, 1)

    def test_checkpoint_is_loaded_once_for_all_two_question_requests(self):
        lt = native_fixtures.fake_laya()
        model = Mock()
        model.encoder = SimpleNamespace(config=SimpleNamespace(reference_compile=True))
        model.to.return_value = model
        model.eval.return_value = model
        lt.load_checkpoint = Mock(return_value=(model, None, {}))

        def records(_model, _tokenizer, items, _device, _max_len, _head_len, *, batch_size):
            self.assertEqual(len(items), 2)
            self.assertEqual(batch_size, 2)
            return [(0, [0, float(index + 2)], item["target"], 2) for index, item in enumerate(items)]

        lt.calibration_records = Mock(side_effect=records)
        statistics = outcomes.conditional_costs([[0, 0], [1, 1]], [[2, 4], [1, 3]])
        router = benchmark.WarmRouter(lt, "checkpoint", "cpu", statistics, {"mode": "threshold", "cutoff": 0})
        for row in source_rows()[:21]:
            measured = router.score_one(row)
            self.assertEqual(measured["margins"], [2.0, 3.0])
            self.assertEqual(measured["route"], "strong")
        lt.load_checkpoint.assert_called_once_with("checkpoint")
        self.assertEqual(lt.calibration_records.call_count, 21)
        self.assertFalse(model.encoder.config.reference_compile)
        model.eval.assert_called_once()

    def test_existing_output_is_rejected_before_loading_or_verifying_models(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text("preserved")
            with patch.object(benchmark.native, "checked_frozen") as check, self.assertRaises(FileExistsError):
                benchmark.run("missing", "missing", "missing", output)
            check.assert_not_called()
            self.assertEqual(output.read_text(), "preserved")


if __name__ == "__main__":
    unittest.main()

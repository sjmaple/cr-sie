"""Synthetic metadata through the actual grade, calibration and report contracts.

No fitting, API requests, original outcomes or large source populations are used.
"""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import collect_labels
import label_workflow as workflow


core = workflow.core


class NewLabelLineageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.pair = collect_labels.read(collect_labels.SOURCES / "pair-spec.json")
        (self.run / "pair.json").write_bytes((collect_labels.SOURCES / "pair-spec.json").read_bytes())
        self.rows = []
        for role, count in (("train", 5), ("calibration", 2), ("final", 2)):
            for index in range(count):
                row = {
                    "id": f"{role}-{index}",
                    "role": role,
                    "family": f"{role}-family-{index}",
                    "source": "bbh",
                    "task": "navigate",
                    "request": "Synthetic request",
                }
                if role == "train":
                    row["fold"] = index
                self.rows.append(row)
        core.write(self.run / "source-rows.json", self.rows)
        self.plan = {
            "protocol": core.PROTOCOL,
            "source_code_sha256": core.sources(),
            "source_plan_sha256": self.pair["source_plan_sha256"],
            "source_rows_sha256": core.sha(self.run / "source-rows.json"),
            "pair": self.pair,
            "pair_sha256": core.sha(self.run / "pair.json"),
            "pair_spec_sha256": core.sha(self.run / "pair.json"),
            "acquisition_protocol_sha256": "d" * 64,
            "populations": {
                role: [r["id"] for r in self.rows if r["role"] == role]
                for role in ("train", "calibration", "final", "development")
            },
        }
        core.write(self.run / "plan.json", self.plan)
        train, calibration = self.grades("train"), self.grades("calibration")
        self.train_path, self.cal_path = self.root / "train.json", self.root / "calibration.json"
        core.write(self.train_path, train)
        core.write(self.cal_path, calibration)
        _, labels, costs = core.grade_arrays(self.plan, self.rows, train, "train")
        statistics = core.conditional_usd(labels, costs)
        common = {
            "plan_sha256": core.sha(self.run / "plan.json"),
            "training_grades_sha256": core.sha(self.train_path),
            "grader_sha256": train["grader_sha256"],
        }
        ids = self.plan["populations"]["train"]
        core.write(
            self.run / "embedding.json", {**common, "fit": {"training_ids": ids, "conditional_cost_usd": statistics}}
        )
        core.write(
            self.run / "native/trained.json", {**common, "training_ids": ids, "conditional_cost_usd": statistics}
        )
        core.write(self.run / "topic-model.json", {**common, "training_ids": ids})
        roles = {
            role: {
                "ids": self.plan["populations"][role],
                "probabilities": [[0.9, 0.8], [0.1, 0.9]],
                "expected_usd": [[0.1, 1.0]] * 2,
            }
            for role in ("calibration", "final")
        }
        base = {"plan_sha256": core.sha(self.run / "plan.json"), "outcomes_accessed": False}
        core.write(
            self.run / "embedding-scores.json",
            {**base, "model_sha256": core.sha(self.run / "embedding.json"), "roles": roles},
        )
        core.write(
            self.run / "native-scores.json",
            {
                **base,
                "trained_sha256": core.sha(self.run / "native/trained.json"),
                "variants": {
                    variant: {"roles": roles, "conditional_cost_usd": statistics} for variant in ("base", "lora")
                },
            },
        )
        topic_roles = deepcopy(roles)
        for value in topic_roles.values():
            value["source_topic_diagnostic"] = deepcopy(value)
        core.write(
            self.run / "topic-scores.json",
            {**base, "model_sha256": core.sha(self.run / "topic-model.json"), "roles": topic_roles},
        )
        self.fees = self.root / "fees.json"
        core.write(
            self.fees,
            {
                "format": "routerbench-singleton-embedding-cost-projection-v1",
                "model": core.MODEL,
                "hf_revision": core.REVISION,
                "instruction": core.INSTRUCTION,
                "batch_validation": {"rows": len(self.rows), "every_observed_batch_reproduced": True},
                "singleton_validation": {"rows": 90, "batches": 90, "every_observed_batch_reproduced": True},
                "rows": [
                    {"id": row["id"], "role": row["role"], "projected_singleton_usd": "0.00001"} for row in self.rows
                ],
            },
        )

    def tearDown(self):
        self.temporary.cleanup()

    def grades(self, split):
        values = []
        for index, row in enumerate(r for r in self.rows if r["role"] == split):
            values.append(
                {
                    "id": row["id"],
                    "role": split,
                    "dataset": row["source"],
                    "family": row["family"],
                    "task": row["task"],
                    "cheap_correct": int(index % 2 == 0),
                    "strong_correct": int(index % 3 != 2),
                    "cheap_cost_usd": 0.1,
                    "strong_cost_usd": 1.0,
                    "cheap_record_sha256": "1" * 64,
                    "strong_record_sha256": "2" * 64,
                }
            )
        return {
            "format": "routerbench-paired-grades-v1",
            "pair": "sol_luna",
            "split": split,
            "source_plan_sha256": self.plan["source_plan_sha256"],
            "pair_spec_sha256": self.plan["pair_spec_sha256"],
            "acquisition_protocol_sha256": self.plan["acquisition_protocol_sha256"],
            "grader_sha256": "e" * 64,
            "arms": self.pair["arms"],
            "cost_scope": self.pair["cost_scope"],
            "rows": values,
        }

    def freeze_and_final(self):
        envelope = workflow.calibrate(self.run, self.cal_path, self.fees)
        raw = self.run / "frozen/internal/frozen-final-routes.json"
        grades = self.grades("final")
        grades.update(
            acquisition_protocol_sha256="f" * 64,
            parent_acquisition_protocol_sha256=self.plan["acquisition_protocol_sha256"],
            parent_grader_sha256="e" * 64,
            frozen_final_routing_sha256=core.sha(raw),
        )
        path = self.root / "final.json"
        core.write(path, grades)
        return envelope, raw, path

    def test_new_protocol_flows_to_raw_final_gate_and_retrospective_report(self):
        envelope, raw, final = self.freeze_and_final()
        self.assertFalse(envelope["frozen_before_final_model_outcomes"])
        with patch.dict(collect_labels.COUNTS, {"final": 2}):
            _, summary = collect_labels.checked_routes(raw, core.sha(raw), {}, self.pair, self.rows)
        self.assertEqual(summary["policies"], 26)
        output = self.root / "report.json"
        workflow.evaluate(self.run, final, output)
        result = core.read(output)
        self.assertFalse(result["new_blind_evaluation"])
        self.assertEqual(result["reproduction_kind"], "new_label_acquisition")
        self.assertEqual(result["provenance"]["parent_acquisition_protocol_sha256"], "d" * 64)
        self.assertEqual(result["questions"], 2)

    def test_original_reference_protocol_cannot_replace_new_train_or_final_lineage(self):
        changed = self.grades("train")
        changed["acquisition_protocol_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "contract changed"):
            core.grade_arrays(self.plan, self.rows, changed, "train")
        _, _, final = self.freeze_and_final()
        changed = core.read(final)
        changed["parent_acquisition_protocol_sha256"] = "a" * 64
        bad = self.root / "bad-final.json"
        core.write(bad, changed)
        with self.assertRaisesRegex(ValueError, "parent lineage"):
            workflow.evaluate(self.run, bad, self.root / "report.json")

    def test_changed_raw_freeze_is_rejected_by_outer_envelope(self):
        _, raw, final = self.freeze_and_final()
        raw.write_text(raw.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "route artifact changed"):
            workflow.evaluate(self.run, final, self.root / "report.json")

    def test_mismatched_calibration_protocol_stops_before_freeze(self):
        changed = self.grades("calibration")
        changed["acquisition_protocol_sha256"] = "a" * 64
        path = self.root / "bad-calibration.json"
        core.write(path, changed)
        with self.assertRaisesRegex(ValueError, "contract changed"):
            workflow.calibrate(self.run, path, self.fees)
        self.assertFalse((self.run / "frozen").exists())


if __name__ == "__main__":
    unittest.main()

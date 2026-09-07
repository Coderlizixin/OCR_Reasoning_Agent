import json
import tempfile
import unittest
from pathlib import Path

from rig_eval import load_predictions
from test_H2_Static_Plan_at6_on_Charxiv import (
    GATE_VERSION,
    K,
    build_prefix_result,
    canonical_json_sha256,
    freeze_plan_directory,
    parse_fixed_plan,
    protocol_spec,
    verify_plan_directory,
)


def six_steps():
    return [
        {"step_id": f"s{position}", "evidence_need": f"need {position}"}
        for position in range(1, K + 1)
    ]


class StaticPlanAt6Tests(unittest.TestCase):
    def test_parser_requires_exact_six_canonical_ids(self):
        parsed = parse_fixed_plan(json.dumps({"steps": six_steps()}))
        self.assertEqual(len(parsed), 6)
        self.assertEqual(parsed[-1]["step_id"], "s6")
        self.assertIsNone(parse_fixed_plan(json.dumps({"steps": six_steps()[:5]})))

    def test_parser_rejects_duplicate_needs(self):
        steps = six_steps()
        steps[-1]["evidence_need"] = steps[0]["evidence_need"]
        self.assertIsNone(parse_fixed_plan(json.dumps({"steps": steps})))

    def test_prefix_union_contains_only_first_k_steps(self):
        records = [
            {
                "step_id": f"s{i}",
                "status": "valid",
                "elements": [
                    {
                        "description": f"clue {i}",
                        "bbox": [i * 10, 0, i * 10 + 5, 5],
                    }
                ],
                "token_usage": {"prompt": 1, "completion": 2, "total": 3},
                "elapsed_seconds": 1.0,
            }
            for i in range(1, 7)
        ]
        result = build_prefix_result(
            index=1,
            question="q",
            question_source="raw_question",
            category=1,
            steps=six_steps(),
            records=records,
            prefix_k=3,
            model_name="model",
        )
        self.assertEqual(result["prefix_k"], 3)
        self.assertEqual(len(result["elements"]), 3)
        self.assertEqual(result["token_usage"]["total"], 9)

    def test_prefix_result_is_evaluator_compatible(self):
        result = build_prefix_result(
            index=1,
            question="q",
            question_source="raw_question",
            category=1,
            steps=six_steps(),
            records=[
                {
                    "step_id": "s1",
                    "status": "valid",
                    "elements": [{"description": "axis", "bbox": [0, 0, 100, 100]}],
                    "token_usage": {},
                    "elapsed_seconds": 1,
                }
            ],
            prefix_k=1,
            model_name="model",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rig_results.json"
            path.write_text(json.dumps([result]), encoding="utf-8")
            self.assertEqual(len(load_predictions(path)[1]["elements"]), 1)

    def test_plan_freeze_rejects_non_six_step_plan_and_verifies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "experiment_version": GATE_VERSION,
                "plan_only": True,
                "selected_indices": [1],
                "protocol_sha256": canonical_json_sha256(protocol_spec()),
            }
            status = {"complete": True, "request_failed_samples": 0}
            (root / "experiment_config.json").write_text(json.dumps(config), encoding="utf-8")
            (root / "run_status.json").write_text(json.dumps(status), encoding="utf-8")
            bad = [{"index": 1, "status": "valid", "steps": six_steps()[:5]}]
            (root / "plans.json").write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "six-step"):
                freeze_plan_directory(root)
            good = [{"index": 1, "status": "valid", "steps": six_steps()}]
            (root / "plans.json").write_text(json.dumps(good), encoding="utf-8")
            freeze_plan_directory(root)
            self.assertTrue(verify_plan_directory(root)["valid"])


if __name__ == "__main__":
    unittest.main()

import unittest

from rig_eval.core import EvaluationConfig
from test_H2_Oracle_Grounding_on_Charxiv import (
    evaluate_record,
    make_report,
    parse_oracle_response,
)


class OracleGroundingTest(unittest.TestCase):
    def test_parse_valid_json_after_reasoning(self):
        parsed = parse_oracle_response(
            'thinking {not json}\n{"elements":[{"description":"axis","bbox":[10,20,30,40]}]}'
        )
        self.assertEqual(parsed[0]["bbox"], [10, 20, 30, 40])

    def test_parse_rejects_invalid_box(self):
        self.assertIsNone(
            parse_oracle_response(
                '{"elements":[{"description":"axis","bbox":[30,20,10,40]}]}'
            )
        )

    def test_controlled_containment_profile(self):
        record = {
            "index": 1,
            "annotation_id": "gt1",
            "evidence_need": "label",
            "gt_bbox": [100, 100, 200, 200],
            "status": "valid",
            "elements": [{"description": "label", "bbox": [90, 90, 210, 210]}],
        }
        row = evaluate_record(record, EvaluationConfig(semantic_policy="spatial"))
        self.assertTrue(row["profile_hit_0_5"])
        self.assertTrue(row["strict_iou_hit_0_5"])

    def test_large_context_box_does_not_pass_containment(self):
        record = {
            "index": 1,
            "annotation_id": "gt1",
            "evidence_need": "label",
            "gt_bbox": [100, 100, 200, 200],
            "status": "valid",
            "elements": [{"description": "context", "bbox": [0, 0, 1000, 1000]}],
        }
        row = evaluate_record(record, EvaluationConfig(semantic_policy="spatial"))
        self.assertFalse(row["profile_hit_0_3"])
        self.assertTrue(row["large_box_only"])

    def test_model_failure_counts_as_miss(self):
        record = {
            "index": 1,
            "annotation_id": "gt1",
            "evidence_need": "label",
            "gt_bbox": [100, 100, 200, 200],
            "status": "truncated",
            "elements": [],
            "elapsed_seconds": 2,
            "token_usage": {"prompt": 1, "completion": 2, "total": 3},
        }
        report = make_report([record])
        self.assertEqual(report["summary"]["profile_recall_0_3"], 0.0)
        self.assertEqual(report["summary"]["model_failure_rate"], 1.0)
        self.assertEqual(report["summary"]["token_usage"]["total"], 3)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from rig_eval.core import (
    BBox,
    EvaluationConfig,
    audit_annotations,
    evaluate_predictions,
    load_annotations,
    prepare_review,
)


def _element(element_id, bbox, description="clue"):
    return {"id": element_id, "bbox": BBox.from_value(bbox), "description": description}


class RigEvaluationTests(unittest.TestCase):
    def test_formal_annotation_file_has_41_samples_and_excludes_test_case(self):
        annotation_path = (
            Path(__file__).resolve().parents[2]
            / "RIG-Annotating-main"
            / "annotation of RIG"
            / "Charxiv_reasoning_val"
            / "annotation_1.json"
        )
        annotations = load_annotations(annotation_path)
        audit = audit_annotations(annotations)
        self.assertEqual(audit["n_samples"], 41)
        self.assertEqual(audit["n_elements"], 303)
        self.assertEqual(audit["n_overlap_samples"], 15)
        self.assertEqual(audit["n_overlap_pairs"], 53)

    def test_one_prediction_cannot_match_two_overlapping_gt_elements(self):
        annotations = {
            1: {
                "elements": [
                    _element("gt_a", [0, 0, 100, 100], "semantic clue A"),
                    _element("gt_b", [0, 0, 100, 100], "semantic clue B"),
                ]
            }
        }
        predictions = {
            1: {
                "elements": [_element("pred_000", [0, 0, 100, 100], "only one clue")],
                "answer_score": None,
            }
        }
        report = evaluate_predictions(
            annotations,
            predictions,
            EvaluationConfig(semantic_policy="spatial"),
        )
        profile = report["samples"][0]["profiles"]["match_at_0.3"]
        self.assertEqual(profile["n_matched"], 1)
        self.assertEqual(profile["necessary_recall"], 0.5)
        self.assertFalse(profile["complete_evidence"])

    def test_union_region_coverage_does_not_double_count_gt_overlap(self):
        annotations = {
            1: {
                "elements": [
                    _element("gt_a", [0, 0, 100, 100]),
                    _element("gt_b", [50, 0, 150, 100]),
                ]
            }
        }
        predictions = {
            1: {
                "elements": [_element("pred_000", [0, 0, 150, 100])],
                "answer_score": None,
            }
        }
        row = evaluate_predictions(
            annotations, predictions, EvaluationConfig(semantic_policy="spatial")
        )["samples"][0]
        self.assertEqual(row["gt_union_area"], 15000)
        self.assertEqual(row["pred_union_area"], 15000)
        self.assertEqual(row["union_region_coverage"], 1.0)
        self.assertEqual(row["profiles"]["match_at_0.3"]["necessary_recall"], 0.5)

    def test_overlap_review_stays_pending_until_every_candidate_is_decided(self):
        annotations = {
            1: {
                "elements": [
                    _element("gt_a", [0, 0, 100, 100], "A"),
                    _element("gt_b", [50, 0, 150, 100], "B"),
                ]
            }
        }
        predictions = {
            1: {
                "elements": [
                    _element("pred_000", [0, 0, 100, 100], "A"),
                    _element("pred_001", [50, 0, 150, 100], "B"),
                ],
                "answer_score": 1,
            }
        }
        config = EvaluationConfig(semantic_policy="overlap_review")
        with tempfile.TemporaryDirectory() as temp_dir:
            review_path = Path(temp_dir) / "review.json"
            payload = prepare_review(annotations, predictions, review_path, config)
            self.assertTrue(payload["decisions"])

            pending_report = evaluate_predictions(annotations, predictions, config, review_path)
            pending_profile = pending_report["samples"][0]["profiles"]["match_at_0.3"]
            self.assertFalse(pending_profile["finalized"])
            self.assertIsNone(pending_profile["necessary_recall"])

            for row in payload["decisions"]:
                row["decision"] = row["gt_description"] == row["pred_description"]
            review_path.write_text(json.dumps(payload), encoding="utf-8")

            final_report = evaluate_predictions(annotations, predictions, config, review_path)
            final_profile = final_report["samples"][0]["profiles"]["match_at_0.3"]
            self.assertTrue(final_profile["finalized"])
            self.assertEqual(final_profile["necessary_recall"], 1.0)
            self.assertTrue(final_profile["complete_evidence"])

    def test_non_overlap_samples_are_auto_matched_in_overlap_review_mode(self):
        annotations = {1: {"elements": [_element("gt_a", [0, 0, 100, 100])]}}
        predictions = {
            1: {
                "elements": [_element("pred_000", [0, 0, 100, 100])],
                "answer_score": None,
            }
        }
        profile = evaluate_predictions(
            annotations, predictions, EvaluationConfig(semantic_policy="overlap_review")
        )["samples"][0]["profiles"]["match_at_0.3"]
        self.assertTrue(profile["finalized"])
        self.assertEqual(profile["necessary_recall"], 1.0)

    def test_external_answer_scores_are_joined_by_index_including_bbox_failures(self):
        annotations = {
            10: {"elements": [_element("gt_a", [0, 0, 100, 100])]},
            20: {"elements": [_element("gt_b", [0, 0, 100, 100])]},
        }
        predictions = {
            10: {
                "elements": [_element("pred_000", [0, 0, 100, 100])],
                "answer_score": None,
            }
        }
        report = evaluate_predictions(
            annotations,
            predictions,
            EvaluationConfig(semantic_policy="spatial"),
            answer_scores={10: 1.0, 20: 1.0},
        )
        rows = {row["index"]: row for row in report["samples"]}
        self.assertEqual(rows[10]["answer_score"], 1.0)
        self.assertEqual(rows[20]["answer_score"], 1.0)
        self.assertFalse(rows[20]["has_prediction"])
        conditioned = report["summary"]["profiles"]["match_at_0.3"]["answer_conditioned"]
        self.assertEqual(conditioned["n_with_answer_score"], 2)
        self.assertEqual(conditioned["correct"]["n_samples"], 2)
        self.assertEqual(conditioned["correct"]["complete_evidence_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()

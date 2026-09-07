import json
import tempfile
import unittest
from pathlib import Path

from prepare_H2_Human_Image_Aware_Plan_at6 import (
    ATTESTATION_FIELDS,
    INDICES,
    VERSION,
    canonical_sha256,
    freeze_plan,
    validate_plan,
    verify_frozen,
)


def context_payload():
    return {
        "format": VERSION,
        "packet_type": "blind-human-authoring-context",
        "policy": {},
        "samples": [
            {
                "index": index,
                "question": f"question {index}",
                "question_source": "raw_question",
                "packet_image": f"images/q{index:04d}.jpg",
                "source_image_sha256": f"hash-{index}",
                "packet_image_sha256": f"hash-{index}",
            }
            for index in INDICES
        ],
    }


def plan_payload(context):
    return {
        "format": VERSION,
        "condition": "human-image-aware-plan@6",
        "context_sha256": canonical_sha256(context),
        "author": {
            "anonymous_id": "reviewer-A",
            "attestation": {field: True for field in ATTESTATION_FIELDS},
        },
        "plans": [
            {
                "index": index,
                "question": f"question {index}",
                "steps": [
                    {
                        "step_id": f"s{position}",
                        "evidence_need": f"Visible evidence group {position} for sample {index}",
                    }
                    for position in range(1, 7)
                ],
                "author_notes": "",
            }
            for index in INDICES
        ],
    }


class HumanImageAwarePlanTests(unittest.TestCase):
    def write_inputs(self, directory: Path):
        context = context_payload()
        plan = plan_payload(context)
        context_path = directory / "authoring_context.json"
        plan_path = directory / "human_plans.json"
        context_path.write_text(json.dumps(context), encoding="utf-8")
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        return plan_path, context_path

    def test_valid_complete_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan_path, context_path = self.write_inputs(Path(temporary))
            result = validate_plan(plan_path, context_path)
            self.assertTrue(result["valid"])
            self.assertEqual(result["warnings"], [])
            self.assertEqual(len(result["normalized_plans"]), 5)

    def test_requires_truthful_access_attestation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan_path, context_path = self.write_inputs(directory)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["author"]["attestation"]["did_not_view_gt_annotation"] = False
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            result = validate_plan(plan_path, context_path)
            self.assertFalse(result["valid"])
            self.assertTrue(any("did_not_view_gt_annotation" in error for error in result["errors"]))

    def test_rejects_duplicate_needs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan_path, context_path = self.write_inputs(directory)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["plans"][0]["steps"][1]["evidence_need"] = plan["plans"][0]["steps"][0]["evidence_need"]
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            result = validate_plan(plan_path, context_path)
            self.assertFalse(result["valid"])
            self.assertTrue(any("duplicate" in error for error in result["errors"]))

    def test_warns_on_over_broad_panel_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan_path, context_path = self.write_inputs(directory)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["plans"][0]["steps"][0]["evidence_need"] = "The entire chart and all of its contents"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            result = validate_plan(plan_path, context_path)
            self.assertTrue(result["valid"])
            self.assertTrue(any("over-broad" in warning for warning in result["warnings"]))

    def test_rejects_answer_derived_ranking_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan_path, context_path = self.write_inputs(directory)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["plans"][0]["steps"][0]["evidence_need"] = "Point of the model with the second highest time"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            result = validate_plan(plan_path, context_path)
            self.assertFalse(result["valid"])
            self.assertTrue(any("with the second highest" in error for error in result["errors"]))

    def test_warns_on_unmarked_derived_coordinate(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan_path, context_path = self.write_inputs(directory)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["plans"][0]["steps"][0]["evidence_need"] = "The coordinate (0.75, -0.25) on the plot"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            result = validate_plan(plan_path, context_path)
            self.assertTrue(result["valid"])
            self.assertTrue(any("target coordinate is derived" in warning for warning in result["warnings"]))

    def test_freeze_and_detect_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source"
            source.mkdir()
            plan_path, context_path = self.write_inputs(source)
            frozen = directory / "frozen"
            freeze_plan(plan_path, context_path, frozen)
            self.assertTrue(verify_frozen(frozen)["valid"])
            plans = json.loads((frozen / "plans.json").read_text(encoding="utf-8"))
            plans[0]["steps"][0]["evidence_need"] = "tampered"
            (frozen / "plans.json").write_text(json.dumps(plans), encoding="utf-8")
            self.assertFalse(verify_frozen(frozen)["valid"])


if __name__ == "__main__":
    unittest.main()

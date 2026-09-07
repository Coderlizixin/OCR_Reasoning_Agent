import json
import tempfile
import unittest
from pathlib import Path

from rig_eval import load_predictions
from test_H2_Multistep_Grounding_on_Charxiv import (
    H2_VERSION,
    PLAN_USER_TEMPLATE,
    STEP_JSON_SCHEMA,
    STEP_SYSTEM_PROMPT,
    build_protocol_spec,
    canonical_json_sha256,
    export_review_template,
    freeze_protocol,
    freeze_run_directory,
    load_predicted_plans,
    load_repeat_counts,
    normalize_token_usage,
    load_reviewed_plans,
    parse_plan_response,
    rectangle_union_area,
    resolve_api_base,
    sha256_file,
    union_step_elements,
    verify_frozen_directory,
)


class H2MultistepGroundingTests(unittest.TestCase):
    def test_model_specific_api_base(self):
        self.assertTrue(resolve_api_base("Qwen3.5-9B", None).endswith(":22630/v1"))
        self.assertTrue(resolve_api_base("Qwen3.5-27B", None).endswith(":23290/v1"))
        self.assertEqual(resolve_api_base("custom", "http://server/v1"), "http://server/v1")

    def test_repeat_counts_are_read_from_predicted_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plans.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "index": 7,
                            "steps": [
                                {"step_id": "s1", "evidence_need": "legend identity"},
                                {"step_id": "s2", "evidence_need": "axis scale"},
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(load_repeat_counts(path), {7: 2})

    def test_failed_plan_is_preserved_as_zero_call_control(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plans.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "index": 7,
                            "question": "q",
                            "status": "truncated",
                            "steps": [],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            plans = load_predicted_plans(path, require_frozen=False)
            self.assertEqual(plans[7]["steps"], [])
            self.assertEqual(load_repeat_counts(path), {7: 0})

    def test_missing_total_tokens_is_reconstructed(self):
        self.assertEqual(
            normalize_token_usage({"prompt": 10, "completion": 7}),
            {"prompt": 10, "completion": 7, "total": 17},
        )

    def test_plan_parser_accepts_valid_guided_json(self):
        steps = parse_plan_response(
            '{"steps":[{"step_id":"s1","evidence_need":"Locate the scale."},'
            '{"step_id":"s2","evidence_need":"Locate candidate marks."}]}'
        )
        self.assertEqual([step["step_id"] for step in steps], ["s1", "s2"])

    def test_plan_parser_recovers_after_unmatched_brace_in_thinking(self):
        steps = parse_plan_response(
            'thinking mentions a stray } before output\n'
            '{"steps":[{"step_id":"s1","evidence_need":"Locate the scale."},'
            '{"step_id":"s2","evidence_need":"Locate candidate marks."}]}'
        )
        self.assertEqual([step["step_id"] for step in steps], ["s1", "s2"])

    def test_union_removes_only_exact_semantic_duplicates_and_keeps_provenance(self):
        records = [
            {
                "step_id": "s1",
                "call_id": "a",
                "elements": [{"description": "axis", "bbox": [0, 0, 100, 100]}],
            },
            {
                "step_id": "s2",
                "call_id": "b",
                "elements": [
                    {"description": "axis", "bbox": [0, 0, 100, 100]},
                    {"description": "different clue", "bbox": [0, 0, 100, 100]},
                ],
            },
        ]
        elements, stats = union_step_elements(records)
        self.assertEqual(len(elements), 2)
        self.assertEqual(stats["exact_duplicates_removed"], 1)
        self.assertEqual(len(elements[0]["provenance"]), 2)

    def test_rectangle_union_area_does_not_double_count_overlap(self):
        self.assertAlmostEqual(rectangle_union_area([[0, 0, 100, 100], [50, 0, 150, 100]]), 0.015)

    def test_reviewed_plan_rejects_pending_and_accepts_approved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reviewed.json"
            payload = {
                "plans": [
                    {
                        "index": 1,
                        "review_status": "pending",
                        "steps": [
                            {"step_id": "s1", "evidence_need": "Locate scale."},
                            {"step_id": "s2", "evidence_need": "Locate marks."},
                        ],
                    }
                ]
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "review_status='approved'"):
                load_reviewed_plans(path)
            payload["plans"][0]["review_status"] = "approved"
            payload["plans"][0]["reviewer"] = "tester"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIn(1, load_reviewed_plans(path))

    def test_reviewed_plan_must_preserve_source_step_structure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "plans.json"
            source_rows = [
                {
                    "index": 1,
                    "question": "q",
                    "status": "valid",
                    "steps": [
                        {"step_id": "s1", "evidence_need": "scale"},
                        {"step_id": "s2", "evidence_need": "marks"},
                    ],
                }
            ]
            source.write_text(json.dumps(source_rows), encoding="utf-8")
            predicted = load_predicted_plans(source, require_frozen=False)
            reviewed = root / "reviewed.json"
            payload = {
                "source_sha256": sha256_file(source),
                "plans": [
                    {
                        "index": 1,
                        "question": "q",
                        "review_status": "approved",
                        "reviewer": "tester",
                        "steps": [
                            {"step_id": "s1", "evidence_need": "tight scale"},
                            {"step_id": "changed", "evidence_need": "tight marks"},
                        ],
                    }
                ],
            }
            reviewed.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed step count"):
                load_reviewed_plans(
                    reviewed,
                    predicted_plan_path=source,
                    predicted_plans=predicted,
                )

    def test_v2_protocol_caps_each_step_at_sixteen_elements(self):
        self.assertEqual(STEP_JSON_SCHEMA["properties"]["elements"]["maxItems"], 16)
        self.assertEqual(build_protocol_spec()["protocol_version"], "h2-grounding-v2-concise-complete")
        self.assertIn("not estimated centroids", PLAN_USER_TEMPLATE)
        self.assertIn("do not request bbox numbers", PLAN_USER_TEMPLATE)
        self.assertIn("do not assume an unmentioned visual encoding", PLAN_USER_TEMPLATE)
        self.assertIn("internal visual reasoning", STEP_SYSTEM_PROMPT)

    def test_run_freeze_rejects_incomplete_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "run"
            root.mkdir()
            status_path = root / "run_status.json"
            status = {
                "experiment_version": H2_VERSION,
                "mode": "predicted-plan",
                "requested_indices": [1],
                "completed_indices": [],
                "request_failed_indices": [],
                "request_failed_samples": 0,
                "complete": False,
                "protocol_complete": False,
            }
            status_path.write_text(json.dumps(status), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                freeze_run_directory(root, reason="test")
            status.update(
                {"completed_indices": [1], "complete": True, "protocol_complete": True}
            )
            status_path.write_text(json.dumps(status), encoding="utf-8")
            protocol_dir = base / "protocol"
            freeze_protocol(protocol_dir)
            (root / "experiment_config.json").write_text(
                json.dumps(
                    {
                        "experiment_version": H2_VERSION,
                        "protocol_sha256": canonical_json_sha256(build_protocol_spec()),
                        "protocol_frozen": True,
                        "protocol_directory": str(protocol_dir),
                        "run_scope": "formal",
                        "annotation_sha256": build_protocol_spec()["formal_dataset"][
                            "annotation_sha256"
                        ],
                    }
                ),
                encoding="utf-8",
            )
            freeze_run_directory(root, reason="test")
            self.assertTrue(verify_frozen_directory(root)["valid"])
            status_path.write_text("{}", encoding="utf-8")
            self.assertFalse(verify_frozen_directory(root)["valid"])

    def test_exported_union_is_compatible_with_existing_evaluator_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rig_results.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "index": 1,
                            "question": "q",
                            "inst_category": 1,
                            "elements": [
                                {
                                    "id": "pred_000",
                                    "description": "axis",
                                    "bbox": [0, 0, 100, 100],
                                    "provenance": [{"step_id": "s1"}],
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            loaded = load_predictions(path)
            self.assertEqual(loaded[1]["elements"][0]["id"], "pred_000")

    def test_review_template_is_pending_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "plans.json"
            output = Path(temporary) / "reviewed.template.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "index": 1,
                            "question": "q",
                            "steps": [
                                {"step_id": "s1", "evidence_need": "Locate scale."},
                                {"step_id": "s2", "evidence_need": "Locate marks."},
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            export_review_template(source, output)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["plans"][0]["review_status"], "pending")

    def test_review_template_does_not_repair_failed_plans(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "plans.json"
            output = Path(temporary) / "reviewed.template.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "index": 1,
                            "question": "q",
                            "status": "parse_failed",
                            "steps": [],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            export_review_template(source, output)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["plans"][0]["review_status"], "source_invalid")
            reviewed = load_reviewed_plans(
                output,
                predicted_plan_path=source,
                predicted_plans=load_predicted_plans(source, require_frozen=False),
            )
            self.assertEqual(reviewed[1]["steps"], [])


if __name__ == "__main__":
    unittest.main()

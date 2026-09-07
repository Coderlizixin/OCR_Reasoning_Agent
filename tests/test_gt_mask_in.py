import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from test_GT_MaskIn_on_Charxiv import (
    EXPERIMENT_VERSION,
    FREEZE_MARKER,
    NO_VALID_ANSWER_FAILED,
    NO_VALID_ANSWER_TRUNCATED,
    apply_condition_mask,
    assert_run_directory_writable,
    build_messages,
    build_roi_inputs,
    build_gt_union_mask,
    build_prediction_dataframe,
    extract_judge_prediction,
    freeze_run_directory,
    load_annotations,
    normalized_box_to_pixels,
    resolve_served_model_name,
    run_evidence_condition,
    run_gt_mask_in,
    shifted_random_control_mask,
    summarize_answer_quality,
    validate_prepared_outputs,
    verify_frozen_directory,
)
from vlmeval.dataset.charxiv import auxeval, build_pending_tasks


class GTMaskInUnitTests(unittest.TestCase):
    def test_charxiv_auxeval_accepts_nan_prediction_as_no_answer(self):
        class Judge:
            def generate(self, prompt, **kwargs):
                self.prompt = prompt
                return '{"extract_answer": "", "score": 0}'

        judge = Judge()
        result = auxeval(
            judge,
            pd.Series({"grading_query": "Prediction: {PREDICTION}", "prediction": float("nan")}),
            retry=1,
        )
        self.assertEqual(judge.prompt, "Prediction: ")
        self.assertEqual(result["score"], 0)

    def test_partial_charxiv_judge_cache_builds_aligned_pending_tasks(self):
        data = pd.DataFrame(
            [{"index": 10}, {"index": 20}, {"index": 30}, {"index": 40}]
        )
        keys, tasks = build_pending_tasks("judge", data, {0: {"score": 1}, 2: {"score": 0}})
        self.assertEqual(keys, [1, 3])
        self.assertEqual([int(task[1]["index"]) for task in tasks], [20, 40])
        self.assertEqual(len(keys), len(tasks))

    def test_judge_prediction_removes_completed_thinking(self):
        quality = extract_judge_prediction(
            {
                "status": "success",
                "finish_reason": "stop",
                "prediction": "private reasoning</think>\n\nfinal answer",
            }
        )
        self.assertEqual(quality["answer_status"], "valid")
        self.assertEqual(quality["judge_prediction"], "final answer")
        self.assertEqual(quality["model_raw_output"], "private reasoning</think>\n\nfinal answer")

    def test_token_limited_response_is_invalid_even_if_it_contains_a_think_end(self):
        quality = extract_judge_prediction(
            {
                "status": "success",
                "finish_reason": "length",
                "prediction": "reasoning</think>partial final response",
            }
        )
        self.assertEqual(quality["answer_status"], "truncated")
        self.assertEqual(quality["judge_prediction"], NO_VALID_ANSWER_TRUNCATED)

    def test_quality_summary_separates_http_completion_from_valid_answers(self):
        report = summarize_answer_quality(
            [
                {"index": 1, "status": "success", "finish_reason": "stop", "prediction": "a"},
                {"index": 2, "status": "success", "finish_reason": "length", "prediction": "b"},
                {"index": 3, "status": "failed", "prediction": ""},
            ],
            [1, 2, 3, 4],
        )
        self.assertEqual(report["http_completed_samples"], 2)
        self.assertEqual(report["valid_answer_indices"], [1])
        self.assertEqual(report["truncated_indices"], [2])
        self.assertEqual(report["request_failed_indices"], [3])
        self.assertEqual(report["not_run_indices"], [4])
        self.assertFalse(report["protocol_complete"])

    def test_api_alias_is_mapped_to_the_server_model_id(self):
        self.assertEqual(resolve_served_model_name("Qwen3.5-9B-API"), "Qwen3.5-9B")
        self.assertEqual(resolve_served_model_name("custom-model"), "custom-model")

    def test_normalized_box_uses_floor_for_start_and_ceil_for_end(self):
        self.assertEqual(
            normalized_box_to_pixels([108, 870, 971, 933], (100, 200)),
            (10, 174, 98, 187),
        )

    def test_overlapping_boxes_form_one_union_mask(self):
        image = Image.new("RGB", (10, 10), color=(10, 20, 30))
        annotations = [
            {"bbox": [0, 0, 600, 600]},
            {"bbox": [400, 400, 1000, 1000]},
        ]

        masked, mask, stats = build_gt_union_mask(
            image, annotations, background=(127, 127, 127)
        )

        self.assertEqual(stats["summed_box_pixels"], 72)
        self.assertEqual(stats["overlap_pixels"], 4)
        self.assertEqual(stats["visible_pixels"], 68)
        self.assertAlmostEqual(stats["visible_ratio"], 0.68)
        self.assertEqual(mask.getpixel((4, 4)), 255)
        self.assertEqual(masked.getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(masked.getpixel((9, 0)), (127, 127, 127))

    def test_gt_mask_out_removes_only_evidence_pixels(self):
        image = Image.new("RGB", (4, 4), color=(10, 20, 30))
        mask = Image.new("L", (4, 4), color=0)
        mask.putpixel((1, 1), 255)
        output = apply_condition_mask(
            image, mask, condition="gt-mask-out", background=(127, 127, 127)
        )
        self.assertEqual(output.getpixel((1, 1)), (127, 127, 127))
        self.assertEqual(output.getpixel((0, 0)), (10, 20, 30))

    def test_random_control_is_deterministic_and_exactly_area_matched(self):
        mask = Image.new("L", (12, 10), color=0)
        for x in range(2, 6):
            for y in range(3, 8):
                mask.putpixel((x, y), 255)
        first, first_stats = shifted_random_control_mask(mask, seed=123, candidates=32)
        second, second_stats = shifted_random_control_mask(mask, seed=123, candidates=32)
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(first_stats, second_stats)
        self.assertEqual(first.histogram()[255], mask.histogram()[255])
        self.assertTrue(first_stats["area_preserved_exactly"])

    def test_roi_inputs_put_masked_overview_before_ordered_crops(self):
        image = Image.new("RGB", (10, 10), color=(9, 8, 7))
        annotations = [
            {"id": "a", "bbox": [0, 0, 500, 500]},
            {"id": "b", "bbox": [500, 500, 1000, 1000]},
        ]
        images, _, stats = build_roi_inputs(
            image, annotations, padding=0, background=(127, 127, 127)
        )
        self.assertEqual(len(images), 3)
        self.assertEqual(images[0].size, (10, 10))
        self.assertEqual(images[1].size, (5, 5))
        self.assertEqual(stats["model_image_count"], 3)
        self.assertEqual(
            [item["annotation_id"] for item in stats["crop_metadata"]], ["a", "b"]
        )

    def test_multimodal_message_orders_all_images_before_question(self):
        messages = build_messages("question", ["data:image/png;base64,a", "data:image/png;base64,b"])
        content = messages[0]["content"]
        self.assertEqual([item["type"] for item in content], ["image_url", "image_url", "text"])
        self.assertEqual(content[-1]["text"], "question")

    def test_frozen_directory_rejects_runner_writes_and_verifies_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            (output_dir / "result.txt").write_text("fixed", encoding="utf-8")
            (output_dir / "run_status.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "protocol_complete": True,
                        "requested_indices": [1],
                        "completed_indices": [1],
                        "request_failed_samples": 0,
                        "request_failed_indices": [],
                    }
                ),
                encoding="utf-8",
            )
            freeze_run_directory(output_dir, reason="unit test")
            with self.assertRaises(PermissionError):
                assert_run_directory_writable(output_dir, operation="overwrite")
            self.assertTrue(verify_frozen_directory(output_dir)["valid"])

    def test_freeze_rejects_incomplete_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            (output_dir / "run_status.json").write_text(
                json.dumps(
                    {
                        "complete": False,
                        "requested_indices": [1, 2],
                        "completed_indices": [1],
                        "request_failed_samples": 1,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "incomplete run"):
                freeze_run_directory(output_dir, reason="must fail")
            self.assertFalse((output_dir / FREEZE_MARKER).exists())

    def test_annotation_loader_keeps_semantic_duplicates_as_separate_boxes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "annotation.json"
            path.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "path": "images/41.jpg",
                                "annotations": [
                                    {
                                        "id": "a",
                                        "description": "first clue",
                                        "x1": 10,
                                        "y1": 20,
                                        "x2": 30,
                                        "y2": 40,
                                    },
                                    {
                                        "id": "b",
                                        "description": "second clue",
                                        "x1": 10,
                                        "y1": 20,
                                        "x2": 30,
                                        "y2": 40,
                                    },
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            loaded = load_annotations(path)

        self.assertEqual(list(loaded), [41])
        self.assertEqual(len(loaded[41]["annotations"]), 2)
        self.assertEqual(loaded[41]["annotations"][0]["bbox"], [10.0, 20.0, 30.0, 40.0])

    def test_prediction_workbook_preserves_requested_order_and_drops_base64(self):
        source = pd.DataFrame(
            [
                {"index": 1, "question": "q1", "grading_query": "g1", "image": "base64"},
                {"index": 2, "question": "q2", "grading_query": "g2", "image": "base64"},
            ]
        )
        results = [
            {
                "index": 2,
                "status": "success",
                "prediction": "answer two",
                "masked_image_path": "two.png",
                "mask_statistics": {"visible_ratio": 0.2},
            },
            {
                "index": 1,
                "status": "failed",
                "prediction": "",
                "mask_statistics": {"visible_ratio": 0.1},
            },
        ]

        exported = build_prediction_dataframe(source, [2, 1], results)

        self.assertEqual(exported["index"].tolist(), [2, 1])
        self.assertEqual(
            exported["prediction"].tolist(), ["answer two", NO_VALID_ANSWER_FAILED]
        )
        self.assertEqual(
            exported["gt_mask_answer_status"].tolist(), ["valid", "request_failed"]
        )
        self.assertNotIn("image", exported.columns)
        self.assertIn("grading_query", exported.columns)


class GTMaskInIntegrationTests(unittest.TestCase):
    def test_prepare_only_writes_exact_mask_and_auditable_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "7.png"
            Image.new("RGB", (10, 10), color=(9, 8, 7)).save(image_path)
            annotations_path = root / "annotation.json"
            annotations_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "path": str(image_path),
                                "annotations": [
                                    {
                                        "id": "x",
                                        "description": "clue",
                                        "x1": 0,
                                        "y1": 0,
                                        "x2": 500,
                                        "y2": 500,
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            tsv_path = root / "data.tsv"
            pd.DataFrame(
                [
                    {
                        "index": 7,
                        "inst_category": 1,
                        "raw_question": "raw question",
                        "question": "dataset question",
                        "answer": "answer",
                        "grading_query": "grade {PREDICTION}",
                    }
                ]
            ).to_csv(tsv_path, sep="\t", index=False)
            output_dir = root / "output"

            results = run_gt_mask_in(
                annotations_path=annotations_path,
                tsv_path=tsv_path,
                image_dir=root,
                indices=[7],
                model_name="unit-test-model",
                output_dir=output_dir,
                prepare_only=True,
            )

            self.assertEqual(results[0]["status"], "prepared")
            self.assertEqual(results[0]["question"], "dataset question")
            self.assertEqual(results[0]["experiment_version"], EXPERIMENT_VERSION)
            self.assertTrue((output_dir / "predictions.json").is_file())
            self.assertTrue((output_dir / "predictions.xlsx").is_file())
            self.assertTrue((output_dir / "binary_masks/q0007_union_mask.png").is_file())
            masked_path = output_dir / "masked_images/q0007_gt_mask_in.png"
            self.assertTrue(masked_path.is_file())
            with Image.open(masked_path) as masked:
                self.assertEqual(masked.getpixel((0, 0)), (9, 8, 7))
                self.assertEqual(masked.getpixel((9, 9)), (127, 127, 127))

    def test_prepare_only_supports_all_new_conditions_and_audits_input_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "9.png"
            Image.new("RGB", (20, 10), color=(30, 40, 50)).save(image_path)
            annotations_path = root / "annotation.json"
            annotations_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "path": str(image_path),
                                "annotations": [
                                    {"id": "a", "x1": 0, "y1": 0, "x2": 250, "y2": 500},
                                    {"id": "b", "x1": 500, "y1": 500, "x2": 1000, "y2": 1000},
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            tsv_path = root / "data.tsv"
            pd.DataFrame(
                [
                    {
                        "index": 9,
                        "inst_category": 1,
                        "question": "dataset question",
                        "answer": "answer",
                        "grading_query": "grade {PREDICTION}",
                    }
                ]
            ).to_csv(tsv_path, sep="\t", index=False)

            for condition in ("full-image", "gt-mask-out", "random-mask-in", "gt-roi"):
                output_dir = root / condition
                results = run_evidence_condition(
                    annotations_path=annotations_path,
                    tsv_path=tsv_path,
                    image_dir=root,
                    indices=[9],
                    model_name="unit-test-model",
                    condition=condition,
                    output_dir=output_dir,
                    prepare_only=True,
                    random_seed=7,
                )
                self.assertEqual(results[0]["condition"], condition)
                self.assertTrue(Path(results[0]["condition_image_path"]).is_file())
                manifest = json.loads(
                    (output_dir / "model_input_manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(manifest[0]["content"][-1]["type"], "text")
                self.assertEqual(
                    manifest[0]["model_input_image_count"],
                    results[0]["model_input_image_count"],
                )
                if condition == "gt-roi":
                    self.assertEqual(results[0]["model_input_image_count"], 3)
                    self.assertEqual(len(results[0]["roi_image_paths"]), 2)
                else:
                    self.assertEqual(results[0]["model_input_image_count"], 1)
                self.assertTrue(validate_prepared_outputs(output_dir)["valid"])


if __name__ == "__main__":
    unittest.main()

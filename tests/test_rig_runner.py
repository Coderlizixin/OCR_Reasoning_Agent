import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from test_RIG_on_Charxiv import (
    PROMPT_VERSION,
    RIG_JSON_SCHEMA,
    RIG_SYSTEM_PROMPT,
    RIG_USER_PROMPT_TEMPLATE,
    build_rig_extra_body,
    build_rig_messages,
    get_rig_question,
    model_run_dir,
    run_rig_test,
)


class RigRunnerIsolationTests(unittest.TestCase):
    def test_default_run_directory_is_model_specific(self):
        self.assertEqual(
            model_run_dir("Qwen3.5-9B"),
            Path(f"test_case/RIG_Charxiv/runs/Qwen3.5-9B__{PROMPT_VERSION}"),
        )
        self.assertNotEqual(model_run_dir("Qwen3.5-9B"), model_run_dir("Qwen3.5-27B"))

    def test_non_empty_run_directory_is_protected_before_api_or_data_loading(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "existing.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_rig_test(
                    indices=[1],
                    model_name="Qwen3.5-9B",
                    output_dir=output_dir,
                )

    def test_resume_and_overwrite_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                run_rig_test(
                    indices=[1],
                    model_name="Qwen3.5-9B",
                    output_dir=Path(temp_dir),
                    resume=True,
                    overwrite_existing=True,
                )

    def test_resume_rejects_legacy_prompt_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "rig_results.json").write_text(
                json.dumps([{"index": 1, "model_name": "Qwen3.5-9B"}]),
                encoding="utf-8",
            )
            (output_dir / "run_status.json").write_text(
                json.dumps({"model_name": "Qwen3.5-9B"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "prompt versions"):
                run_rig_test(
                    indices=[1],
                    model_name="Qwen3.5-9B",
                    output_dir=output_dir,
                    resume=True,
                )

    def test_prompt_v4_is_concise_and_comparison_complete(self):
        prompt = f"{RIG_SYSTEM_PROMPT}\n{RIG_USER_PROMPT_TEMPLATE}"
        self.assertIn("Do not solve the question", RIG_SYSTEM_PROMPT)
        self.assertIn("Do not ground only the final answer", RIG_USER_PROMPT_TEMPLATE)
        self.assertIn("every compared candidate", RIG_USER_PROMPT_TEMPLATE)
        self.assertIn("instead of debating", RIG_USER_PROMPT_TEMPLATE)
        self.assertNotIn("removing it would make the answer impossible", prompt)
        self.assertNotIn("smallest complete set", prompt)
        self.assertLess(len(prompt.split()), 250)

    def test_clean_raw_question_is_preferred_over_dataset_answer_instructions(self):
        question, field = get_rig_question(pd.Series({
            "index": 1,
            "raw_question": "  Which line is furthest?  ",
            "question": "Which line is furthest? Final answer must be grounded to text.",
        }))
        self.assertEqual(question, "Which line is furthest?")
        self.assertEqual(field, "raw_question")

    def test_question_fallback_is_preserved_for_datasets_without_raw_question(self):
        question, field = get_rig_question(pd.Series({
            "index": 1,
            "raw_question": None,
            "question": "Fallback question",
        }))
        self.assertEqual(question, "Fallback question")
        self.assertEqual(field, "question")

    def test_guided_json_schema_requires_nonempty_grounded_elements(self):
        self.assertEqual(RIG_JSON_SCHEMA["required"], ["elements"])
        elements = RIG_JSON_SCHEMA["properties"]["elements"]
        self.assertEqual(elements["minItems"], 1)
        bbox = elements["items"]["properties"]["bbox"]
        self.assertEqual((bbox["minItems"], bbox["maxItems"]), (4, 4))
        self.assertEqual(bbox["items"]["minimum"], 0)
        self.assertEqual(bbox["items"]["maximum"], 1000)

    def test_request_keeps_thinking_and_temperature_unspecified(self):
        extra_body = build_rig_extra_body()
        self.assertEqual(extra_body["guided_json"], RIG_JSON_SCHEMA)
        self.assertNotIn("enable_thinking", extra_body)
        self.assertNotIn("chat_template_kwargs", extra_body)
        self.assertNotIn("temperature", extra_body)
        messages = build_rig_messages("question", "data:image/jpeg;base64,test")
        self.assertEqual([message["role"] for message in messages], ["system", "user"])


if __name__ == "__main__":
    unittest.main()

"""Static question-only plan@6 feasibility runner for the H2 adaptive gate.

This independent engineering experiment does not modify the frozen H2-v2 runner
or any repeated-single output. A question-only planner must produce exactly six
evidence needs; the same H2 step grounder executes those fixed needs. Final and
per-prefix (k=1..6) evidence unions are evaluator-compatible.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import httpx
import pandas as pd
from openai import OpenAI

from test_H2_Multistep_Grounding_on_Charxiv import (
    CATEGORY_NAMES,
    DEFAULT_ANNOTATION_PATH,
    DEFAULT_IMAGE_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_SMOKE_INDICES,
    DEFAULT_TSV_PATH,
    STEP_JSON_SCHEMA,
    STEP_SYSTEM_PROMPT,
    STEP_USER_TEMPLATE,
    call_model,
    canonical_json_sha256,
    encode_image,
    extract_json_objects,
    load_image,
    load_question,
    resolve_api_base,
    response_metadata,
    save_json_atomic,
    sha256_file,
    source_image_sha256,
    sum_tokens,
    union_step_elements,
    utc_now,
)
from test_RIG_on_Charxiv import build_rig_extra_body, draw_bboxes_on_image, parse_rig_response


SCRIPT_DIR = Path(__file__).resolve().parent
GATE_VERSION = "h2-static-question-plan-at6-v1"
K = 6
PLAN_FREEZE_MARKER = "PLAN_FROZEN.json"
DEFAULT_OUTPUT_ROOT = (
    SCRIPT_DIR / "test_case/RIG_Charxiv/h2_grounding/feasibility/adaptive-gate"
)

PLAN_SYSTEM_PROMPT = """You create a static evidence-grounding plan for a visual reasoning task.
You receive only the question and never see the image. Do not solve the question,
predict the answer, state candidate values, or invent image content. Return exactly
six complementary visible-evidence needs as the requested JSON object, without
markdown or exposed analysis."""

PLAN_USER_TEMPLATE = """QUESTION:
__QUESTION_PLACEHOLDER__

Create exactly 6 ordered evidence-grounding steps for a downstream visual grounder.
The downstream grounder will see the image, but you cannot see it now.

Each step must request one coherent, bbox-localizable class of directly visible
evidence needed to derive the answer. Across all six steps, cover the applicable
roles among: scope or panel identity, every candidate or operand, identity or
legend mapping, visible values/marks/traces, axes/scales/units/reference criteria,
and remaining structural or spatial relations. If a role is inapplicable, use that
step for another distinct evidence group; never add filler or duplicate a prior need.

Because the number of visible candidates is unknown, use set-valued requests such
as "every candidate relevant to ..." rather than guessing a count. Do not assume an
unmentioned encoding such as a table, row, column, legend, curve, or marker.

Do not create a step that compares, ranks, counts, calculates, concludes, or selects
the answer. Request the visible inputs needed for those operations instead. Do not
request canvas coordinates, bbox numbers, image dimensions, estimated centroids,
derived values, or other non-visible results. Do not include the final answer.

OUTPUT ONLY:
{
  "steps": [
    {"step_id": "s1", "evidence_need": "one precise visible-evidence requirement"},
    {"step_id": "s2", "evidence_need": "one different visible-evidence requirement"},
    {"step_id": "s3", "evidence_need": "one different visible-evidence requirement"},
    {"step_id": "s4", "evidence_need": "one different visible-evidence requirement"},
    {"step_id": "s5", "evidence_need": "one different visible-evidence requirement"},
    {"step_id": "s6", "evidence_need": "one different visible-evidence requirement"}
  ]
}"""

PLAN_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "minItems": K,
            "maxItems": K,
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "string", "minLength": 1, "maxLength": 24},
                    "evidence_need": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 320,
                    },
                },
                "required": ["step_id", "evidence_need"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}


def normalize_fixed_plan(raw: Any) -> Optional[List[Dict[str, str]]]:
    if not isinstance(raw, list) or len(raw) != K:
        return None
    steps: List[Dict[str, str]] = []
    for position, item in enumerate(raw, start=1):
        if not isinstance(item, dict) or set(item) != {"step_id", "evidence_need"}:
            return None
        step_id = str(item["step_id"]).strip()
        need = str(item["evidence_need"]).strip()
        if step_id != f"s{position}" or not need:
            return None
        steps.append({"step_id": step_id, "evidence_need": need[:320]})
    if len({step["evidence_need"].casefold() for step in steps}) != K:
        return None
    return steps


def parse_fixed_plan(text: str) -> Optional[List[Dict[str, str]]]:
    for candidate in reversed(extract_json_objects(text or "")):
        steps = normalize_fixed_plan(candidate.get("steps"))
        if steps:
            return steps
    return None


def plan_messages(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": PLAN_USER_TEMPLATE.replace("__QUESTION_PLACEHOLDER__", question),
        },
    ]


def step_messages(question: str, evidence_need: str, image_uri: str) -> List[Dict[str, Any]]:
    prompt = STEP_USER_TEMPLATE.replace("__QUESTION_PLACEHOLDER__", question).replace(
        "__EVIDENCE_NEED_PLACEHOLDER__", evidence_need
    )
    return [
        {"role": "system", "content": STEP_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_uri}},
            ],
        },
    ]


def step_extra_body() -> Dict[str, Any]:
    """Match repeated-single image preprocessing while changing only the JSON schema."""
    body = build_rig_extra_body()
    body["guided_json"] = STEP_JSON_SCHEMA
    return body


def protocol_spec() -> Dict[str, Any]:
    return {
        "version": GATE_VERSION,
        "budget_k": K,
        "planner_image_access": False,
        "planner_exact_steps": K,
        "planner_prompt": {
            "system": PLAN_SYSTEM_PROMPT,
            "user": PLAN_USER_TEMPLATE,
            "schema": PLAN_JSON_SCHEMA,
        },
        "grounder_prompt": {
            "system": STEP_SYSTEM_PROMPT,
            "user": STEP_USER_TEMPLATE,
            "schema": STEP_JSON_SCHEMA,
            "image_preprocessing": "same mm_processor_kwargs as repeated-single",
        },
        "thinking_disabled": False,
        "temperature_explicitly_set": False,
        "failure_policy": {
            "request_failed": "resume allowed",
            "truncated_or_parse_failed": "model failure; no resampling",
        },
        "prefixes": list(range(1, K + 1)),
    }


def file_manifest(directory: Path, marker_name: str) -> List[Dict[str, Any]]:
    return [
        {
            "path": path.relative_to(directory).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != marker_name
    ]


def freeze_plan_directory(directory: Path) -> Dict[str, Any]:
    directory = Path(directory)
    marker = directory / PLAN_FREEZE_MARKER
    if marker.exists():
        raise ValueError(f"plan directory is already frozen: {marker}")
    status = json.loads((directory / "run_status.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "experiment_config.json").read_text(encoding="utf-8"))
    plans = json.loads((directory / "plans.json").read_text(encoding="utf-8"))
    if not status.get("complete") or status.get("request_failed_samples"):
        raise ValueError("refusing to freeze an incomplete plan-only directory")
    if config.get("experiment_version") != GATE_VERSION or not config.get("plan_only"):
        raise ValueError("directory is not a compatible static plan@6 plan-only run")
    if config.get("protocol_sha256") != canonical_json_sha256(protocol_spec()):
        raise ValueError("plan-only protocol hash differs from current protocol")
    if len(plans) != len(config.get("selected_indices", [])):
        raise ValueError("plan count differs from requested sample count")
    for plan in plans:
        if plan.get("status") != "valid" or normalize_fixed_plan(plan.get("steps")) is None:
            raise ValueError(
                f"refusing to freeze: sample {plan.get('index')} lacks a valid six-step plan"
            )
    payload = {
        "frozen": True,
        "frozen_at": utc_now(),
        "experiment_version": GATE_VERSION,
        "reason": "Completed static question-only plan@6 plan generation.",
        "files": file_manifest(directory, PLAN_FREEZE_MARKER),
    }
    save_json_atomic(marker, payload)
    print(f"Frozen plan@6 directory: {marker.resolve()}")
    return payload


def verify_plan_directory(directory: Path) -> Dict[str, Any]:
    directory = Path(directory)
    marker = directory / PLAN_FREEZE_MARKER
    if not marker.is_file():
        result = {"valid": False, "mismatches": [{"path": marker.name, "reason": "missing"}]}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    mismatches: List[Dict[str, str]] = []
    recorded = {item["path"]: item for item in manifest.get("files", [])}
    for relative, item in recorded.items():
        path = directory / relative
        if not path.is_file():
            mismatches.append({"path": relative, "reason": "missing"})
        elif path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
            mismatches.append({"path": relative, "reason": "content_changed"})
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != PLAN_FREEZE_MARKER
    }
    for relative in sorted(actual - set(recorded)):
        mismatches.append({"path": relative, "reason": "unexpected_file"})
    result = {"valid": not mismatches, "checked_files": len(recorded), "mismatches": mismatches}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def load_frozen_plans(path: Path) -> Dict[int, Dict[str, Any]]:
    path = Path(path)
    if not verify_plan_directory(path.parent)["valid"]:
        raise ValueError(f"plan source is not a valid frozen plan@6 directory: {path.parent}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    plans: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        steps = normalize_fixed_plan(row.get("steps"))
        if row.get("status") != "valid" or steps is None:
            raise ValueError(f"source plan plan {row.get('index')} is not a valid six-step plan")
        plans[int(row["index"])] = {**row, "steps": steps}
    return plans


def build_prefix_result(
    *,
    index: int,
    question: str,
    question_source: str,
    category: int,
    steps: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    prefix_k: int,
    model_name: str,
) -> Dict[str, Any]:
    prefix_records = list(records[:prefix_k])
    elements, union_stats = union_step_elements(prefix_records)
    request_failed = any(item.get("status") == "request_failed" for item in prefix_records)
    status = "request_failed" if request_failed else "valid" if elements else "model_failed"
    return {
        "index": index,
        "model_name": model_name,
        "prompt_version": GATE_VERSION,
        "h2_mode": "static-question-plan",
        "budget_k": K,
        "prefix_k": prefix_k,
        "inst_category": category,
        "category_name": CATEGORY_NAMES.get(category, f"Unknown({category})"),
        "question": question,
        "question_source_field": question_source,
        "status": status,
        "plan_steps": list(steps),
        "elements": elements,
        "union_statistics": union_stats,
        "token_usage": sum_tokens(prefix_records),
        "elapsed_seconds": sum(float(item.get("elapsed_seconds") or 0.0) for item in prefix_records),
        "step_statuses": [
            {
                "step_id": item["step_id"],
                "status": item["status"],
                "element_count": len(item.get("elements", [])),
            }
            for item in prefix_records
        ],
    }


def save_prefix_outputs(
    output_dir: Path,
    selected_indices: Sequence[int],
    prefix_by_index: Mapping[int, Mapping[int, Dict[str, Any]]],
) -> None:
    flat = [
        prefix_by_index[index][prefix]
        for index in selected_indices
        if index in prefix_by_index
        for prefix in range(1, K + 1)
        if prefix in prefix_by_index[index]
    ]
    save_json_atomic(output_dir / "prefix_results.json", flat)
    for prefix in range(1, K + 1):
        rows = [
            prefix_by_index[index][prefix]
            for index in selected_indices
            if index in prefix_by_index and prefix in prefix_by_index[index]
        ]
        save_json_atomic(output_dir / "prefix_results" / f"k{prefix}" / "rig_results.json", rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--plans-from", type=Path)
    parser.add_argument("--tsv-path", type=Path, default=DEFAULT_TSV_PATH)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--annotation-path", type=Path, default=DEFAULT_ANNOTATION_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--api-base")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--plan-max-tokens", type=int, default=12288)
    parser.add_argument("--step-max-tokens", type=int, default=12288)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-wait", type=float, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--freeze-plan-existing", action="store_true")
    parser.add_argument("--verify-plan-frozen", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        phase = "plan-only" if args.plan_only else "grounding"
        output_dir = DEFAULT_OUTPUT_ROOT / (
            f"{args.model_name}__static-question-plan-at6__{phase}"
        )
    if args.freeze_plan_existing:
        freeze_plan_directory(output_dir)
        return 0
    if args.verify_plan_frozen:
        return 0 if verify_plan_directory(output_dir)["valid"] else 2
    if args.smoke and args.indices:
        raise ValueError("--smoke and --indices are mutually exclusive")
    selected_indices = list(DEFAULT_SMOKE_INDICES if args.smoke else (args.indices or []))
    if not selected_indices:
        raise ValueError("use --smoke or provide --indices")
    if len(selected_indices) != len(set(selected_indices)):
        raise ValueError("duplicate indices are not allowed")
    if args.plan_only and args.plans_from:
        raise ValueError("--plan-only cannot be combined with --plans-from")
    if not args.plan_only and not args.plans_from:
        raise ValueError("grounding execution requires --plans-from from a frozen plan-only run")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory is not empty: {output_dir}; use --resume")

    dataframe = pd.read_csv(args.tsv_path, sep="\t")
    selected = dataframe[dataframe["index"].astype(int).isin(selected_indices)].copy()
    rows = {int(row["index"]): row for _, row in selected.iterrows()}
    if set(rows) != set(selected_indices):
        raise ValueError(f"TSV indices differ from requested indices: found {sorted(rows)}")

    plans_source: Dict[int, Dict[str, Any]] = {}
    if args.plans_from:
        plans_source = load_frozen_plans(args.plans_from)
        if set(plans_source) != set(selected_indices):
            raise ValueError("frozen plan index set differs from selected indices")

    config = {
        "experiment_version": GATE_VERSION,
        "protocol_sha256": canonical_json_sha256(protocol_spec()),
        "model_name": args.model_name,
        "mode": "static-question-plan",
        "budget_k": K,
        "plan_only": args.plan_only,
        "selected_indices": selected_indices,
        "question_source_policy": "prefer_raw_question_then_question",
        "thinking_disabled": False,
        "temperature_explicitly_set": False,
        "guided_json": True,
        "plan_max_tokens": args.plan_max_tokens,
        "step_max_tokens": args.step_max_tokens,
        "api_base": resolve_api_base(args.model_name, args.api_base),
        "annotation_path": str(args.annotation_path.resolve()),
        "annotation_sha256": sha256_file(args.annotation_path),
        "tsv_path": str(args.tsv_path.resolve()),
        "tsv_sha256": sha256_file(args.tsv_path),
        "image_dir": str(args.image_dir.resolve()),
        "input_image_sha256": {
            str(index): source_image_sha256(rows[index], args.image_dir)
            for index in selected_indices
        },
        "plans_from": str(args.plans_from.resolve()) if args.plans_from else None,
        "plans_from_sha256": sha256_file(args.plans_from) if args.plans_from else None,
        "image_preprocessing_matched_to_repeated_single": True,
        "created_at": utc_now(),
    }
    if args.dry_run:
        print(json.dumps({**config, "output_dir": str(output_dir.resolve())}, ensure_ascii=False, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "experiment_config.json"
    if args.resume and config_path.is_file():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        for key in (
            "experiment_version",
            "protocol_sha256",
            "model_name",
            "budget_k",
            "plan_only",
            "selected_indices",
            "plan_max_tokens",
            "step_max_tokens",
            "api_base",
            "annotation_sha256",
            "tsv_sha256",
            "input_image_sha256",
            "plans_from_sha256",
        ):
            if existing_config.get(key) != config.get(key):
                raise ValueError(f"cannot resume with changed {key}")
    else:
        save_json_atomic(config_path, config)

    client = OpenAI(
        api_key=args.api_key,
        base_url=config["api_base"],
        timeout=args.timeout,
        http_client=httpx.Client(timeout=args.timeout, trust_env=False),
    )
    plans_path = output_dir / "plans.json"
    steps_path = output_dir / "step_outputs.json"
    status_path = output_dir / "run_status.json"
    results_path = output_dir / "rig_results.json"

    existing_plans = json.loads(plans_path.read_text(encoding="utf-8")) if plans_path.is_file() else []
    plan_by_index = {int(item["index"]): item for item in existing_plans}
    existing_steps = json.loads(steps_path.read_text(encoding="utf-8")) if steps_path.is_file() else []
    step_by_key = {
        (int(item["index"]), str(item["step_id"])): item for item in existing_steps
    }

    if args.plan_only:
        for position, index in enumerate(selected_indices, start=1):
            prior = plan_by_index.get(index)
            if prior and prior.get("status") != "request_failed":
                print(f"[{position}/{len(selected_indices)}] q{index} skip={prior['status']}")
                continue
            question, source = load_question(rows[index])
            print(f"[{position}/{len(selected_indices)}] q{index} plan@6")
            response, elapsed, error, attempts = call_model(
                client,
                model_name=args.model_name,
                messages=plan_messages(question),
                max_tokens=args.plan_max_tokens,
                extra_body={"guided_json": PLAN_JSON_SCHEMA},
                retries=args.retries,
                retry_wait=args.retry_wait,
            )
            if response is None:
                record = {
                    "index": index,
                    "question": question,
                    "question_source": source,
                    "status": "request_failed",
                    "steps": [],
                    "attempts": attempts,
                    "error": error,
                    "elapsed_seconds": elapsed,
                    "token_usage": {},
                }
            else:
                metadata = response_metadata(response, elapsed)
                steps = parse_fixed_plan(metadata["raw_output"])
                status = (
                    "truncated"
                    if metadata["finish_reason"] == "length"
                    else "valid"
                    if steps
                    else "parse_failed"
                )
                raw_path = output_dir / "raw" / f"q{index:04d}" / "plan.txt"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(metadata["raw_output"], encoding="utf-8")
                record = {
                    "index": index,
                    "question": question,
                    "question_source": source,
                    "status": status,
                    "steps": steps or [],
                    "attempts": attempts,
                    "raw_output_path": str(raw_path.resolve()),
                    **metadata,
                }
            plan_by_index[index] = record
            ordered = [plan_by_index[i] for i in selected_indices if i in plan_by_index]
            save_json_atomic(plans_path, ordered)
            request_failed = [i for i, item in plan_by_index.items() if item["status"] == "request_failed"]
            run_status = {
                "experiment_version": GATE_VERSION,
                "mode": "static-question-plan",
                "plan_only": True,
                "budget_k": K,
                "requested_indices": selected_indices,
                "completed_indices": [
                    i for i in selected_indices
                    if i in plan_by_index and plan_by_index[i]["status"] != "request_failed"
                ],
                "request_failed_indices": sorted(request_failed),
                "request_failed_samples": len(request_failed),
                "valid_plan_samples": sum(item["status"] == "valid" for item in plan_by_index.values()),
                "complete": len(plan_by_index) == len(selected_indices) and not request_failed,
                "updated_at": utc_now(),
            }
            save_json_atomic(status_path, run_status)
            print(f"  status={record['status']} steps={len(record['steps'])}")
        print(f"Plans: {plans_path.resolve()}")
        return 0

    # Grounding execution always copies the frozen plans into its own result directory.
    if not plans_path.is_file():
        save_json_atomic(plans_path, [plans_source[index] for index in selected_indices])
    prefix_by_index: Dict[int, Dict[int, Dict[str, Any]]] = {}
    result_by_index: Dict[int, Dict[str, Any]] = {}
    for sample_position, index in enumerate(selected_indices, start=1):
        row = rows[index]
        question, question_source = load_question(row)
        plan = plans_source[index]
        if plan["question"] != question:
            raise ValueError(f"frozen plan question mismatch for index {index}")
        steps = plan["steps"]
        image = load_image(row, args.image_dir)
        image_uri, original_size = encode_image(image)
        print(f"[{sample_position}/{len(selected_indices)}] q{index} execute 6 static steps")
        for step_position, step in enumerate(steps, start=1):
            key = (index, step["step_id"])
            prior = step_by_key.get(key)
            if prior and prior.get("status") != "request_failed":
                print(f"  [{step_position}/{K}] {step['step_id']} skip={prior['status']}")
                continue
            response, elapsed, error, attempts = call_model(
                client,
                model_name=args.model_name,
                messages=step_messages(question, step["evidence_need"], image_uri),
                max_tokens=args.step_max_tokens,
                extra_body=step_extra_body(),
                retries=args.retries,
                retry_wait=args.retry_wait,
            )
            if response is None:
                record = {
                    "index": index,
                    "step_id": step["step_id"],
                    "round": step_position,
                    "evidence_need": step["evidence_need"],
                    "status": "request_failed",
                    "attempts": attempts,
                    "error": error,
                    "elapsed_seconds": elapsed,
                    "token_usage": {},
                    "elements": [],
                }
            else:
                metadata = response_metadata(response, elapsed)
                parsed = parse_rig_response(metadata["raw_output"])
                status = (
                    "truncated"
                    if metadata["finish_reason"] == "length"
                    else "valid"
                    if parsed
                    else "parse_failed"
                )
                elements = parsed if status == "valid" else []
                raw_path = output_dir / "raw" / f"q{index:04d}" / f"{step['step_id']}.txt"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(metadata["raw_output"], encoding="utf-8")
                record = {
                    "index": index,
                    "step_id": step["step_id"],
                    "round": step_position,
                    "evidence_need": step["evidence_need"],
                    "status": status,
                    "attempts": attempts,
                    "elements": elements,
                    "diagnostic_parsed_elements": parsed if status == "truncated" else None,
                    "raw_output_path": str(raw_path.resolve()),
                    **metadata,
                }
            step_by_key[key] = record
            ordered_steps = [
                step_by_key[(i, f"s{k}")]
                for i in selected_indices
                for k in range(1, K + 1)
                if (i, f"s{k}") in step_by_key
            ]
            save_json_atomic(steps_path, ordered_steps)
            print(
                f"  [{step_position}/{K}] {step['step_id']} "
                f"status={record['status']} boxes={len(record['elements'])}"
            )

        sample_records = [step_by_key[(index, f"s{k}")] for k in range(1, K + 1)]
        prefix_by_index[index] = {}
        category = int(row.get("inst_category", 0))
        for prefix in range(1, K + 1):
            prefix_result = build_prefix_result(
                index=index,
                question=question,
                question_source=question_source,
                category=category,
                steps=steps,
                records=sample_records,
                prefix_k=prefix,
                model_name=args.model_name,
            )
            prefix_by_index[index][prefix] = prefix_result
            visualization = output_dir / "visualizations" / f"q{index:04d}_k{prefix}.jpg"
            draw_bboxes_on_image(
                image,
                prefix_result["elements"],
                original_size,
                visualization,
                show=False,
            )
        result_by_index[index] = prefix_by_index[index][K]
        save_prefix_outputs(output_dir, selected_indices, prefix_by_index)
        save_json_atomic(
            results_path,
            [result_by_index[i] for i in selected_indices if i in result_by_index],
        )
        request_failed_indices = sorted(
            i
            for i in selected_indices
            if any(
                step_by_key.get((i, f"s{k}"), {}).get("status") == "request_failed"
                for k in range(1, K + 1)
            )
        )
        completed_indices = [
            i
            for i in selected_indices
            if all(
                (i, f"s{k}") in step_by_key
                and step_by_key[(i, f"s{k}")]["status"] != "request_failed"
                for k in range(1, K + 1)
            )
        ]
        save_json_atomic(
            status_path,
            {
                "experiment_version": GATE_VERSION,
                "mode": "static-question-plan",
                "plan_only": False,
                "budget_k": K,
                "requested_indices": selected_indices,
                "completed_indices": completed_indices,
                "request_failed_indices": request_failed_indices,
                "request_failed_samples": len(request_failed_indices),
                "complete": len(completed_indices) == len(selected_indices),
                "updated_at": utc_now(),
            },
        )

    print(f"Final results: {results_path.resolve()}")
    print(f"Prefix results: {(output_dir / 'prefix_results').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""H2 pilot runner: single-pass versus plan-conditioned multi-step grounding.

This runner never writes to the existing Direct-BBox run directories.  Each mode
has an isolated directory and emits an evaluator-compatible ``rig_results.json``
plus richer plan, step, provenance, token, latency, and status artifacts.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from openai import OpenAI
from PIL import Image

from test_RIG_on_Charxiv import (
    PROMPT_VERSION as SINGLE_PROMPT_VERSION,
    RIG_USER_PROMPT_TEMPLATE,
    build_rig_extra_body,
    build_rig_messages,
    draw_bboxes_on_image,
    parse_rig_response,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TSV_PATH = Path(r"C:/Users/19639/LMUData/CharXiv_reasoning_val.tsv")
DEFAULT_IMAGE_DIR = Path(r"C:/Users/19639/LMUData/images/CharXiv_reasoning_val")
DEFAULT_ANNOTATION_PATH = (
    SCRIPT_DIR.parent
    / "RIG-Annotating-main/annotation of RIG/Charxiv_reasoning_val/annotation_1.json"
)
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "test_case/RIG_Charxiv/h2_grounding"
DEFAULT_MODEL_NAME = "Qwen3.5-9B"
DEFAULT_API_BASES = {
    "Qwen3.5-9B": os.environ.get("QWEN35_9B_API_BASE", "http://10.3.112.34:22630/v1"),
    "Qwen3.5-9B-API": os.environ.get("QWEN35_9B_API_BASE", "http://10.3.112.34:22630/v1"),
    "Qwen3.5-27B": os.environ.get("QWEN35_27B_API_BASE", "http://10.3.112.34:23290/v1"),
    "Qwen3.5-27B-API": os.environ.get("QWEN35_27B_API_BASE", "http://10.3.112.34:23290/v1"),
}
DEFAULT_SMOKE_INDICES = (1, 17, 8, 244, 346)
H2_VERSION = "h2-grounding-v2-concise-complete"
FREEZE_MARKER = "FROZEN.json"
DEFAULT_PROTOCOL_DIR = DEFAULT_OUTPUT_ROOT / "protocols" / H2_VERSION
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "reviews" / H2_VERSION
SUPPORTED_MODES = ("single", "repeated-single", "predicted-plan", "reviewed-plan")
CATEGORY_NAMES = {
    1: "Text-in-Chart",
    2: "Text-in-General",
    3: "Number-in-Chart",
    4: "Number-in-General",
}


PLAN_SYSTEM_PROMPT = """You decompose a visual grounding task into evidence needs.
You receive only the question, not the image. Do not solve the question, predict the
answer, state candidate values, or invent visible content. Return only the requested
JSON object without markdown or analysis."""

PLAN_USER_TEMPLATE = """QUESTION:
__QUESTION_PLACEHOLDER__

Create 2 to 6 ordered evidence-grounding steps for a downstream visual grounder.
Each step must request one coherent kind of directly visible evidence needed to
derive the answer. Split identity mappings, scales or criteria, candidate observations,
and spatial/structural relations when applicable.

The plan must describe evidence requirements only. It must not contain the final
answer.
Do not create a step whose job is to compare, rank, count, calculate, conclude, or
select the answer. Those operations belong to a later reasoner; request the visible
inputs needed by those operations instead.
For a comparison, ranking, distance, or count question, preserve all operands of the
criterion: request every candidate and counterpart together with the shared axis,
scale, or complete trace needed to compare them. Name both sides of a spatial
relation; never replace a specific relation with a generic request for "position".
Request bbox-localizable visible content, not estimated centroids, derived data
coordinates, distances, rankings, or counts. Transcription is allowed only for text
or values explicitly printed in the image.
Describe the visual content in natural language; do not request bbox numbers, image
dimensions, pixels, or coordinate conversion. If the question supplies a target data
coordinate that is not visibly marked, request the visible axes and tick labels needed
to locate it rather than requesting canvas metadata.
Because you cannot see the image, do not assume an unmentioned visual encoding such
as a table, row, column, chart, legend, curve, or marker. Use encoding-neutral terms
such as visible candidate, identity evidence, value evidence, and spatial position;
the downstream grounder will identify how that evidence is actually displayed.

OUTPUT ONLY:
{
  "steps": [
    {
      "step_id": "s1",
      "evidence_need": "one precise visible-evidence requirement"
    }
  ]
}"""

PLAN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "minItems": 2,
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "string", "minLength": 1, "maxLength": 24},
                    "evidence_need": {"type": "string", "minLength": 1, "maxLength": 320},
                },
                "required": ["step_id", "evidence_need"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}

STEP_SYSTEM_PROMPT = """You locate visible evidence for one assigned step.
Use internal visual reasoning to identify and localize the requested evidence
accurately. Do not solve the final question or choose an answer. Return only the
requested JSON object; do not expose analysis or a checklist."""

STEP_USER_TEMPLATE = """IMAGE: A chart or scientific figure.

QUESTION:
__QUESTION_PLACEHOLDER__

CURRENT EVIDENCE NEED:
__EVIDENCE_NEED_PLACEHOLDER__

Locate every visible clue required for CURRENT EVIDENCE NEED. Cover every relevant
candidate, series, or panel, not only the most salient item or likely answer.

RULES
- One item is one semantic clue; keep distinct clues separate even if boxes overlap.
- Box the full visible clue. For a curve, axis, or spatial relation, include the
  complete relevant segment or region needed to perceive it.
- Include only evidence for this step. Do not use the whole image as a shortcut.
- Return at most 16 elements. bbox uses integer [xmin,ymin,xmax,ymax] in [0,1000].
- Transcribe visible text or values exactly when readable.

OUTPUT ONLY:
{
  "elements": [
    {
      "description": "visible identity, value, or structural clue",
      "bbox": [xmin, ymin, xmax, ymax]
    }
  ]
}"""

STEP_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "elements": {
            "type": "array",
            "maxItems": 16,
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "minLength": 1, "maxLength": 320},
                    "bbox": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                    },
                },
                "required": ["description", "bbox"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["elements"],
    "additionalProperties": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not result:
        raise ValueError(f"cannot form safe name from {value!r}")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_protocol_spec() -> Dict[str, Any]:
    if not DEFAULT_ANNOTATION_PATH.is_file():
        raise FileNotFoundError(f"H2 annotation file is missing: {DEFAULT_ANNOTATION_PATH}")
    return {
        "protocol_version": H2_VERSION,
        "task": "question-conditioned necessary visual evidence grounding",
        "scientific_hypothesis": (
            "step-conditioned grounding improves necessary-evidence discovery beyond "
            "the gain from an equal number of repeated single-pass calls"
        ),
        "modes": list(SUPPORTED_MODES),
        "question_source_policy": "prefer_raw_question_then_question",
        "formal_dataset": {
            "annotation_path": str(DEFAULT_ANNOTATION_PATH.resolve()),
            "annotation_sha256": sha256_file(DEFAULT_ANNOTATION_PATH),
            "expected_samples": 41,
            "coverage_policy": "exactly all indices in annotation_1.json",
        },
        "planner": {
            "image_visible": False,
            "system_prompt": PLAN_SYSTEM_PROMPT,
            "user_prompt_template": PLAN_USER_TEMPLATE,
            "guided_json_schema": PLAN_JSON_SCHEMA,
            "min_steps": 2,
            "max_steps": 6,
            "terminal_failure_policy": (
                "truncated or unparseable plans are preserved as zero-step model failures; "
                "they are never dropped, repaired, or regenerated in a formal run"
            ),
        },
        "step_grounder": {
            "system_prompt": STEP_SYSTEM_PROMPT,
            "user_prompt_template": STEP_USER_TEMPLATE,
            "guided_json_schema": STEP_JSON_SCHEMA,
            "soft_target_max_elements": None,
            "hard_max_elements": 16,
        },
        "single_control": {
            "source_prompt_version": SINGLE_PROMPT_VERSION,
            "repeated_control_policy": "match predicted step count per question",
        },
        "review_policy": {
            "image_visible_to_reviewer": False,
            "same_step_ids_order_and_count_as_predicted": True,
            "failed_source_plans_remain_zero_step_failures": True,
            "may_edit": ["evidence_need", "review_notes"],
            "prohibited_inputs": [
                "answer",
                "GT bbox",
                "annotation description",
                "grounding output",
            ],
        },
        "inference": {
            "max_completion_tokens_per_call": 12288,
            "thinking_disabled": False,
            "temperature_explicitly_set": False,
        },
        "union_policy": {
            "deduplicate_only": "casefolded identical description plus identical bbox",
            "preserve_identical_bbox_with_different_semantics": True,
            "coordinate_space": "integer [0,1000] xyxy",
        },
        "evaluation": {
            "primary_comparison": "predicted-plan versus call-matched repeated-single",
            "primary_metric": "paired question-level macro necessary recall difference",
            "primary_semantic_policy": "overlap_review",
            "primary_iou_profile": 0.3,
            "secondary_iou_profile": 0.5,
            "secondary_metrics": [
                "macro precision",
                "macro F1",
                "complete evidence rate",
                "mean prediction union area ratio",
                "mean prediction count",
            ],
            "uncertainty": {
                "method": "paired question bootstrap percentile 95% CI",
                "resamples": 10000,
                "seed": 20260903,
                "multiplicity": "one primary comparison; no correction required",
            },
            "claim_tiers": {
                "coverage_support": "predicted-plan improves recall over repeated-single",
                "efficient_grounding_support": (
                    "coverage support plus improved macro F1 without increased mean union area"
                ),
                "planning_bottleneck_evidence": (
                    "reviewed-plan improves over predicted-plan under identical step count"
                ),
            },
        },
        "reporting": {
            "paired_unit": "question",
            "required_failure_rates": ["sample_level", "plan_level", "step_level"],
            "required_costs": ["image_calls", "prompt_tokens", "completion_tokens", "wall_time"],
            "all_sample_primary": True,
            "valid_plan_subset_secondary_only": True,
        },
    }


def load_formal_indices(annotation_path: Path) -> List[int]:
    payload = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list):
        raise ValueError("formal annotation must contain an images list")
    indices: List[int] = []
    for image in images:
        if not isinstance(image, dict) or not image.get("path"):
            raise ValueError("formal annotation contains an image without a path")
        try:
            indices.append(int(Path(str(image["path"])).stem))
        except ValueError as exc:
            raise ValueError(f"cannot derive dataset index from {image.get('path')!r}") from exc
    if len(indices) != 41 or len(set(indices)) != 41:
        raise ValueError(f"formal H2 annotation must contain 41 unique indices, found {len(indices)}")
    return indices


def freeze_protocol(protocol_dir: Path) -> Dict[str, Any]:
    protocol_dir = Path(protocol_dir)
    marker = protocol_dir / FREEZE_MARKER
    if marker.is_file():
        result = verify_protocol(protocol_dir)
        if not result["valid"]:
            raise RuntimeError("existing frozen H2 protocol no longer matches runtime code")
        print(f"Already frozen and valid: {protocol_dir}")
        return json.loads(marker.read_text(encoding="utf-8"))
    protocol_dir.mkdir(parents=True, exist_ok=True)
    spec_path = protocol_dir / "protocol.json"
    spec = build_protocol_spec()
    if spec_path.is_file():
        existing = json.loads(spec_path.read_text(encoding="utf-8"))
        if existing != spec:
            raise ValueError(f"refusing to overwrite a different protocol spec: {spec_path}")
    else:
        save_json_atomic(spec_path, spec)
    payload = {
        "frozen": True,
        "frozen_at": utc_now(),
        "protocol_version": H2_VERSION,
        "protocol_directory": str(protocol_dir.resolve()),
        "protocol_file": "protocol.json",
        "protocol_file_sha256": sha256_file(spec_path),
        "canonical_protocol_sha256": canonical_json_sha256(spec),
        "runner_file": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    save_json_atomic(marker, payload)
    print(f"Frozen H2 protocol: {marker}")
    return payload


def verify_protocol(protocol_dir: Path) -> Dict[str, Any]:
    protocol_dir = Path(protocol_dir)
    marker = protocol_dir / FREEZE_MARKER
    if not marker.is_file():
        raise FileNotFoundError(f"protocol freeze marker not found: {marker}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    spec_path = protocol_dir / str(payload.get("protocol_file", "protocol.json"))
    mismatches: List[Dict[str, str]] = []
    if not spec_path.is_file():
        mismatches.append({"target": "protocol_file", "reason": "missing"})
    else:
        saved_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if sha256_file(spec_path) != payload.get("protocol_file_sha256"):
            mismatches.append({"target": "protocol_file", "reason": "content_changed"})
        if canonical_json_sha256(saved_spec) != payload.get("canonical_protocol_sha256"):
            mismatches.append({"target": "protocol_canonical", "reason": "content_changed"})
        if saved_spec != build_protocol_spec():
            mismatches.append({"target": "runtime_protocol", "reason": "runtime_drift"})
    if sha256_file(Path(__file__).resolve()) != payload.get("runner_sha256"):
        mismatches.append({"target": "runner", "reason": "content_changed"})
    result = {
        "valid": not mismatches,
        "protocol_version": payload.get("protocol_version"),
        "mismatches": mismatches,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def assert_run_directory_writable(output_dir: Path, operation: str) -> None:
    marker = Path(output_dir) / FREEZE_MARKER
    if marker.is_file():
        raise PermissionError(
            f"refusing to {operation}: run is frozen by {marker}; use a new --output-dir"
        )


def freeze_run_directory(output_dir: Path, *, reason: str) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {output_dir}")
    marker = output_dir / FREEZE_MARKER
    if marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        print(f"Already frozen: {output_dir}")
        return payload
    status_path = output_dir / "run_status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"refusing to freeze run without {status_path.name}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("complete") is not True or status.get("protocol_complete") is not True:
        raise ValueError(
            "refusing to freeze incomplete H2 run: "
            f"complete={status.get('complete')!r}, "
            f"protocol_complete={status.get('protocol_complete')!r}"
        )
    requested = {int(index) for index in status.get("requested_indices", [])}
    completed = {int(index) for index in status.get("completed_indices", [])}
    if not requested or completed != requested:
        raise ValueError("refusing to freeze H2 run without complete index coverage")
    if status.get("request_failed_indices") or status.get("request_failed_samples"):
        raise ValueError("refusing to freeze H2 run with request failures")
    config_path = output_dir / "experiment_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"refusing to freeze run without {config_path.name}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_protocol_hash = canonical_json_sha256(build_protocol_spec())
    if config.get("experiment_version") != H2_VERSION:
        raise ValueError("refusing to freeze a run from a different H2 protocol version")
    if config.get("protocol_sha256") != expected_protocol_hash:
        raise ValueError("refusing to freeze a run whose protocol hash differs from runtime")
    if config.get("annotation_sha256") != build_protocol_spec()["formal_dataset"]["annotation_sha256"]:
        raise ValueError("refusing to freeze a run with a different formal annotation file")
    if config.get("protocol_frozen") is not True:
        raise ValueError("refusing to freeze a run that was not started under a frozen protocol")
    if config.get("run_scope") != "formal":
        raise ValueError("refusing to freeze a calibration/smoke run as a formal H2 result")
    protocol_directory = config.get("protocol_directory")
    if not protocol_directory or not verify_protocol(Path(protocol_directory))["valid"]:
        raise ValueError("refusing to freeze a run whose protocol freeze cannot be verified")
    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != FREEZE_MARKER and not path.name.endswith(".tmp"):
            files.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    payload = {
        "frozen": True,
        "frozen_at": utc_now(),
        "reason": reason,
        "run_directory": str(output_dir.resolve()),
        "experiment_version": status.get("experiment_version"),
        "mode": status.get("mode"),
        "file_count": len(files),
        "files": files,
    }
    save_json_atomic(marker, payload)
    print(f"Frozen {output_dir}: {len(files)} files")
    return payload


def verify_frozen_directory(output_dir: Path) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    marker = output_dir / FREEZE_MARKER
    if not marker.is_file():
        raise FileNotFoundError(f"freeze marker not found: {marker}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    recorded = {str(item["path"]): item for item in payload.get("files", [])}
    mismatches = []
    for relative, item in recorded.items():
        path = output_dir / relative
        if not path.is_file():
            mismatches.append({"path": relative, "reason": "missing"})
        elif path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            mismatches.append({"path": relative, "reason": "content_changed"})
    actual = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != FREEZE_MARKER and not path.name.endswith(".tmp")
    }
    for relative in sorted(actual - set(recorded)):
        mismatches.append({"path": relative, "reason": "unexpected_file"})
    result = {
        "valid": not mismatches,
        "checked_files": len(recorded),
        "mismatches": mismatches,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def mode_output_dir(model_name: str, mode: str, output_dir: Optional[Path]) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return DEFAULT_OUTPUT_ROOT / mode / f"{safe_name(model_name)}__{H2_VERSION}"


def default_single_source(model_name: str) -> Path:
    return (
        SCRIPT_DIR
        / "test_case/RIG_Charxiv/runs"
        / f"{safe_name(model_name)}__{SINGLE_PROMPT_VERSION}"
        / "rig_results.json"
    )


def resolve_api_base(model_name: str, explicit_api_base: Optional[str]) -> str:
    if explicit_api_base:
        return explicit_api_base
    try:
        return DEFAULT_API_BASES[model_name]
    except KeyError as exc:
        raise ValueError(
            f"no default API base for {model_name!r}; pass --api-base explicitly"
        ) from exc


def load_question(row: pd.Series) -> Tuple[str, str]:
    raw_question = row.get("raw_question")
    if isinstance(raw_question, str) and raw_question.strip():
        return raw_question.strip(), "raw_question"
    question = row.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip(), "question"
    raise ValueError(f"sample {row.get('index')} has no usable question")


def load_image(row: pd.Series, image_dir: Path) -> Image.Image:
    index = int(row["index"])
    local_path = image_dir / f"{index}.jpg"
    if local_path.is_file():
        return Image.open(local_path).convert("RGB")
    encoded = row.get("image")
    if isinstance(encoded, str) and encoded:
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    raise FileNotFoundError(f"no image found for index {index}")


def source_image_sha256(row: pd.Series, image_dir: Path) -> str:
    index = int(row["index"])
    local_path = image_dir / f"{index}.jpg"
    if local_path.is_file():
        return sha256_file(local_path)
    encoded = row.get("image")
    if isinstance(encoded, str) and encoded:
        return hashlib.sha256(base64.b64decode(encoded)).hexdigest()
    raise FileNotFoundError(f"no image found for index {index}")


def encode_image(image: Image.Image, max_size: int = 8192) -> Tuple[str, Tuple[int, int]]:
    original_size = image.size
    prepared = image
    if image.width > max_size or image.height > max_size:
        prepared = image.copy()
        prepared.thumbnail((max_size, max_size))
    buffer = io.BytesIO()
    prepared.save(buffer, format="JPEG", quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}", original_size


def extract_json_objects(text: str) -> List[Mapping[str, Any]]:
    # Try JSON decoding from every opening brace.  This recovers a valid final
    # object even when the model's visible thinking contains an unmatched brace.
    # A single depth counter would remain negative and miss the final object.
    source = text or ""
    decoder = json.JSONDecoder()
    parsed: List[Mapping[str, Any]] = []
    for position, character in enumerate(source):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(source[position:])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            parsed.append(value)
    return parsed


def recover_plan_records(plans: List[Dict[str, Any]]) -> bool:
    changed = False
    for record in plans:
        if record.get("status") == "valid" and record.get("steps"):
            continue
        recovered = parse_plan_response(str(record.get("raw_output") or ""))
        if recovered:
            record["steps"] = recovered
            record["status"] = "valid"
            record["parser_recovered_at"] = utc_now()
            changed = True
    return changed


def normalize_steps(raw_steps: Any) -> Optional[List[Dict[str, str]]]:
    if not isinstance(raw_steps, list) or not 2 <= len(raw_steps) <= 6:
        return None
    steps: List[Dict[str, str]] = []
    used_ids = set()
    for position, raw in enumerate(raw_steps, start=1):
        if not isinstance(raw, dict):
            return None
        need = str(raw.get("evidence_need", "")).strip()
        if not need:
            return None
        step_id = safe_name(str(raw.get("step_id") or f"s{position}"))[:24]
        if step_id in used_ids:
            step_id = f"s{position}"
        used_ids.add(step_id)
        steps.append({"step_id": step_id, "evidence_need": need[:320]})
    return steps


def parse_plan_response(text: str) -> Optional[List[Dict[str, str]]]:
    for candidate in reversed(extract_json_objects(text)):
        steps = normalize_steps(candidate.get("steps"))
        if steps:
            return steps
    return None


def iou(first: Sequence[int], second: Sequence[int]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def rectangle_union_area(boxes: Sequence[Sequence[int]]) -> float:
    if not boxes:
        return 0.0
    xs = sorted({int(box[0]) for box in boxes} | {int(box[2]) for box in boxes})
    area = 0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (int(box[1]), int(box[3])) for box in boxes if box[0] < right and box[2] > left
        )
        covered = 0
        if intervals:
            start, end = intervals[0]
            for current_start, current_end in intervals[1:]:
                if current_start > end:
                    covered += end - start
                    start, end = current_start, current_end
                else:
                    end = max(end, current_end)
            covered += end - start
        area += (right - left) * covered
    return area / 1_000_000.0


def union_step_elements(step_records: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    elements: List[Dict[str, Any]] = []
    exact_lookup: Dict[Tuple[str, Tuple[int, int, int, int]], int] = {}
    raw_elements: List[Tuple[str, Mapping[str, Any]]] = []
    exact_duplicates = 0
    for record in step_records:
        step_id = str(record["step_id"])
        for element in record.get("elements", []):
            raw_elements.append((step_id, element))
            bbox = tuple(int(value) for value in element["bbox"])
            description = re.sub(r"\s+", " ", str(element.get("description", "")).strip())
            key = (description.casefold(), bbox)
            provenance = {"step_id": step_id, "call_id": record.get("call_id")}
            if key in exact_lookup:
                elements[exact_lookup[key]]["provenance"].append(provenance)
                exact_duplicates += 1
                continue
            exact_lookup[key] = len(elements)
            elements.append(
                {
                    "id": f"pred_{len(elements):03d}",
                    "description": description,
                    "bbox": list(bbox),
                    "provenance": [provenance],
                }
            )
    near_duplicate_pairs = 0
    cross_step_pairs = 0
    for left in range(len(raw_elements)):
        for right in range(left + 1, len(raw_elements)):
            if raw_elements[left][0] == raw_elements[right][0]:
                continue
            cross_step_pairs += 1
            if iou(raw_elements[left][1]["bbox"], raw_elements[right][1]["bbox"]) >= 0.9:
                near_duplicate_pairs += 1
    stats = {
        "raw_element_count": len(raw_elements),
        "union_element_count": len(elements),
        "exact_duplicates_removed": exact_duplicates,
        "cross_step_candidate_pairs": cross_step_pairs,
        "cross_step_near_duplicate_pairs_iou_0_9": near_duplicate_pairs,
        "cross_step_near_duplicate_rate": (
            near_duplicate_pairs / cross_step_pairs if cross_step_pairs else 0.0
        ),
        "bbox_union_area_ratio": rectangle_union_area([element["bbox"] for element in elements]),
    }
    return elements, stats


def response_metadata(response: Any, elapsed_seconds: float) -> Dict[str, Any]:
    message = response.choices[0].message
    usage = getattr(response, "usage", None)
    return {
        "raw_output": message.content or "",
        "reasoning_content": getattr(message, "reasoning_content", None),
        "finish_reason": getattr(response.choices[0], "finish_reason", None),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_seconds_available": True,
        "token_usage": normalize_token_usage({
            "prompt": getattr(usage, "prompt_tokens", None),
            "completion": getattr(usage, "completion_tokens", None),
            "total": getattr(usage, "total_tokens", None),
        }),
    }


def call_model(
    client: OpenAI,
    *,
    model_name: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    extra_body: Mapping[str, Any],
    retries: int,
    retry_wait: float,
) -> Tuple[Optional[Any], float, Optional[str], int]:
    last_error: Optional[str] = None
    started = time.perf_counter()
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                extra_body=dict(extra_body),
            )
            return response, time.perf_counter() - started, None, attempt
        except Exception as exc:  # API clients expose several transport-specific exceptions.
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(retry_wait * attempt)
    return None, time.perf_counter() - started, last_error, retries


def plan_messages(question: str) -> List[Dict[str, str]]:
    prompt = PLAN_USER_TEMPLATE.replace("__QUESTION_PLACEHOLDER__", question)
    return [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def step_messages(question: str, need: str, image_data_uri: str) -> List[Dict[str, Any]]:
    prompt = STEP_USER_TEMPLATE.replace("__QUESTION_PLACEHOLDER__", question).replace(
        "__EVIDENCE_NEED_PLACEHOLDER__", need
    )
    return [
        {"role": "system", "content": STEP_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ],
        },
    ]


def assert_file_is_frozen(path: Path) -> None:
    path = Path(path).resolve()
    marker = path.parent / FREEZE_MARKER
    if not marker.is_file():
        raise ValueError(f"plan source is not frozen: {marker} is missing")
    verification = verify_frozen_directory(path.parent)
    if not verification["valid"]:
        raise ValueError(f"plan source freeze verification failed: {path.parent}")
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    relative = path.relative_to(path.parent).as_posix()
    item = next((entry for entry in manifest.get("files", []) if entry["path"] == relative), None)
    if item is None or item.get("sha256") != sha256_file(path):
        raise ValueError(f"plan source is not covered by its freeze manifest: {path}")


def load_predicted_plans(path: Path, *, require_frozen: bool = True) -> Dict[int, Dict[str, Any]]:
    path = Path(path)
    if require_frozen:
        assert_file_is_frozen(path)
        config_path = path.parent / "experiment_config.json"
        if not config_path.is_file():
            raise ValueError("frozen plan source has no experiment_config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("experiment_version") != H2_VERSION or config.get("plan_only") is not True:
            raise ValueError("--plans-from must be a frozen plan-only run of the current protocol")
        if config.get("protocol_sha256") != canonical_json_sha256(build_protocol_spec()):
            raise ValueError("frozen plan source protocol hash does not match the current protocol")
        protocol_directory = config.get("protocol_directory")
        if config.get("protocol_frozen") is not True or not protocol_directory:
            raise ValueError("formal plan source was not generated under a frozen protocol")
        if not verify_protocol(Path(protocol_directory))["valid"]:
            raise ValueError("formal plan source protocol freeze verification failed")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("--plans-from must point to a plans.json list")
    plans: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        index = int(row["index"])
        if index in plans:
            raise ValueError(f"duplicate predicted plan index: {index}")
        status = row.get("status")
        steps = normalize_steps(row.get("steps")) or []
        if status == "valid":
            if not steps:
                raise ValueError(f"predicted plan {index} is marked valid but has no valid steps")
        elif status in {"truncated", "parse_failed"}:
            if steps:
                raise ValueError(f"failed predicted plan {index} unexpectedly contains valid steps")
        else:
            raise ValueError(f"predicted plan {index} has non-terminal status {status!r}")
        plans[index] = {**row, "index": index, "steps": steps}
    return plans


def load_reviewed_plans(
    path: Path,
    *,
    predicted_plan_path: Optional[Path] = None,
    predicted_plans: Optional[Mapping[int, Mapping[str, Any]]] = None,
    allow_engineering_review: bool = False,
) -> Dict[int, Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("plans") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("reviewed plan file must contain a plans list")
    if isinstance(payload, dict):
        scope = str(payload.get("review_scope") or payload.get("use_scope") or "")
        if "engineering" in scope and not allow_engineering_review:
            raise ValueError("engineering-only reviewed plans cannot be used without explicit override")
        if predicted_plan_path is not None:
            expected_hash = sha256_file(Path(predicted_plan_path))
            if payload.get("source_sha256") != expected_hash:
                raise ValueError("reviewed plan source_sha256 does not match --plans-from")
    plans: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        index = int(row["index"])
        if index in plans:
            raise ValueError(f"duplicate reviewed plan index: {index}")
        allowed_row_keys = {
            "index",
            "question",
            "review_status",
            "reviewer",
            "review_notes",
            "steps",
        }
        unexpected_row_keys = set(row) - allowed_row_keys
        if unexpected_row_keys:
            raise ValueError(
                f"reviewed plan {index} contains prohibited fields: "
                f"{sorted(unexpected_row_keys)}"
            )
        source = predicted_plans.get(index) if predicted_plans is not None else None
        source_failed = source is not None and source.get("status") in {
            "truncated",
            "parse_failed",
        }
        expected_review_status = "source_invalid" if source_failed else "approved"
        if row.get("review_status") != expected_review_status:
            raise ValueError(
                f"reviewed plan {index} must have review_status={expected_review_status!r}"
            )
        if not source_failed and not str(row.get("reviewer") or "").strip():
            raise ValueError(f"reviewed plan {index} has no reviewer identity")
        for step in row.get("steps", []):
            if not isinstance(step, dict) or set(step) != {"step_id", "evidence_need"}:
                raise ValueError(
                    f"reviewed plan {index} step may contain only step_id and evidence_need"
                )
        steps = normalize_steps(row.get("steps")) or []
        if not steps and not source_failed:
            raise ValueError(f"reviewed plan {index} has invalid steps")
        if source_failed and steps:
            raise ValueError(f"reviewed plan {index} must preserve its failed source as zero steps")
        if predicted_plans is not None:
            if source is None:
                raise ValueError(f"reviewed plan has no predicted source for index {index}")
            source_steps = source["steps"]
            if [step["step_id"] for step in steps] != [step["step_id"] for step in source_steps]:
                raise ValueError(
                    f"reviewed plan {index} changed step count, order, or step IDs"
                )
            if str(row.get("question", "")) != str(source.get("question", "")):
                raise ValueError(f"reviewed plan {index} changed the question")
        plans[index] = {**row, "index": index, "steps": steps}
    if predicted_plans is not None and set(plans) != set(predicted_plans):
        raise ValueError("reviewed and predicted plan index sets do not match")
    return plans


def export_review_template(source: Path, output: Path) -> None:
    plans = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(plans, list):
        raise ValueError("plans.json must contain a list")
    exported = {
        "format": "h2-reviewed-plans-v2",
        "source_plans": str(Path(source).resolve()),
        "source_sha256": sha256_file(source),
        "review_scope": "formal_blind_review",
        "review_protocol": {
            "same_step_ids_order_and_count_required": True,
            "prohibited_inputs": [
                "answer",
                "GT bbox",
                "annotation description",
                "grounding output",
            ],
        },
        "instructions": (
            "Edit evidence_need text only and preserve every step_id, its order, and the total "
            "step count. Do not inspect or add answers, GT boxes, annotation descriptions, "
            "grounding outputs, or claimed visible values. Set review_status to approved and "
            "record reviewer identity after blind human review. Leave source_invalid rows "
            "unchanged; failed predicted plans must not be repaired."
        ),
        "plans": [
            {
                "index": int(row["index"]),
                "question": row.get("question", ""),
                "review_status": (
                    "source_invalid"
                    if row.get("status") in {"truncated", "parse_failed"}
                    else "pending"
                ),
                "reviewer": None,
                "review_notes": "",
                "steps": row.get("steps", []),
            }
            for row in plans
        ],
    }
    save_json_atomic(output, exported)
    print(f"Review template: {output.resolve()}")


def load_single_source(path: Path) -> Dict[int, Dict[str, Any]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"single source must contain a result list: {path}")
    return {int(row["index"]): row for row in rows}


def normalize_token_usage(usage: Any) -> Dict[str, Optional[int]]:
    source = usage if isinstance(usage, Mapping) else {}
    prompt = source.get("prompt")
    completion = source.get("completion")
    total = source.get("total")
    if not isinstance(total, (int, float)) and isinstance(prompt, (int, float)) and isinstance(
        completion, (int, float)
    ):
        total = prompt + completion
    return {
        "prompt": int(prompt) if isinstance(prompt, (int, float)) else None,
        "completion": int(completion) if isinstance(completion, (int, float)) else None,
        "total": int(total) if isinstance(total, (int, float)) else None,
    }


def load_repeat_counts(path: Path) -> Dict[int, int]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("--repeat-counts-from must point to a plans.json list")
    counts: Dict[int, int] = {}
    for row in rows:
        index = int(row["index"])
        status = row.get("status")
        steps = normalize_steps(row.get("steps")) or []
        if status in (None, "valid") and steps:
            counts[index] = len(steps)
        elif status in {"truncated", "parse_failed"} and not steps:
            counts[index] = 0
        else:
            raise ValueError(f"repeat-count source plan {index} has inconsistent status/steps")
    return counts


def sum_tokens(records: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    totals = {"prompt": 0, "completion": 0, "total": 0}
    for record in records:
        usage = record.get("token_usage", {})
        for key in totals:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                totals[key] += int(value)
    return totals


def plan_for_mode(
    *,
    mode: str,
    index: int,
    question: str,
    repeats: int,
    repeat_counts: Mapping[int, int],
    reviewed_plans: Mapping[int, Mapping[str, Any]],
) -> Optional[List[Dict[str, str]]]:
    if mode == "single":
        return [{"step_id": "single", "evidence_need": "Ground all necessary evidence jointly."}]
    if mode == "repeated-single":
        repeat_count = repeat_counts.get(index, repeats)
        return [
            {"step_id": f"repeat_{position}", "evidence_need": "Ground all necessary evidence jointly."}
            for position in range(1, repeat_count + 1)
        ]
    if mode == "reviewed-plan":
        if index not in reviewed_plans:
            raise ValueError(f"approved reviewed plan missing index {index}")
        return list(reviewed_plans[index]["steps"])
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default="predicted-plan")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--smoke", action="store_true", help="Use fixed indices 1 17 8 244 346.")
    parser.add_argument("--tsv-path", type=Path, default=DEFAULT_TSV_PATH)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--annotation-path", type=Path, default=DEFAULT_ANNOTATION_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--formal",
        action="store_true",
        help="Require the exact 41-sample annotation_1 index set and frozen protocol/sources.",
    )
    parser.add_argument("--single-source", type=Path)
    parser.add_argument("--reviewed-plans", type=Path)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Generate predicted plans without loading images or running step grounding.",
    )
    parser.add_argument(
        "--plans-from",
        type=Path,
        help="Frozen plan-only plans.json used by predicted-plan or reviewed-plan execution.",
    )
    parser.add_argument(
        "--allow-unfrozen-plans",
        action="store_true",
        help="Engineering-only override; formal runs must consume frozen plan files.",
    )
    parser.add_argument(
        "--allow-engineering-review",
        action="store_true",
        help="Engineering-only override for a review file marked as engineering smoke.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--repeat-counts-from",
        type=Path,
        help="Optional predicted plans.json; match repeated-single calls per question to its step count.",
    )
    parser.add_argument("--api-base")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    # Keep every model call at the established CharXiv reasoning budget.  With
    # thinking enabled, smaller planner/step budgets can exhaust tokens before
    # the guided JSON body is emitted and would confound H2 with truncation.
    parser.add_argument("--plan-max-tokens", type=int, default=12288)
    parser.add_argument("--step-max-tokens", type=int, default=12288)
    parser.add_argument("--single-max-tokens", type=int, default=12288)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-wait", type=float, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--freeze-existing", action="store_true")
    parser.add_argument("--verify-frozen", action="store_true")
    parser.add_argument("--freeze-protocol", action="store_true")
    parser.add_argument("--verify-protocol", action="store_true")
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    parser.add_argument("--require-frozen-protocol", action="store_true")
    parser.add_argument("--export-review-template-from", type=Path)
    parser.add_argument("--review-template-output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = mode_output_dir(args.model_name, args.mode, args.output_dir)
    management_modes = sum(
        bool(value)
        for value in (
            args.freeze_existing,
            args.verify_frozen,
            args.freeze_protocol,
            args.verify_protocol,
        )
    )
    if management_modes > 1:
        raise ValueError("freeze/verify management operations are mutually exclusive")
    if args.freeze_protocol:
        freeze_protocol(args.protocol_dir)
        return 0
    if args.verify_protocol:
        return 0 if verify_protocol(args.protocol_dir)["valid"] else 2
    if args.freeze_existing:
        freeze_run_directory(
            output_dir,
            reason=f"Completed {args.model_name} {args.mode} {H2_VERSION} run.",
        )
        return 0
    if args.verify_frozen:
        return 0 if verify_frozen_directory(output_dir)["valid"] else 2
    if args.export_review_template_from:
        if not args.allow_unfrozen_plans:
            assert_file_is_frozen(args.export_review_template_from)
        source_parent = args.export_review_template_from.parent
        default_review_name = safe_name(
            f"{source_parent.parent.name}__{source_parent.name}__reviewed_plans.template.json"
        )
        output = args.review_template_output or (DEFAULT_REVIEW_DIR / default_review_name)
        if output.resolve().is_relative_to(source_parent.resolve()):
            raise ValueError("review template output must not modify the frozen plan directory")
        export_review_template(args.export_review_template_from, output)
        return 0
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")
    if args.plan_only and args.mode != "predicted-plan":
        raise ValueError("--plan-only requires --mode predicted-plan")
    if args.plan_only and args.plans_from:
        raise ValueError("--plan-only generates plans and cannot be combined with --plans-from")
    if args.formal and args.smoke:
        raise ValueError("--formal and --smoke are mutually exclusive")
    formal_indices = load_formal_indices(args.annotation_path)
    selected_indices = list(
        DEFAULT_SMOKE_INDICES
        if args.smoke
        else (args.indices if args.indices is not None else (formal_indices if args.formal else []))
    )
    if not selected_indices:
        raise ValueError("provide --indices or --smoke")
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError("duplicate indices are not allowed")
    if args.formal and selected_indices != formal_indices:
        raise ValueError("--formal requires all 41 annotation_1 indices in frozen order")
    if args.formal and sha256_file(args.annotation_path) != sha256_file(DEFAULT_ANNOTATION_PATH):
        raise ValueError("--formal annotation file differs from the protocol annotation_1.json")
    if args.formal and args.allow_unfrozen_plans:
        raise ValueError("--formal forbids --allow-unfrozen-plans")
    if args.formal and args.allow_engineering_review:
        raise ValueError("--formal forbids --allow-engineering-review")
    api_base = resolve_api_base(args.model_name, args.api_base)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"H2 output directory is not empty: {output_dir}; use --resume or a new directory"
        )
    if output_dir.resolve().is_relative_to((SCRIPT_DIR / "test_case/RIG_Charxiv/runs").resolve()):
        raise ValueError("H2 output directory must not be inside the existing single-pass runs root")
    assert_run_directory_writable(output_dir, "run or resume H2 inference")

    protocol_spec = build_protocol_spec()
    protocol_sha256 = canonical_json_sha256(protocol_spec)
    protocol_frozen = False
    if (args.protocol_dir / FREEZE_MARKER).is_file():
        protocol_frozen = verify_protocol(args.protocol_dir)["valid"]
    if args.require_frozen_protocol and not protocol_frozen:
        raise ValueError("--require-frozen-protocol was set but protocol verification failed")
    if args.formal and not protocol_frozen:
        raise ValueError("--formal requires a valid frozen protocol")

    single_source_path = args.single_source or default_single_source(args.model_name)
    if args.mode == "single" and not single_source_path.is_file():
        raise FileNotFoundError(f"single-pass source not found: {single_source_path}")
    if args.mode == "reviewed-plan" and (not args.reviewed_plans or not args.plans_from):
        raise ValueError("reviewed-plan requires both --reviewed-plans and --plans-from")
    if args.plans_from and args.mode not in ("predicted-plan", "reviewed-plan"):
        raise ValueError("--plans-from is only valid in predicted-plan or reviewed-plan mode")
    if args.repeat_counts_from and args.mode != "repeated-single":
        raise ValueError("--repeat-counts-from is only valid in repeated-single mode")
    predicted_plans = (
        load_predicted_plans(
            args.plans_from, require_frozen=not args.allow_unfrozen_plans
        )
        if args.plans_from
        else {}
    )
    reviewed_plans = (
        load_reviewed_plans(
            args.reviewed_plans,
            predicted_plan_path=args.plans_from,
            predicted_plans=predicted_plans,
            allow_engineering_review=args.allow_engineering_review,
        )
        if args.mode == "reviewed-plan"
        else {}
    )
    repeat_count_plans = (
        load_predicted_plans(
            args.repeat_counts_from,
            require_frozen=not args.allow_unfrozen_plans,
        )
        if args.repeat_counts_from
        else {}
    )
    repeat_counts = {
        index: len(plan["steps"]) for index, plan in repeat_count_plans.items()
    }
    missing_repeat_counts = sorted(set(selected_indices) - set(repeat_counts)) if repeat_counts else []
    if missing_repeat_counts:
        raise ValueError(f"repeat-count source is missing indices: {missing_repeat_counts}")
    missing_predicted_plans = (
        sorted(set(selected_indices) - set(predicted_plans)) if predicted_plans else []
    )
    if missing_predicted_plans:
        raise ValueError(f"predicted plan source is missing indices: {missing_predicted_plans}")

    dataframe = pd.read_csv(args.tsv_path, sep="\t")
    selected = dataframe[dataframe["index"].astype(int).isin(selected_indices)].copy()
    found = {int(value) for value in selected["index"]}
    missing = sorted(set(selected_indices) - found)
    if missing:
        raise ValueError(f"indices missing from TSV: {missing}")
    selected["_order"] = selected["index"].astype(int).map(
        {index: position for position, index in enumerate(selected_indices)}
    )
    selected = selected.sort_values("_order")
    input_image_sha256 = {
        str(int(row["index"])): source_image_sha256(row, args.image_dir)
        for _, row in selected.iterrows()
    }

    config = {
        "experiment_version": H2_VERSION,
        "run_scope": "formal" if args.formal else "calibration",
        "mode": args.mode,
        "plan_only": args.plan_only,
        "model_name": args.model_name,
        "selected_indices": selected_indices,
        "question_source_policy": "prefer_raw_question_then_question",
        "thinking_disabled": False,
        "temperature_explicitly_set": False,
        "guided_json": True,
        "protocol_sha256": protocol_sha256,
        "protocol_frozen": protocol_frozen,
        "protocol_directory": str(args.protocol_dir.resolve()),
        "repeats": args.repeats if args.mode == "repeated-single" else None,
        "repeat_counts_from": str(args.repeat_counts_from.resolve()) if args.repeat_counts_from else None,
        "repeat_counts_from_sha256": sha256_file(args.repeat_counts_from) if args.repeat_counts_from else None,
        "plans_from": str(args.plans_from.resolve()) if args.plans_from else None,
        "plans_from_sha256": sha256_file(args.plans_from) if args.plans_from else None,
        "plan_max_tokens": args.plan_max_tokens,
        "step_max_tokens": args.step_max_tokens,
        "single_max_tokens": args.single_max_tokens,
        "api_base": api_base,
        "tsv_path": str(args.tsv_path.resolve()),
        "tsv_sha256": sha256_file(args.tsv_path),
        "image_dir": str(args.image_dir.resolve()),
        "input_image_sha256": input_image_sha256,
        "annotation_path": str(args.annotation_path.resolve()),
        "annotation_sha256": sha256_file(args.annotation_path),
        "single_source": str(single_source_path.resolve()) if args.mode == "single" else None,
        "single_source_sha256": sha256_file(single_source_path) if args.mode == "single" else None,
        "reviewed_plans": str(args.reviewed_plans.resolve()) if args.reviewed_plans else None,
        "reviewed_plans_sha256": sha256_file(args.reviewed_plans) if args.reviewed_plans else None,
        "created_at": utc_now(),
    }
    if args.dry_run:
        print(json.dumps({**config, "output_dir": str(output_dir.resolve())}, ensure_ascii=False, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_single_source_path = single_source_path
    if args.mode == "single":
        runtime_single_source_path = output_dir / "inputs" / "single_source.json"
        runtime_single_source_path.parent.mkdir(parents=True, exist_ok=True)
        if not runtime_single_source_path.is_file():
            shutil.copyfile(single_source_path, runtime_single_source_path)
        if sha256_file(runtime_single_source_path) != config["single_source_sha256"]:
            raise ValueError("single-source snapshot hash differs from the configured source")
        config["single_source_snapshot"] = str(runtime_single_source_path.resolve())
    config_path = output_dir / "experiment_config.json"
    plans_path = output_dir / "plans.json"
    steps_path = output_dir / "step_outputs.json"
    results_path = output_dir / "rig_results.json"
    status_path = output_dir / "run_status.json"
    if args.resume and config_path.is_file():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        for field in (
            "experiment_version",
            "run_scope",
            "mode",
            "plan_only",
            "model_name",
            "selected_indices",
            "protocol_sha256",
            "protocol_frozen",
            "annotation_sha256",
            "tsv_sha256",
            "input_image_sha256",
            "plans_from_sha256",
            "reviewed_plans_sha256",
            "repeat_counts_from_sha256",
            "plan_max_tokens",
            "step_max_tokens",
            "single_max_tokens",
        ):
            if existing_config.get(field) != config.get(field):
                raise ValueError(f"cannot resume with changed {field}")
        if existing_config.get("api_base") != api_base:
            existing_status_path = output_dir / "run_status.json"
            existing_status = (
                json.loads(existing_status_path.read_text(encoding="utf-8"))
                if existing_status_path.is_file()
                else {}
            )
            if existing_status.get("completed_indices"):
                raise ValueError("cannot change --api-base after any sample has completed")
            config["created_at"] = existing_config.get("created_at", config["created_at"])
            config["configuration_changes"] = [
                *existing_config.get("configuration_changes", []),
                {
                    "changed_at": utc_now(),
                    "field": "api_base",
                    "old_value": existing_config.get("api_base"),
                    "new_value": api_base,
                    "reason": "endpoint correction before any sample completed",
                },
            ]
            save_json_atomic(config_path, config)
    elif not config_path.exists():
        save_json_atomic(config_path, config)

    plans = json.loads(plans_path.read_text(encoding="utf-8")) if plans_path.is_file() else []
    if recover_plan_records(plans):
        save_json_atomic(plans_path, plans)
    step_outputs = json.loads(steps_path.read_text(encoding="utf-8")) if steps_path.is_file() else []
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.is_file() else []
    plan_by_index = {int(row["index"]): row for row in plans}
    steps_by_index: Dict[int, List[Dict[str, Any]]] = {}
    for row in step_outputs:
        steps_by_index.setdefault(int(row["index"]), []).append(row)
    result_by_index = {int(row["index"]): row for row in results}
    completed_indices = (
        {index for index, row in plan_by_index.items() if row.get("status") != "request_failed"}
        if args.plan_only
        else {
            index for index, row in result_by_index.items() if row.get("status") != "request_failed"
        }
    )
    single_source = (
        load_single_source(runtime_single_source_path) if args.mode == "single" else {}
    )
    client = OpenAI(api_key=args.api_key, base_url=api_base, timeout=args.timeout)

    status = {
        "experiment_version": H2_VERSION,
        "mode": args.mode,
        "plan_only": args.plan_only,
        "model_name": args.model_name,
        "requested_indices": selected_indices,
        "completed_indices": sorted(completed_indices),
        "request_failed_indices": [],
        "updated_at": utc_now(),
        "complete": False,
        "protocol_complete": False,
    }
    save_json_atomic(status_path, status)

    for _, row in selected.iterrows():
        index = int(row["index"])
        if index in completed_indices:
            print(f"[{index}] already complete; skipped")
            continue
        question, question_field = load_question(row)
        category = int(row["inst_category"])
        print(f"[{index}] {args.mode} ({CATEGORY_NAMES.get(category, category)})")

        if args.mode == "single":
            source = single_source.get(index)
            elements = list(source.get("elements", [])) if source else []
            step_records = [
                {
                    "index": index,
                    "step_id": "single",
                    "call_id": "source_single_pass",
                    "evidence_need": "Ground all necessary evidence jointly.",
                    "status": "valid" if elements else "source_missing_or_failed",
                    "raw_output": source.get("model_raw_output", "") if source else "",
                    "reasoning_content": source.get("reasoning_content") if source else None,
                    "finish_reason": source.get("finish_reason") if source else None,
                    "elements": elements,
                    "token_usage": normalize_token_usage(source.get("token_usage", {})) if source else {},
                    "elapsed_seconds": source.get("elapsed_seconds") if source else None,
                    "elapsed_seconds_available": bool(
                        source and source.get("elapsed_seconds") is not None
                    ),
                    "source_result": str(single_source_path.resolve()),
                    "source_snapshot": str(runtime_single_source_path.resolve()),
                }
            ]
            plan_record = {
                "index": index,
                "question": question,
                "plan_source": "single_pass_import",
                "status": "not_applicable",
                "steps": plan_for_mode(
                    mode=args.mode,
                    index=index,
                    question=question,
                    repeats=args.repeats,
                    repeat_counts=repeat_counts,
                    reviewed_plans=reviewed_plans,
                ),
                "token_usage": {"prompt": 0, "completion": 0, "total": 0},
                "elapsed_seconds": 0.0,
            }
        else:
            steps = plan_for_mode(
                mode=args.mode,
                index=index,
                question=question,
                repeats=args.repeats,
                repeat_counts=repeat_counts,
                reviewed_plans=reviewed_plans,
            )
            if args.mode == "predicted-plan":
                if args.plans_from:
                    source_plan = predicted_plans[index]
                    if str(source_plan.get("question", "")) != question:
                        raise ValueError(f"frozen predicted plan question mismatch for index {index}")
                    steps = list(source_plan["steps"])
                    plan_record = {
                        "index": index,
                        "question": question,
                        "plan_source": "frozen_predicted",
                        "status": source_plan.get("status"),
                        "steps": steps,
                        "source_plans": str(args.plans_from.resolve()),
                        "source_plans_sha256": sha256_file(args.plans_from),
                        "source_plan_token_usage": source_plan.get("token_usage", {}),
                        "source_plan_elapsed_seconds": source_plan.get("elapsed_seconds"),
                        "token_usage": {"prompt": 0, "completion": 0, "total": 0},
                        "elapsed_seconds": 0.0,
                    }
                else:
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
                        plan_record = {
                            "index": index,
                            "question": question,
                            "plan_source": "predicted",
                            "status": "request_failed",
                            "steps": [],
                            "error": error,
                            "attempts": attempts,
                            "elapsed_seconds": elapsed,
                            "token_usage": {},
                        }
                        steps = None
                    else:
                        metadata = response_metadata(response, elapsed)
                        steps = parse_plan_response(metadata["raw_output"])
                        plan_status = (
                            "valid"
                            if steps
                            else "truncated"
                            if metadata["finish_reason"] == "length"
                            else "parse_failed"
                        )
                        plan_record = {
                            "index": index,
                            "question": question,
                            "plan_source": "predicted",
                            "status": plan_status,
                            "steps": steps or [],
                            "attempts": attempts,
                            **metadata,
                        }
                        raw_path = output_dir / "raw" / f"q{index:04d}" / "plan.txt"
                        raw_path.parent.mkdir(parents=True, exist_ok=True)
                        raw_path.write_text(metadata["raw_output"], encoding="utf-8")
                        plan_record["raw_output_path"] = str(raw_path.resolve())
            else:
                reviewed = reviewed_plans.get(index, {})
                plan_record = {
                    "index": index,
                    "question": question,
                    "plan_source": "repeated_single_control" if args.mode == "repeated-single" else "reviewed",
                    "status": "valid" if steps else "source_plan_failed",
                    "reviewer": reviewed.get("reviewer"),
                    "review_notes": reviewed.get("review_notes"),
                    "steps": steps,
                    "token_usage": {"prompt": 0, "completion": 0, "total": 0},
                    "elapsed_seconds": 0.0,
                }

            if args.plan_only:
                plan_by_index[index] = plan_record
                plans = [plan_by_index[key] for key in sorted(plan_by_index)]
                save_json_atomic(plans_path, plans)
                if plan_record.get("status") != "request_failed":
                    completed_indices.add(index)
                status["completed_indices"] = sorted(completed_indices)
                status["request_failed_indices"] = sorted(
                    plan_index
                    for plan_index, entry in plan_by_index.items()
                    if entry.get("status") == "request_failed"
                )
                status["updated_at"] = utc_now()
                save_json_atomic(status_path, status)
                print(
                    f"  plan_status={plan_record.get('status')}, "
                    f"steps={len(plan_record.get('steps', []))}"
                )
                continue

            image = load_image(row, args.image_dir)
            image_uri, original_size = encode_image(image)
            step_records = []
            if steps:
                for position, step in enumerate(steps, start=1):
                    step_id = step["step_id"]
                    call_id = f"{step_id}_{position}"
                    if args.mode == "repeated-single":
                        prompt = RIG_USER_PROMPT_TEMPLATE.replace("__QUESTION_PLACEHOLDER__", question)
                        messages = build_rig_messages(prompt, image_uri)
                        max_tokens = args.single_max_tokens
                    else:
                        messages = step_messages(question, step["evidence_need"], image_uri)
                        max_tokens = args.step_max_tokens
                    response, elapsed, error, attempts = call_model(
                        client,
                        model_name=args.model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        extra_body=(
                            build_rig_extra_body()
                            if args.mode == "repeated-single"
                            else {"guided_json": STEP_JSON_SCHEMA}
                        ),
                        retries=args.retries,
                        retry_wait=args.retry_wait,
                    )
                    if response is None:
                        record = {
                            "index": index,
                            "step_id": step_id,
                            "call_id": call_id,
                            "evidence_need": step["evidence_need"],
                            "status": "request_failed",
                            "error": error,
                            "attempts": attempts,
                            "elapsed_seconds": elapsed,
                            "token_usage": {},
                            "elements": [],
                        }
                    else:
                        metadata = response_metadata(response, elapsed)
                        elements = parse_rig_response(metadata["raw_output"]) or []
                        record_status = (
                            "valid"
                            if elements
                            else "truncated"
                            if metadata["finish_reason"] == "length"
                            else "parse_failed"
                        )
                        record = {
                            "index": index,
                            "step_id": step_id,
                            "call_id": call_id,
                            "evidence_need": step["evidence_need"],
                            "status": record_status,
                            "attempts": attempts,
                            "elements": elements,
                            **metadata,
                        }
                        raw_path = output_dir / "raw" / f"q{index:04d}" / f"{call_id}.txt"
                        raw_path.parent.mkdir(parents=True, exist_ok=True)
                        raw_path.write_text(metadata["raw_output"], encoding="utf-8")
                        record["raw_output_path"] = str(raw_path.resolve())
                    step_records.append(record)
                    steps_by_index[index] = step_records
                    step_outputs = [
                        record
                        for sample_index in sorted(steps_by_index)
                        for record in steps_by_index[sample_index]
                    ]
                    save_json_atomic(steps_path, step_outputs)

        plan_by_index[index] = plan_record
        plans = [plan_by_index[key] for key in sorted(plan_by_index)]
        save_json_atomic(plans_path, plans)
        steps_by_index[index] = step_records
        step_outputs = [
            record
            for sample_index in sorted(steps_by_index)
            for record in steps_by_index[sample_index]
        ]
        save_json_atomic(steps_path, step_outputs)

        union_elements, union_stats = union_step_elements(step_records)
        request_failed = plan_record.get("status") == "request_failed" or any(
            record.get("status") == "request_failed" for record in step_records
        )
        sample_status = (
            "request_failed"
            if request_failed
            else "valid"
            if union_elements
            else "model_failed"
        )
        all_usage_records = [plan_record, *step_records]
        result = {
            "index": index,
            "model_name": args.model_name,
            "prompt_version": H2_VERSION,
            "h2_mode": args.mode,
            "inst_category": category,
            "category_name": CATEGORY_NAMES.get(category, f"Unknown({category})"),
            "question": question,
            "question_source_field": question_field,
            "dataset_question": str(row.get("question", "")),
            "gt_answer": str(row.get("answer", "")),
            "status": sample_status,
            "plan_status": plan_record.get("status"),
            "plan_steps": plan_record.get("steps", []),
            "elements": union_elements,
            "union_statistics": union_stats,
            "token_usage": sum_tokens(all_usage_records),
            "elapsed_seconds": sum(
                float(record.get("elapsed_seconds") or 0.0) for record in all_usage_records
            ),
            "elapsed_seconds_complete": all(
                record.get("elapsed_seconds") is not None for record in step_records
            ),
            "step_statuses": [
                {
                    "step_id": record["step_id"],
                    "status": record["status"],
                    "element_count": len(record.get("elements", [])),
                }
                for record in step_records
            ],
        }
        result_by_index[index] = result
        results = [result_by_index[key] for key in sorted(result_by_index)]
        save_json_atomic(results_path, results)

        if args.mode != "single":
            visualization_path = output_dir / "visualizations" / f"q{index:04d}_union.jpg"
            draw_bboxes_on_image(image, union_elements, original_size, visualization_path, show=False)
        print(
            f"  status={sample_status}, steps={len(step_records)}, "
            f"raw_boxes={union_stats['raw_element_count']}, union_boxes={len(union_elements)}"
        )

        if sample_status != "request_failed":
            completed_indices.add(index)
        status["completed_indices"] = sorted(completed_indices)
        status["request_failed_indices"] = sorted(
            result_index
            for result_index, entry in result_by_index.items()
            if entry.get("status") == "request_failed"
        )
        status["updated_at"] = utc_now()
        save_json_atomic(status_path, status)

    status["completed_indices"] = sorted(completed_indices)
    terminal_by_index = plan_by_index if args.plan_only else result_by_index
    status["request_failed_indices"] = sorted(
        index for index, entry in terminal_by_index.items() if entry.get("status") == "request_failed"
    )
    status["requested_samples"] = len(selected_indices)
    status["completed_samples"] = len(set(selected_indices) & completed_indices)
    status["request_failed_samples"] = len(status["request_failed_indices"])
    status["valid_samples"] = sum(
        terminal_by_index.get(index, {}).get("status") == "valid" for index in selected_indices
    )
    status["model_failed_samples"] = sum(
        terminal_by_index.get(index, {}).get("status")
        in ({"truncated", "parse_failed"} if args.plan_only else {"model_failed"})
        for index in selected_indices
    )
    status["complete"] = (
        set(selected_indices).issubset(completed_indices) and not status["request_failed_indices"]
    )
    # A fixed-budget planner failure is a terminal model outcome, not a reason to
    # exclude the question or regenerate its plan. Formal freezes therefore retain
    # truncated/parse-failed plans as zero-step failures without selection bias.
    status["protocol_complete"] = status["complete"]
    status["finished_at"] = utc_now()
    save_json_atomic(status_path, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare, validate, freeze, and verify a blind human image-aware plan@6 packet.

The authoring packet contains only the five source images and their questions.
It deliberately excludes annotations, answers, model predictions, and earlier
plans.  A separate frozen plan-only directory is produced only after an
independent author completes six concrete evidence needs per question and signs
the information-access attestation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
VERSION = "h2-human-image-aware-plan-at6-v1"
INDICES = (1, 17, 8, 244, 346)
K = 6
DEFAULT_TSV = Path.home() / "LMUData/CharXiv_reasoning_val.tsv"
DEFAULT_IMAGE_DIR = Path.home() / "LMUData/images/CharXiv_reasoning_val"
DEFAULT_ROOT = SCRIPT_DIR / "test_case/RIG_Charxiv/h2_grounding/feasibility/adaptive-gate"
DEFAULT_AUTHORING_DIR = DEFAULT_ROOT / "Qwen3.5-27B__human-image-aware-plan-at6__authoring"
DEFAULT_FROZEN_DIR = DEFAULT_ROOT / "Qwen3.5-27B__human-image-aware-plan-at6__plan-only"
MARKER = "PLAN_FROZEN.json"

ATTESTATION_FIELDS = (
    "viewed_only_packet_question_and_image",
    "did_not_view_gt_annotation",
    "did_not_view_reference_answer",
    "did_not_view_model_predictions_or_prior_plans",
    "did_not_receive_target_bbox_or_annotation_description",
)

PROHIBITED_RESULT_TERMS = (
    "final answer",
    "the answer is",
    "therefore",
    "thus the",
    "select the",
    "choose the",
    "return the answer",
    "with the highest",
    "with the second highest",
    "the greatest increase",
    "the first peak",
    "the second peak",
)


def atomic_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_questions(tsv_path: Path, indices: Sequence[int]) -> Dict[int, Dict[str, str]]:
    rows: Dict[int, Dict[str, str]] = {}
    # CharXiv TSV rows may contain an embedded base64 image field even though
    # this packet only reads the question columns.
    csv.field_size_limit(1024 * 1024 * 1024)
    with Path(tsv_path).open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            try:
                index = int(row.get("index", ""))
            except ValueError:
                continue
            if index not in indices:
                continue
            raw_question = str(row.get("raw_question") or "").strip()
            question = raw_question or str(row.get("question") or "").strip()
            source = "raw_question" if raw_question else "question"
            if not question:
                raise ValueError(f"question is empty for index {index}")
            rows[index] = {"question": question, "question_source": source}
    missing = sorted(set(indices) - set(rows))
    if missing:
        raise ValueError(f"TSV is missing requested indices: {missing}")
    return rows


def source_image(image_dir: Path, index: int) -> Path:
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = Path(image_dir) / f"{index}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no source image found for index {index} in {image_dir}")


def authoring_policy() -> Dict[str, Any]:
    return {
        "condition": "human-image-aware-plan@6",
        "allowed_inputs": ["question text", "unaltered source image"],
        "forbidden_inputs": [
            "GT annotations or annotation descriptions",
            "reference answer or model answer",
            "model bbox predictions or grounding visualizations",
            "static/repeated/oracle plans or outputs",
        ],
        "fixed_visual_grounding_calls_per_question": K,
        "requirements": [
            "Exactly six ordered and non-duplicate evidence needs.",
            "Each need names a concrete visible object or homogeneous visible set that can be bounded.",
            "Use image-specific identities, counts, panels, series, labels, marks, or values when visibly available.",
            "Prioritize expected marginal evidence value; do not spend a step only requesting the full chart panel.",
            "Request inputs for comparison/counting, never the derived comparison, count, selection, or answer.",
            "Do not include bbox coordinates; the downstream grounder predicts them.",
        ],
    }


def create_packet(tsv_path: Path, image_dir: Path, output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"authoring directory is not empty: {output_dir}")
    questions = load_questions(tsv_path, INDICES)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    context_rows: List[Dict[str, Any]] = []
    plan_rows: List[Dict[str, Any]] = []
    for index in INDICES:
        source = source_image(image_dir, index)
        destination = images_dir / f"q{index:04d}{source.suffix.lower()}"
        shutil.copyfile(source, destination)
        context = {
            "index": index,
            **questions[index],
            "packet_image": destination.relative_to(output_dir).as_posix(),
            "source_image_sha256": sha256_file(source),
            "packet_image_sha256": sha256_file(destination),
        }
        if context["source_image_sha256"] != context["packet_image_sha256"]:
            raise RuntimeError(f"copied image hash mismatch for index {index}")
        context_rows.append(context)
        plan_rows.append(
            {
                "index": index,
                "question": context["question"],
                "steps": [
                    {"step_id": f"s{position}", "evidence_need": ""}
                    for position in range(1, K + 1)
                ],
                "author_notes": "",
            }
        )
    context_payload = {
        "format": VERSION,
        "packet_type": "blind-human-authoring-context",
        "policy": authoring_policy(),
        "samples": context_rows,
    }
    template = {
        "format": VERSION,
        "condition": "human-image-aware-plan@6",
        "context_sha256": canonical_sha256(context_payload),
        "author": {
            "anonymous_id": "",
            "attestation": {field: False for field in ATTESTATION_FIELDS},
        },
        "plans": plan_rows,
    }
    atomic_json(output_dir / "authoring_context.json", context_payload)
    atomic_json(output_dir / "human_plans.json", template)
    (output_dir / "AUTHORING_INSTRUCTIONS.txt").write_text(
        "Open authoring_context.json and only the images in ./images.\n"
        "Fill all 30 evidence_need fields in human_plans.json, add an anonymous author ID,\n"
        "and set every attestation field to true only if it is factually correct.\n"
        "Do not open annotations, answers, predictions, visualizations, or previous plans.\n",
        encoding="utf-8",
    )
    print(f"Authoring packet: {output_dir.resolve()}")


def normalize_need(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def validate_plan(plan_path: Path, context_path: Path) -> Dict[str, Any]:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    errors: List[str] = []
    warnings: List[str] = []
    if plan.get("format") != VERSION or context.get("format") != VERSION:
        errors.append("format/version mismatch")
    if plan.get("context_sha256") != canonical_sha256(context):
        errors.append("authoring context hash mismatch")
    author = plan.get("author") or {}
    if not normalize_need(author.get("anonymous_id")):
        errors.append("author.anonymous_id is required")
    attestation = author.get("attestation") or {}
    for field in ATTESTATION_FIELDS:
        if attestation.get(field) is not True:
            errors.append(f"author attestation must be true: {field}")
    context_by_index = {int(row["index"]): row for row in context.get("samples", [])}
    plans = plan.get("plans")
    if not isinstance(plans, list):
        errors.append("plans must be a list")
        plans = []
    plan_indices = [int(row.get("index", -1)) for row in plans if isinstance(row, dict)]
    if plan_indices != list(INDICES):
        errors.append(f"plan order must be exactly {list(INDICES)}")
    normalized_plans: List[Dict[str, Any]] = []
    for row in plans:
        if not isinstance(row, dict):
            errors.append("each plan must be an object")
            continue
        index = int(row.get("index", -1))
        expected = context_by_index.get(index)
        if expected is None:
            errors.append(f"unexpected plan index {index}")
            continue
        if str(row.get("question", "")) != expected["question"]:
            errors.append(f"q{index}: question differs from blind context")
        steps = row.get("steps")
        if not isinstance(steps, list) or len(steps) != K:
            errors.append(f"q{index}: exactly {K} steps are required")
            continue
        normalized_steps: List[Dict[str, str]] = []
        for position, step in enumerate(steps, start=1):
            if not isinstance(step, dict) or step.get("step_id") != f"s{position}":
                errors.append(f"q{index}: step {position} must have step_id=s{position}")
                continue
            need = normalize_need(step.get("evidence_need"))
            if not need:
                errors.append(f"q{index}/s{position}: evidence_need is empty")
            if len(need) > 320:
                errors.append(f"q{index}/s{position}: evidence_need exceeds 320 characters")
            lowered = need.casefold()
            for term in PROHIBITED_RESULT_TERMS:
                if term in lowered:
                    errors.append(f"q{index}/s{position}: result-oriented phrase is prohibited: {term}")
            if "bbox" in lowered or "bounding box" in lowered:
                warnings.append(f"q{index}/s{position}: describe visible evidence, not a bbox")
            if "main chart panel" in lowered or "entire chart" in lowered or "whole chart" in lowered:
                warnings.append(f"q{index}/s{position}: likely over-broad panel request")
            if lowered.startswith("the coordinate (") and "on the plot" in lowered:
                warnings.append(
                    f"q{index}/s{position}: target coordinate is derived unless an explicit marker is visible; "
                    "request its tick-aligned grid/reference lines instead"
                )
            normalized_steps.append({"step_id": f"s{position}", "evidence_need": need})
        if len({step["evidence_need"].casefold() for step in normalized_steps if step["evidence_need"]}) != len(
            [step for step in normalized_steps if step["evidence_need"]]
        ):
            errors.append(f"q{index}: duplicate evidence needs are not allowed")
        normalized_plans.append(
            {
                "index": index,
                "question": expected["question"],
                "question_source": expected["question_source"],
                "status": "valid" if len(normalized_steps) == K and all(s["evidence_need"] for s in normalized_steps) else "invalid",
                "steps": normalized_steps,
                "author_notes": normalize_need(row.get("author_notes")),
            }
        )
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "normalized_plans": normalized_plans,
        "author": author,
        "context_sha256": canonical_sha256(context),
        "plan_source_sha256": sha256_file(plan_path),
    }


def manifest(directory: Path) -> List[Dict[str, Any]]:
    return [
        {
            "path": path.relative_to(directory).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != MARKER
    ]


def freeze_plan(plan_path: Path, context_path: Path, output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"frozen plan directory is not empty: {output_dir}")
    validation = validate_plan(plan_path, context_path)
    if not validation["valid"]:
        raise ValueError("plan validation failed:\n- " + "\n- ".join(validation["errors"]))
    if validation["warnings"]:
        raise ValueError("resolve plan warnings before freezing:\n- " + "\n- ".join(validation["warnings"]))
    output_dir.mkdir(parents=True, exist_ok=False)
    context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    plans = validation["normalized_plans"]
    atomic_json(output_dir / "plans.json", plans)
    atomic_json(
        output_dir / "experiment_config.json",
        {
            "experiment_version": VERSION,
            "mode": "human-image-aware-plan",
            "budget_k": K,
            "plan_only": True,
            "selected_indices": list(INDICES),
            "planner_image_access": True,
            "planner_source": "independent human",
            "allowed_inputs": authoring_policy()["allowed_inputs"],
            "forbidden_inputs": authoring_policy()["forbidden_inputs"],
            "author_anonymous_id": validation["author"]["anonymous_id"],
            "author_attestation": validation["author"]["attestation"],
            "context_sha256": validation["context_sha256"],
            "plan_source_sha256": validation["plan_source_sha256"],
            "input_image_sha256": {
                str(row["index"]): row["source_image_sha256"] for row in context["samples"]
            },
        },
    )
    atomic_json(
        output_dir / "run_status.json",
        {
            "experiment_version": VERSION,
            "mode": "human-image-aware-plan",
            "plan_only": True,
            "requested_indices": list(INDICES),
            "completed_indices": list(INDICES),
            "valid_plan_samples": len(INDICES),
            "request_failed_samples": 0,
            "complete": True,
        },
    )
    marker = {
        "format": "H2 human-image-aware plan freeze v1",
        "reason": "Completed blind human image-aware plan@6 for five-question adaptive-gate calibration.",
        "files": manifest(output_dir),
    }
    atomic_json(output_dir / MARKER, marker)
    print(f"Frozen human plan: {output_dir.resolve()}")


def verify_frozen(output_dir: Path) -> Dict[str, Any]:
    marker_path = output_dir / MARKER
    if not marker_path.is_file():
        return {"valid": False, "checked_files": 0, "mismatches": [f"missing {MARKER}"]}
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {row["path"]: row for row in marker.get("files", [])}
    actual = {row["path"]: row for row in manifest(output_dir)}
    mismatches: List[str] = []
    for path in sorted(set(expected) | set(actual)):
        if path not in expected:
            mismatches.append(f"unexpected file: {path}")
        elif path not in actual:
            mismatches.append(f"missing file: {path}")
        elif expected[path] != actual[path]:
            mismatches.append(f"changed file: {path}")
    result = {"valid": not mismatches, "checked_files": len(expected), "mismatches": mismatches}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--tsv-path", type=Path, default=DEFAULT_TSV)
    create.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    create.add_argument("--output-dir", type=Path, default=DEFAULT_AUTHORING_DIR)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--plan-file", type=Path, default=DEFAULT_AUTHORING_DIR / "human_plans.json")
    validate.add_argument("--context-file", type=Path, default=DEFAULT_AUTHORING_DIR / "authoring_context.json")
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--plan-file", type=Path, default=DEFAULT_AUTHORING_DIR / "human_plans.json")
    freeze.add_argument("--context-file", type=Path, default=DEFAULT_AUTHORING_DIR / "authoring_context.json")
    freeze.add_argument("--output-dir", type=Path, default=DEFAULT_FROZEN_DIR)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--output-dir", type=Path, default=DEFAULT_FROZEN_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "create":
        create_packet(args.tsv_path, args.image_dir, args.output_dir)
        return 0
    if args.command == "validate":
        result = validate_plan(args.plan_file, args.context_file)
        print(json.dumps({key: value for key, value in result.items() if key != "normalized_plans"}, ensure_ascii=False, indent=2))
        return 0 if result["valid"] and not result["warnings"] else 2
    if args.command == "freeze":
        freeze_plan(args.plan_file, args.context_file, args.output_dir)
        return 0
    if args.command == "verify":
        return 0 if verify_frozen(args.output_dir)["valid"] else 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

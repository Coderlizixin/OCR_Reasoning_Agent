"""Five-sample GT evidence-need oracle grounding feasibility experiment.

The model receives the question, image, and one GT annotation description, but
never the GT bbox.  Each request therefore measures whether the current
grounder can localize a known-correct semantic evidence need.  Outputs are kept
separate from every frozen H1/H2 run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import httpx
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

from rig_eval.core import BBox, EvaluationConfig, load_annotations
from test_H2_Multistep_Grounding_on_Charxiv import (
    DEFAULT_ANNOTATION_PATH,
    DEFAULT_IMAGE_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_SMOKE_INDICES,
    DEFAULT_TSV_PATH,
    call_model,
    encode_image,
    extract_json_objects,
    load_image,
    load_question,
    resolve_api_base,
    response_metadata,
    save_json_atomic,
    sha256_file,
    source_image_sha256,
    utc_now,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ORACLE_VERSION = "gt-evidence-need-oracle-v1"
DEFAULT_OUTPUT_ROOT = (
    SCRIPT_DIR / "test_case/RIG_Charxiv/h2_grounding/feasibility/oracle-gt-needs"
)

ORACLE_SYSTEM_PROMPT = """You are a precise visual evidence localizer.
You receive one known-correct evidence need for a reasoning question. Locate only
that requested visible clue in the image. Do not solve the question, choose an
answer, or add related context. Use internal visual reasoning, then return only the
requested JSON object without markdown or exposed analysis."""

ORACLE_USER_TEMPLATE = """IMAGE: A chart or scientific figure.

QUESTION:
__QUESTION_PLACEHOLDER__

KNOWN-CORRECT EVIDENCE NEED:
__EVIDENCE_NEED_PLACEHOLDER__

Locate the visible image element described by KNOWN-CORRECT EVIDENCE NEED.

RULES
- The evidence need is semantically correct, but no target coordinates are given.
- Return only the requested clue, not the answer and not surrounding evidence.
- Usually return one box. Return multiple boxes only if the same requested clue is
  genuinely displayed as disconnected visible parts.
- Box the full visible extent of the requested clue. For a curve, axis, trace, or
  region, include the complete relevant visible segment.
- Do not use the whole image or a very large context box as a shortcut.
- bbox is integer [xmin,ymin,xmax,ymax] normalized to [0,1000].
- Transcribe visible text or values exactly when readable.
- Return at most 4 elements.

OUTPUT ONLY:
{
  "elements": [
    {
      "description": "the localized requested clue",
      "bbox": [xmin, ymin, xmax, ymax]
    }
  ]
}"""

ORACLE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "elements": {
            "type": "array",
            "maxItems": 4,
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


def build_messages(question: str, evidence_need: str, image_data_uri: str) -> List[Dict[str, Any]]:
    prompt = ORACLE_USER_TEMPLATE.replace("__QUESTION_PLACEHOLDER__", question).replace(
        "__EVIDENCE_NEED_PLACEHOLDER__", evidence_need
    )
    return [
        {"role": "system", "content": ORACLE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ],
        },
    ]


def build_extra_body() -> Dict[str, Any]:
    return {
        "guided_json": ORACLE_JSON_SCHEMA,
        "mm_processor_kwargs": {
            "images_kwargs": {
                "size": {
                    "longest_edge": 8192 * 32 * 32,
                    "shortest_edge": 4096 * 32 * 32,
                }
            }
        },
    }


def normalize_elements(value: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(value, list) or len(value) > 4:
        return None
    normalized: List[Dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict):
            return None
        description = str(raw.get("description", "")).strip()
        bbox = raw.get("bbox")
        if not description or not isinstance(bbox, list) or len(bbox) != 4:
            return None
        try:
            box = [int(item) for item in bbox]
        except (TypeError, ValueError):
            return None
        if any(item < 0 or item > 1000 for item in box):
            return None
        if box[0] >= box[2] or box[1] >= box[3]:
            return None
        normalized.append({"description": description[:320], "bbox": box})
    return normalized


def parse_oracle_response(text: str) -> Optional[List[Dict[str, Any]]]:
    for candidate in reversed(extract_json_objects(text or "")):
        elements = normalize_elements(candidate.get("elements"))
        if elements is not None:
            return elements
    return None


def pair_metrics(gt_value: Sequence[float], pred_value: Sequence[float]) -> Dict[str, float]:
    gt = BBox.from_value(gt_value)
    pred = BBox.from_value(pred_value)
    intersection = gt.intersection_area(pred)
    return {
        "iou": gt.iou(pred),
        "gt_coverage": intersection / gt.area,
        "pred_coverage": intersection / pred.area,
        "pred_to_gt_area_ratio": pred.area / gt.area,
    }


def profile_hit(metrics: Mapping[str, float], threshold: float, config: EvaluationConfig) -> bool:
    return bool(
        metrics["iou"] >= threshold
        or (
            metrics["gt_coverage"] >= config.containment_threshold
            and metrics["pred_to_gt_area_ratio"] <= config.max_pred_to_gt_area_ratio
        )
    )


def evaluate_record(record: Mapping[str, Any], config: EvaluationConfig) -> Dict[str, Any]:
    elements = record.get("elements", []) if record.get("status") == "valid" else []
    candidates = [pair_metrics(record["gt_bbox"], item["bbox"]) for item in elements]
    best = max(candidates, key=lambda item: item["iou"], default=None)
    return {
        "index": int(record["index"]),
        "annotation_id": str(record["annotation_id"]),
        "evidence_need": str(record["evidence_need"]),
        "status": str(record["status"]),
        "n_pred": len(elements),
        "best_iou": best["iou"] if best else 0.0,
        "best_gt_coverage": max((item["gt_coverage"] for item in candidates), default=0.0),
        "best_pred_to_gt_area_ratio": best["pred_to_gt_area_ratio"] if best else None,
        "strict_iou_hit_0_3": any(item["iou"] >= 0.3 for item in candidates),
        "strict_iou_hit_0_5": any(item["iou"] >= 0.5 for item in candidates),
        "profile_hit_0_3": any(profile_hit(item, 0.3, config) for item in candidates),
        "profile_hit_0_5": any(profile_hit(item, 0.5, config) for item in candidates),
        "large_box_only": bool(candidates) and all(
            item["pred_to_gt_area_ratio"] > config.max_pred_to_gt_area_ratio
            for item in candidates
        ),
    }


def make_report(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    config = EvaluationConfig(semantic_policy="spatial")
    rows = [evaluate_record(record, config) for record in records]
    by_sample = []
    for index in sorted({int(row["index"]) for row in rows}):
        subset = [row for row in rows if int(row["index"]) == index]
        by_sample.append(
            {
                "index": index,
                "n_targets": len(subset),
                "profile_recall_0_3": mean(float(row["profile_hit_0_3"]) for row in subset),
                "profile_recall_0_5": mean(float(row["profile_hit_0_5"]) for row in subset),
                "strict_iou_recall_0_3": mean(float(row["strict_iou_hit_0_3"]) for row in subset),
                "strict_iou_recall_0_5": mean(float(row["strict_iou_hit_0_5"]) for row in subset),
                "mean_best_iou": mean(float(row["best_iou"]) for row in subset),
                "mean_pred_count": mean(float(row["n_pred"]) for row in subset),
            }
        )
    statuses: Dict[str, int] = {}
    for record in records:
        status = str(record.get("status"))
        statuses[status] = statuses.get(status, 0) + 1
    total = len(rows)
    profile_03 = sum(bool(row["profile_hit_0_3"]) for row in rows)
    profile_05 = sum(bool(row["profile_hit_0_5"]) for row in rows)
    strict_03 = sum(bool(row["strict_iou_hit_0_3"]) for row in rows)
    strict_05 = sum(bool(row["strict_iou_hit_0_5"]) for row in rows)
    token_keys = ("prompt", "completion", "total")
    token_usage = {
        key: sum(
            int(record.get("token_usage", {}).get(key) or 0)
            for record in records
        )
        for key in token_keys
    }
    return {
        "experiment_version": ORACLE_VERSION,
        "evaluation_unit": "one GT evidence-need request",
        "matching": {
            "primary": "IoU >= threshold OR GT coverage >= 0.7 with pred/GT area ratio <= 4",
            "strict_secondary": "IoU only",
            "semantic_policy": "target identity is supplied by the request; spatial localization is scored",
        },
        "summary": {
            "samples": len(by_sample),
            "targets": total,
            "status_counts": statuses,
            "profile_hits_0_3": profile_03,
            "profile_recall_0_3": profile_03 / total if total else 0.0,
            "profile_hits_0_5": profile_05,
            "profile_recall_0_5": profile_05 / total if total else 0.0,
            "strict_iou_hits_0_3": strict_03,
            "strict_iou_recall_0_3": strict_03 / total if total else 0.0,
            "strict_iou_hits_0_5": strict_05,
            "strict_iou_recall_0_5": strict_05 / total if total else 0.0,
            "macro_sample_profile_recall_0_3": mean(
                row["profile_recall_0_3"] for row in by_sample
            ) if by_sample else 0.0,
            "macro_sample_profile_recall_0_5": mean(
                row["profile_recall_0_5"] for row in by_sample
            ) if by_sample else 0.0,
            "mean_best_iou": mean(row["best_iou"] for row in rows) if rows else 0.0,
            "mean_pred_count": mean(row["n_pred"] for row in rows) if rows else 0.0,
            "large_box_only_rate": mean(float(row["large_box_only"]) for row in rows) if rows else 0.0,
            "model_failure_rate": mean(
                float(row["status"] in {"truncated", "parse_failed", "empty"}) for row in rows
            ) if rows else 0.0,
            "request_failure_rate": mean(
                float(row["status"] == "request_failed") for row in rows
            ) if rows else 0.0,
            "elapsed_seconds": sum(float(record.get("elapsed_seconds") or 0.0) for record in records),
            "token_usage": token_usage,
        },
        "go_no_go_thresholds": {
            "profile_recall_0_3_at_least": 0.6,
            "model_failure_rate_at_most": 0.2,
            "request_failure_rate_equals": 0.0,
            "large_box_only_rate_note": "must not be the dominant source of hits",
        },
        "by_sample": by_sample,
        "targets": rows,
    }


def font_for_visualization() -> Optional[ImageFont.FreeTypeFont]:
    for path in (Path("C:/Windows/Fonts/consola.ttf"), Path("C:/Windows/Fonts/arial.ttf")):
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), 16)
            except OSError:
                pass
    return None


def draw_record(image: Image.Image, record: Mapping[str, Any], output: Path) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = font_for_visualization()
    width, height = canvas.size

    def pixel_box(box: Sequence[float]) -> List[int]:
        return [
            int(box[0] * width / 1000),
            int(box[1] * height / 1000),
            int(box[2] * width / 1000),
            int(box[3] * height / 1000),
        ]

    draw.rectangle(pixel_box(record["gt_bbox"]), outline="#00ff55", width=4)
    for position, element in enumerate(record.get("elements", []), start=1):
        box = pixel_box(element["bbox"])
        draw.rectangle(box, outline="#ff3344", width=3)
        draw.text((box[0], max(0, box[1] - 18)), f"P{position}", fill="#ff3344", font=font)
    draw.text((5, 5), "GT=green  prediction=red", fill="#00ff55", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)


def annotation_requests(
    annotation_path: Path, selected_indices: Sequence[int]
) -> List[Dict[str, Any]]:
    annotations = load_annotations(annotation_path)
    requests: List[Dict[str, Any]] = []
    for index in selected_indices:
        sample = annotations.get(int(index))
        if sample is None:
            raise ValueError(f"annotation is missing selected index {index}")
        descriptions = [str(item["description"]).strip().casefold() for item in sample["elements"]]
        if len(descriptions) != len(set(descriptions)):
            raise ValueError(f"sample {index} has duplicate evidence descriptions; oracle target is ambiguous")
        for element in sample["elements"]:
            requests.append(
                {
                    "index": int(index),
                    "annotation_id": str(element["id"]),
                    "evidence_need": str(element["description"]),
                    "gt_bbox": [int(value) for value in element["bbox"].as_list()],
                }
            )
    return requests


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    value.add_argument("--indices", nargs="+", type=int)
    value.add_argument("--smoke", action="store_true", help="Use fixed indices 1 17 8 244 346.")
    value.add_argument("--tsv-path", type=Path, default=DEFAULT_TSV_PATH)
    value.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    value.add_argument("--annotation-path", type=Path, default=DEFAULT_ANNOTATION_PATH)
    value.add_argument("--output-dir", type=Path)
    value.add_argument("--api-base")
    value.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    value.add_argument("--max-tokens", type=int, default=12288)
    value.add_argument("--timeout", type=float, default=900)
    value.add_argument("--retries", type=int, default=3)
    value.add_argument("--retry-wait", type=float, default=5)
    value.add_argument("--nproc", type=int, default=1, help="Concurrent API requests; outputs remain per-target.")
    value.add_argument("--resume", action="store_true")
    value.add_argument("--evaluate-only", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.smoke and args.indices:
        raise ValueError("--smoke and --indices are mutually exclusive")
    selected_indices = list(DEFAULT_SMOKE_INDICES if args.smoke else (args.indices or []))
    if not selected_indices:
        raise ValueError("use --smoke or provide --indices")
    if len(selected_indices) != len(set(selected_indices)):
        raise ValueError("duplicate indices are not allowed")

    safe_model = "".join(char if char.isalnum() or char in "._-" else "_" for char in args.model_name)
    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / f"{safe_model}__{ORACLE_VERSION}__smoke5"
    )
    request_path = output_dir / "request_outputs.json"
    report_path = output_dir / "oracle_grounding_report.json"
    status_path = output_dir / "run_status.json"
    config_path = output_dir / "experiment_config.json"
    requests = annotation_requests(args.annotation_path, selected_indices)
    dataframe = pd.read_csv(args.tsv_path, sep="\t")
    selected = dataframe[dataframe["index"].astype(int).isin(selected_indices)].copy()
    rows = {int(row["index"]): row for _, row in selected.iterrows()}
    if set(rows) != set(selected_indices):
        raise ValueError(f"TSV indices differ from requested indices: found {sorted(rows)}")

    config = {
        "experiment_version": ORACLE_VERSION,
        "model_name": args.model_name,
        "selected_indices": selected_indices,
        "target_count": len(requests),
        "evidence_input": "question + GT annotation description; GT bbox is evaluation-only",
        "thinking_disabled": False,
        "temperature_explicitly_set": False,
        "guided_json": True,
        "max_tokens": args.max_tokens,
        "nproc": args.nproc,
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
        "system_prompt": ORACLE_SYSTEM_PROMPT,
        "user_prompt_template": ORACLE_USER_TEMPLATE,
        "guided_json_schema": ORACLE_JSON_SCHEMA,
        "created_at": utc_now(),
    }
    if args.dry_run:
        print(json.dumps({**config, "output_dir": str(output_dir.resolve())}, ensure_ascii=False, indent=2))
        return 0
    if args.evaluate_only:
        records = json.loads(request_path.read_text(encoding="utf-8"))
        report = make_report(records)
        save_json_atomic(report_path, report)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return 0
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory is not empty: {output_dir}; use --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and config_path.is_file():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        for key in (
            "experiment_version", "model_name", "selected_indices", "target_count",
            "max_tokens", "api_base", "annotation_sha256", "tsv_sha256", "input_image_sha256",
        ):
            if existing_config.get(key) != config.get(key):
                raise ValueError(f"cannot resume with changed {key}")
    else:
        save_json_atomic(config_path, config)

    existing = json.loads(request_path.read_text(encoding="utf-8")) if request_path.is_file() else []
    records = {
        (int(item["index"]), str(item["annotation_id"])): item for item in existing
    }
    if args.nproc < 1:
        raise ValueError("--nproc must be at least 1")
    client = OpenAI(
        api_key=args.api_key,
        base_url=config["api_base"],
        timeout=args.timeout,
        http_client=httpx.Client(timeout=args.timeout, trust_env=False),
    )
    image_cache: Dict[int, Tuple[Image.Image, str]] = {}
    total = len(requests)
    pending: List[Tuple[int, Dict[str, Any]]] = []
    for position, request in enumerate(requests, start=1):
        key = (request["index"], request["annotation_id"])
        prior = records.get(key)
        if prior and prior.get("status") != "request_failed":
            print(f"[{position}/{total}] q{request['index']} {request['annotation_id']} skip={prior['status']}")
            continue
        pending.append((position, request))

    image_cache: Dict[int, Tuple[Image.Image, str]] = {}
    for _, request in pending:
        index = request["index"]
        if index not in image_cache:
            image = load_image(rows[index], args.image_dir)
            data_uri, _ = encode_image(image)
            image_cache[index] = (image, data_uri)

    def execute_request(position: int, request: Dict[str, Any]) -> Tuple[int, Dict[str, Any], Optional[str]]:
        index = request["index"]
        _, data_uri = image_cache[index]
        question, question_source = load_question(rows[index])
        response, elapsed, error, attempts = call_model(
            client,
            model_name=args.model_name,
            messages=build_messages(question, request["evidence_need"], data_uri),
            max_tokens=args.max_tokens,
            extra_body=build_extra_body(),
            retries=args.retries,
            retry_wait=args.retry_wait,
        )
        if response is None:
            record = {
                **request,
                "question": question,
                "question_source": question_source,
                "status": "request_failed",
                "attempts": attempts,
                "error": error,
                "elements": [],
                "elapsed_seconds": elapsed,
                "token_usage": {"prompt": None, "completion": None, "total": None},
            }
            return position, record, None
        else:
            metadata = response_metadata(response, elapsed)
            raw_output = metadata.pop("raw_output")
            parsed = parse_oracle_response(raw_output)
            finish_reason = metadata.get("finish_reason")
            if finish_reason == "length":
                status, elements = "truncated", []
            elif parsed is None:
                status, elements = "parse_failed", []
            elif not parsed:
                status, elements = "empty", []
            else:
                status, elements = "valid", parsed
            record = {
                **request,
                "question": question,
                "question_source": question_source,
                "status": status,
                "attempts": attempts,
                "elements": elements,
                "diagnostic_parsed_elements": parsed if status == "truncated" else None,
                **metadata,
            }
            return position, record, raw_output

    def persist_result(position: int, record: Dict[str, Any], raw_output: Optional[str]) -> None:
        index = int(record["index"])
        annotation_id = str(record["annotation_id"])
        if raw_output is not None:
            raw_path = output_dir / "raw" / f"q{index:04d}" / f"{annotation_id}.txt"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(raw_output, encoding="utf-8")
            record["raw_output_path"] = str(raw_path.resolve())
        key = (index, annotation_id)
        records[key] = record
        ordered_records = [records[(item["index"], item["annotation_id"])] for item in requests if (item["index"], item["annotation_id"]) in records]
        save_json_atomic(request_path, ordered_records)
        save_json_atomic(report_path, make_report(ordered_records))
        save_json_atomic(
            status_path,
            {
                "experiment_version": ORACLE_VERSION,
                "model_name": args.model_name,
                "requested_targets": total,
                "completed_targets": len(ordered_records),
                "request_failed_targets": sum(item["status"] == "request_failed" for item in ordered_records),
                "complete": len(ordered_records) == total,
                "updated_at": utc_now(),
            },
        )
        draw_record(
            image_cache[index][0],
            record,
            output_dir / "visualizations" / f"q{index:04d}" / f"{annotation_id}.jpg",
        )
        print(
            f"[{position}/{total}] q{index} {annotation_id} "
            f"status={record['status']} boxes={len(record['elements'])} "
            f"elapsed={float(record.get('elapsed_seconds') or 0.0):.1f}s"
        )

    for position, request in pending:
        print(f"[{position}/{total}] submit q{request['index']} {request['annotation_id']} {request['evidence_need']}")
    if args.nproc == 1:
        for position, request in pending:
            persist_result(*execute_request(position, request))
    else:
        with ThreadPoolExecutor(max_workers=args.nproc) as executor:
            futures = {
                executor.submit(execute_request, position, request): (position, request)
                for position, request in pending
            }
            for future in as_completed(futures):
                position, request = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = (
                        position,
                        {
                            **request,
                            "question": load_question(rows[request["index"]])[0],
                            "question_source": load_question(rows[request["index"]])[1],
                            "status": "request_failed",
                            "attempts": 0,
                            "error": f"worker {type(exc).__name__}: {exc}",
                            "elements": [],
                            "elapsed_seconds": 0.0,
                            "token_usage": {"prompt": None, "completion": None, "total": None},
                        },
                        None,
                    )
                persist_result(*result)

    final_records = [records[(item["index"], item["annotation_id"])] for item in requests]
    final_report = make_report(final_records)
    save_json_atomic(request_path, final_records)
    save_json_atomic(report_path, final_report)
    save_json_atomic(
        status_path,
        {
            "experiment_version": ORACLE_VERSION,
            "model_name": args.model_name,
            "requested_targets": total,
            "completed_targets": len(final_records),
            "request_failed_targets": sum(item["status"] == "request_failed" for item in final_records),
            "complete": len(final_records) == total,
            "finished_at": utc_now(),
        },
    )
    print(json.dumps(final_report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

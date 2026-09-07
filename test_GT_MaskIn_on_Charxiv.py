"""Oracle evidence-condition experiments for 41 annotated CharXiv reasoning samples.

The transformation keeps the union of all annotated necessary-evidence boxes in the
original image coordinate system and replaces every other pixel with a constant
colour.  It intentionally does not crop, tile, draw borders, or expose annotation
descriptions to the answering model.

The same runner supports Full Image, GT Mask-In, GT Mask-Out, area-matched Random
Mask-In, and GT ROI multi-image inputs. It can prepare auditable inputs without an
API call, run an OpenAI-compatible vision model with resume support, export a
CharXiv-judge-compatible workbook, and optionally invoke the existing CharXiv judge.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image, ImageChops, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ANNOTATIONS = SCRIPT_DIR / "test_case/RIG_Charxiv/annotation_1.json"
DEFAULT_LMU_ROOT = Path(os.environ.get("LMU_DATA_ROOT", "C:/Users/19639/LMUData"))
DEFAULT_TSV_PATH = DEFAULT_LMU_ROOT / "CharXiv_reasoning_val.tsv"
DEFAULT_IMAGE_DIR = DEFAULT_LMU_ROOT / "images/CharXiv_reasoning_val"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "test_case/RIG_Charxiv/gt_mask_in"
DEFAULT_CONDITION_ROOT = SCRIPT_DIR / "test_case/RIG_Charxiv/evidence_conditions"

EXPERIMENT_VERSION = "gt-mask-in-v1"
CONDITION_EXPERIMENT_VERSION = "evidence-conditions-v1"
SUPPORTED_CONDITIONS = (
    "gt-mask-in",
    "full-image",
    "gt-mask-out",
    "random-mask-in",
    "gt-roi",
)
DEFAULT_MODEL_NAME = "Qwen3.5-27B"
DEFAULT_MAX_TOKENS = 12288
DEFAULT_MASK_COLOR = (127, 127, 127)
DEFAULT_RANDOM_SEED = 20260901
DEFAULT_RANDOM_CANDIDATES = 128
FREEZE_MARKER = "FROZEN.json"
EXCEL_CELL_LIMIT = 32767
NO_VALID_ANSWER_TRUNCATED = "[NO VALID ANSWER: generation reached the token limit]"
NO_VALID_ANSWER_EMPTY = "[NO VALID ANSWER: model returned no final answer]"
NO_VALID_ANSWER_FAILED = "[NO VALID ANSWER: inference request failed]"
NO_VALID_ANSWER_NOT_RUN = "[NO VALID ANSWER: sample was not run]"

MODEL_API_BASES = {
    "Qwen3.5-9B": os.environ.get("QWEN35_9B_API_BASE", "http://10.3.112.34:22630/v1"),
    "Qwen3.5-9B-API": os.environ.get("QWEN35_9B_API_BASE", "http://10.3.112.34:22630/v1"),
    "Qwen3.5-27B": os.environ.get("QWEN35_27B_API_BASE", "http://10.3.112.34:23290/v1"),
    "Qwen3.5-27B-API": os.environ.get("QWEN35_27B_API_BASE", "http://10.3.112.34:23290/v1"),
}

MODEL_SERVED_NAMES = {
    "Qwen3.5-9B-API": "Qwen3.5-9B",
    "Qwen3.5-27B-API": "Qwen3.5-27B",
}

MM_PROCESSOR_KWARGS = {
    "mm_processor_kwargs": {
        "images_kwargs": {
            "size": {
                "longest_edge": 8192 * 32 * 32,
                "shortest_edge": 4096 * 32 * 32,
            }
        }
    }
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def save_xlsx_atomic(path: Path, data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp.xlsx")
    data.to_excel(temporary, index=False)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not name:
        raise ValueError(f"cannot form a safe directory name from {value!r}")
    return name


def experiment_version(condition: str) -> str:
    if condition not in SUPPORTED_CONDITIONS:
        raise ValueError(f"unsupported condition: {condition!r}")
    return EXPERIMENT_VERSION if condition == "gt-mask-in" else CONDITION_EXPERIMENT_VERSION


def model_run_dir(
    model_name: str,
    output_dir: Optional[Path] = None,
    *,
    condition: str = "gt-mask-in",
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if condition == "gt-mask-in":
        # Preserve the historical path so existing results remain addressable and
        # can be frozen in place. New runs should use an explicit output directory.
        return DEFAULT_OUTPUT_ROOT / f"{safe_name(model_name)}__{EXPERIMENT_VERSION}"
    suffix = f"__seed-{random_seed}" if condition == "random-mask-in" else ""
    return (
        DEFAULT_CONDITION_ROOT
        / condition
        / f"{safe_name(model_name)}__{CONDITION_EXPERIMENT_VERSION}{suffix}"
    )


def freeze_run_directory(output_dir: Path, *, reason: str) -> Dict[str, Any]:
    """Write an integrity manifest that makes a completed run immutable to this runner."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {output_dir}")
    marker = output_dir / FREEZE_MARKER
    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        print(f"Already frozen: {output_dir}")
        return payload

    status_path = output_dir / "run_status.json"
    if not status_path.is_file():
        raise FileNotFoundError(
            f"refusing to freeze run without completion status: {status_path}"
        )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("complete") is not True:
        raise ValueError(
            f"refusing to freeze incomplete run: complete={status.get('complete')!r}"
        )
    if "protocol_complete" in status and status.get("protocol_complete") is not True:
        raise ValueError(
            "refusing to freeze run whose requested protocol is incomplete: "
            f"protocol_complete={status.get('protocol_complete')!r}"
        )
    requested = {int(index) for index in status.get("requested_indices", [])}
    completed_values = status.get("completed_indices", status.get("successful_indices", []))
    completed = {int(index) for index in completed_values}
    if not requested or completed != requested:
        missing = sorted(requested - completed)
        unexpected = sorted(completed - requested)
        raise ValueError(
            "refusing to freeze run with incomplete index coverage: "
            f"requested={len(requested)}, completed={len(completed)}, "
            f"missing={missing}, unexpected={unexpected}"
        )
    request_failures = int(status.get("request_failed_samples", status.get("failed_samples", 0)))
    failed_indices = status.get("request_failed_indices", status.get("failed_indices", []))
    if request_failures or failed_indices:
        raise ValueError(
            "refusing to freeze run with request failures: "
            f"count={request_failures}, indices={failed_indices}"
        )
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
        "file_count": len(files),
        "files": files,
    }
    save_json_atomic(marker, payload)
    print(f"Frozen {output_dir}: {len(files)} files recorded in {marker.name}")
    return payload


def assert_run_directory_writable(output_dir: Path, *, operation: str) -> None:
    marker = Path(output_dir) / FREEZE_MARKER
    if marker.exists():
        raise PermissionError(
            f"refusing to {operation}: run directory is frozen by {marker}; "
            "use a new --output-dir"
        )


def verify_frozen_directory(output_dir: Path) -> Dict[str, Any]:
    marker = Path(output_dir) / FREEZE_MARKER
    if not marker.is_file():
        raise FileNotFoundError(f"freeze marker not found: {marker}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    mismatches = []
    for item in payload.get("files", []):
        path = Path(output_dir) / item["path"]
        if not path.is_file():
            mismatches.append({"path": item["path"], "reason": "missing"})
        elif path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            mismatches.append({"path": item["path"], "reason": "content_changed"})
    result = {
        "valid": not mismatches,
        "checked_files": len(payload.get("files", [])),
        "mismatches": mismatches,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def images_equal(first: Image.Image, second: Image.Image) -> bool:
    return first.mode == second.mode and first.size == second.size and ImageChops.difference(
        first, second
    ).getbbox() is None


def validate_prepared_outputs(output_dir: Path) -> Dict[str, Any]:
    """Validate files, multimodal ordering, and pixel transforms for a prepared run."""
    output_dir = Path(output_dir)
    results_path = output_dir / "predictions.json"
    input_manifest_path = output_dir / "model_input_manifest.json"
    config_path = output_dir / "experiment_config.json"
    if not all(path.is_file() for path in (results_path, config_path)):
        raise FileNotFoundError("prepared run lacks predictions or experiment config")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    input_manifest = (
        json.loads(input_manifest_path.read_text(encoding="utf-8"))
        if input_manifest_path.is_file()
        else []
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_by_index = {int(item["index"]): item for item in input_manifest}
    condition = str(config["condition"])
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    samples = []

    if not input_manifest:
        warnings.append(
            {
                "check": "input_manifest",
                "detail": (
                    "historical run has no model_input_manifest.json; file and pixel "
                    "transforms are validated, but the serialized request ordering cannot "
                    "be independently audited"
                ),
            }
        )

    def record_error(index: int, check: str, detail: str) -> None:
        errors.append({"index": index, "check": check, "detail": detail})

    for result in results:
        index = int(result["index"])
        paths = [Path(path) for path in result.get("model_input_image_paths", [])]
        if not paths and condition == "gt-mask-in" and result.get("masked_image_path"):
            paths = [Path(result["masked_image_path"])]
        manifest = manifest_by_index.get(index)
        if input_manifest and manifest is None:
            record_error(index, "input_manifest", "entry missing")
            continue
        if "model_input_image_count" in result and len(paths) != int(
            result["model_input_image_count"]
        ):
            record_error(index, "image_count", "result count does not match path list")
        if manifest is not None:
            if len(paths) != int(manifest.get("model_input_image_count", -1)):
                record_error(index, "image_count", "manifest count does not match path list")
            expected_types = ["image_url"] * len(paths) + ["text"]
            actual_types = [item.get("type") for item in manifest.get("content", [])]
            if actual_types != expected_types:
                record_error(index, "message_order", f"expected {expected_types}, got {actual_types}")
        elif len(paths) != 1:
            record_error(index, "legacy_image_count", f"expected one GT Mask-In image, got {len(paths)}")
        for path in paths:
            if not path.is_file():
                record_error(index, "input_file", f"missing {path}")
                continue
            try:
                with Image.open(path) as prepared_image:
                    prepared_image.verify()
            except Exception as exc:
                record_error(index, "input_file", f"invalid image {path}: {exc}")

        source_path = Path(str(result.get("source_image_path", "")))
        if source_path.is_file() and paths:
            with Image.open(source_path) as source_handle:
                source = source_handle.convert("RGB")
            with Image.open(result["binary_mask_path"]) as mask_handle:
                gt_mask = mask_handle.convert("L")
            if condition == "gt-roi":
                expected_overview = apply_condition_mask(
                    source, gt_mask, condition="gt-mask-in", background=tuple(config["mask_color_rgb"])
                )
                with Image.open(paths[0]) as overview_handle:
                    if not images_equal(expected_overview, overview_handle.convert("RGB")):
                        record_error(index, "roi_overview_pixels", "overview differs from GT Mask-In")
                roi_paths = [Path(path) for path in result.get("roi_image_paths", [])]
                annotations = result.get("gt_annotations", [])
                if len(roi_paths) != len(annotations) or len(paths) != len(annotations) + 1:
                    record_error(index, "roi_count", "ROI count does not equal GT annotation count")
                for roi_path, annotation in zip(roi_paths, annotations):
                    expected_crop = source.crop(
                        normalized_box_to_pixels(
                            annotation["bbox"], source.size, config["bbox_padding_normalized"]
                        )
                    )
                    with Image.open(roi_path) as crop_handle:
                        if not images_equal(expected_crop, crop_handle.convert("RGB")):
                            record_error(index, "roi_pixels", f"crop differs from source: {roi_path}")
            else:
                if condition == "random-mask-in":
                    condition_mask_path = Path(result["condition_mask_path"])
                    with Image.open(condition_mask_path) as condition_mask_handle:
                        condition_mask = condition_mask_handle.convert("L")
                    random_stats = result.get("mask_statistics", {}).get("random_control", {})
                    if condition_mask.histogram()[255] != gt_mask.histogram()[255]:
                        record_error(index, "random_area", "random and GT masks have different areas")
                    if not random_stats.get("area_preserved_exactly"):
                        record_error(index, "random_area", "area preservation flag is false")
                else:
                    condition_mask = gt_mask
                expected_input = apply_condition_mask(
                    source,
                    condition_mask,
                    condition=condition,
                    background=tuple(config["mask_color_rgb"]),
                )
                with Image.open(paths[0]) as input_handle:
                    if not images_equal(expected_input, input_handle.convert("RGB")):
                        record_error(index, "condition_pixels", "prepared image differs from expected transform")

        samples.append(
            {
                "index": index,
                "model_input_image_count": len(paths),
                "gt_box_count": result.get("mask_statistics", {}).get("gt_box_count"),
                "gt_evidence_ratio": result.get("mask_statistics", {}).get("gt_evidence_ratio"),
                "input_retained_ratio": result.get("mask_statistics", {}).get("input_retained_ratio"),
                "random_gt_iou": result.get("mask_statistics", {})
                .get("random_control", {})
                .get("gt_random_iou"),
            }
        )

    report = {
        "valid": not errors,
        "condition": condition,
        "sample_count": len(results),
        "validated_at": utc_now(),
        "samples": samples,
        "errors": errors,
        "warnings": warnings,
        "request_manifest_audited": bool(input_manifest),
    }
    validation_path = output_dir / "preparation_validation_report.json"
    if (output_dir / FREEZE_MARKER).exists():
        print(
            "Frozen run validated read-only; preparation_validation_report.json "
            "was not modified."
        )
    else:
        save_json_atomic(validation_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_mask_color(value: str) -> Tuple[int, int, int]:
    try:
        values = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("mask color must be R,G,B") from exc
    if len(values) != 3 or any(channel < 0 or channel > 255 for channel in values):
        raise argparse.ArgumentTypeError("mask color must contain three integers in [0,255]")
    return values


def annotation_index(image_entry: Mapping[str, Any]) -> int:
    raw_path = str(image_entry.get("path", "")).replace("\\", "/")
    if not raw_path:
        raise ValueError("annotation image entry has no path")
    try:
        return int(Path(raw_path).stem)
    except ValueError as exc:
        raise ValueError(f"annotation image path does not end in an integer index: {raw_path}") from exc


def validate_box(box: Sequence[Any], *, label: str = "bbox") -> Tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError(f"{label} must contain four coordinates")
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains a non-numeric coordinate: {box!r}") from exc
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError(f"{label} contains a non-finite coordinate: {box!r}")
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        raise ValueError(f"{label} must satisfy 0 <= x1 < x2 <= 1000 and y likewise: {box!r}")
    return x1, y1, x2, y2


def load_annotations(path: Path) -> Dict[int, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    image_entries = payload.get("images")
    if not isinstance(image_entries, list):
        raise ValueError(f"annotation file must contain an images list: {path}")

    indexed: Dict[int, Dict[str, Any]] = {}
    for image_entry in image_entries:
        index = annotation_index(image_entry)
        if index in indexed:
            raise ValueError(f"duplicate annotated image index: {index}")
        raw_annotations = image_entry.get("annotations", [])
        if not isinstance(raw_annotations, list) or not raw_annotations:
            raise ValueError(f"annotated image {index} has no annotations")

        annotations = []
        for position, annotation in enumerate(raw_annotations):
            box = [annotation.get(key) for key in ("x1", "y1", "x2", "y2")]
            x1, y1, x2, y2 = validate_box(box, label=f"image {index} annotation {position}")
            annotations.append(
                {
                    "id": str(annotation.get("id", "")),
                    "description": str(annotation.get("description", "")),
                    "bbox": [x1, y1, x2, y2],
                }
            )
        indexed[index] = {
            "index": index,
            "source_path": str(image_entry.get("path", "")),
            "annotations": annotations,
        }
    return indexed


def normalized_box_to_pixels(
    box: Sequence[Any],
    image_size: Tuple[int, int],
    padding: float = 0,
) -> Tuple[int, int, int, int]:
    """Map [0,1000] coordinates to a half-open pixel rectangle."""
    x1, y1, x2, y2 = validate_box(box)
    if padding < 0:
        raise ValueError("bbox padding must be non-negative")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {image_size}")

    x1 = max(0.0, x1 - padding)
    y1 = max(0.0, y1 - padding)
    x2 = min(1000.0, x2 + padding)
    y2 = min(1000.0, y2 + padding)
    px1 = max(0, min(width, math.floor(x1 / 1000.0 * width)))
    py1 = max(0, min(height, math.floor(y1 / 1000.0 * height)))
    px2 = max(0, min(width, math.ceil(x2 / 1000.0 * width)))
    py2 = max(0, min(height, math.ceil(y2 / 1000.0 * height)))
    if px1 >= px2 or py1 >= py2:
        raise ValueError(f"box became empty after pixel conversion: {box!r}, size={image_size}")
    return px1, py1, px2, py2


def build_gt_union_mask(
    image: Image.Image,
    annotations: Sequence[Mapping[str, Any]],
    *,
    padding: float = 0,
    background: Tuple[int, int, int] = DEFAULT_MASK_COLOR,
) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
    """Return (masked RGB image, binary union mask, spatial statistics)."""
    source = image.convert("RGB")
    mask = Image.new("L", source.size, color=0)
    draw = ImageDraw.Draw(mask)
    pixel_boxes: List[Tuple[int, int, int, int]] = []
    summed_area = 0

    for annotation in annotations:
        pixel_box = normalized_box_to_pixels(annotation["bbox"], source.size, padding)
        pixel_boxes.append(pixel_box)
        x1, y1, x2, y2 = pixel_box
        summed_area += (x2 - x1) * (y2 - y1)
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)

    background_image = Image.new("RGB", source.size, color=background)
    masked = Image.composite(source, background_image, mask)
    total_pixels = source.width * source.height
    visible_pixels = mask.histogram()[255]
    overlap_pixels = max(0, summed_area - visible_pixels)
    statistics = {
        "image_width": source.width,
        "image_height": source.height,
        "gt_box_count": len(annotations),
        "pixel_boxes": [list(box) for box in pixel_boxes],
        "visible_pixels": visible_pixels,
        "visible_ratio": visible_pixels / total_pixels,
        "summed_box_pixels": summed_area,
        "summed_box_ratio": summed_area / total_pixels,
        "overlap_pixels": overlap_pixels,
        "overlap_ratio_of_box_area": overlap_pixels / summed_area if summed_area else 0.0,
    }
    return masked, mask, statistics


def apply_condition_mask(
    image: Image.Image,
    evidence_mask: Image.Image,
    *,
    condition: str,
    background: Tuple[int, int, int] = DEFAULT_MASK_COLOR,
) -> Image.Image:
    source = image.convert("RGB")
    background_image = Image.new("RGB", source.size, color=background)
    if condition in {"gt-mask-in", "random-mask-in"}:
        return Image.composite(source, background_image, evidence_mask)
    if condition == "gt-mask-out":
        return Image.composite(background_image, source, evidence_mask)
    if condition == "full-image":
        return source.copy()
    raise ValueError(f"condition does not use a single condition mask: {condition}")


def shifted_random_control_mask(
    evidence_mask: Image.Image,
    *,
    seed: int,
    candidates: int = DEFAULT_RANDOM_CANDIDATES,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Translate the GT union mask toroidally and select the least-overlapping sample.

    A cyclic translation preserves the exact number and arrangement of visible
    pixels, including overlaps between GT boxes. Candidate selection only uses the
    binary mask, never image content or the answer.
    """
    if candidates < 1:
        raise ValueError("random mask candidates must be at least 1")
    mask = evidence_mask.convert("L")
    width, height = mask.size
    rng = random.Random(seed)
    gt_pixels = mask.histogram()[255]
    if not gt_pixels:
        raise ValueError("cannot randomize an empty evidence mask")

    best_mask: Optional[Image.Image] = None
    best_shift: Optional[Tuple[int, int]] = None
    best_intersection: Optional[int] = None
    for _ in range(candidates):
        dx = rng.randrange(width) if width > 1 else 0
        dy = rng.randrange(height) if height > 1 else 0
        if dx == 0 and dy == 0 and (width > 1 or height > 1):
            dx = 1 if width > 1 else 0
        shifted = ImageChops.offset(mask, dx, dy)
        intersection = ImageChops.multiply(mask, shifted).histogram()[255]
        if best_intersection is None or intersection < best_intersection:
            best_mask = shifted.copy()
            best_shift = (dx, dy)
            best_intersection = intersection
            if intersection == 0:
                break

    assert best_mask is not None and best_shift is not None and best_intersection is not None
    random_pixels = best_mask.histogram()[255]
    if random_pixels != gt_pixels:
        raise AssertionError("cyclic mask translation did not preserve visible area")
    union_pixels = gt_pixels + random_pixels - best_intersection
    return best_mask, {
        "random_seed": seed,
        "candidate_count": candidates,
        "translation_pixels": list(best_shift),
        "visible_pixels": random_pixels,
        "gt_intersection_pixels": best_intersection,
        "gt_overlap_ratio": best_intersection / gt_pixels,
        "gt_random_iou": best_intersection / union_pixels if union_pixels else 0.0,
        "area_preserved_exactly": True,
        "selection_rule": "minimum_gt_overlap_over_random_cyclic_translations",
    }


def build_roi_inputs(
    image: Image.Image,
    annotations: Sequence[Mapping[str, Any]],
    *,
    padding: float,
    background: Tuple[int, int, int],
) -> Tuple[List[Image.Image], Image.Image, Dict[str, Any]]:
    """Return a masked overview followed by one crop per annotated GT box."""
    source = image.convert("RGB")
    overview, union_mask, statistics = build_gt_union_mask(
        source, annotations, padding=padding, background=background
    )
    crops = []
    crop_metadata = []
    for position, annotation in enumerate(annotations):
        pixel_box = normalized_box_to_pixels(annotation["bbox"], source.size, padding)
        crop = source.crop(pixel_box)
        crops.append(crop)
        crop_metadata.append(
            {
                "position": position,
                "annotation_id": str(annotation.get("id", "")),
                "pixel_box": list(pixel_box),
                "width": crop.width,
                "height": crop.height,
            }
        )
    statistics.update(
        {
            "roi_count": len(crops),
            "model_image_count": 1 + len(crops),
            "roi_order": "annotation_file_order",
            "overview_type": "gt-mask-in",
            "crop_metadata": crop_metadata,
        }
    )
    return [overview, *crops], union_mask, statistics


def load_source_image(
    row: pd.Series,
    annotation_entry: Mapping[str, Any],
    image_dir: Path,
) -> Tuple[Image.Image, str]:
    index = int(row["index"])
    candidates = [
        Path(str(annotation_entry.get("source_path", ""))),
        image_dir / f"{index}.jpg",
        image_dir / f"{index}.png",
    ]
    for candidate in candidates:
        if str(candidate) and candidate.is_file():
            with Image.open(candidate) as image:
                return image.convert("RGB"), str(candidate.resolve())

    encoded = row.get("image")
    if isinstance(encoded, str) and encoded.strip():
        try:
            image_bytes = base64.b64decode(encoded)
            with Image.open(io.BytesIO(image_bytes)) as image:
                return image.convert("RGB"), "tsv:image(base64)"
        except Exception as exc:
            raise FileNotFoundError(
                f"could not decode TSV image for index {index}"
            ) from exc
    raise FileNotFoundError(
        f"no image found for index {index}; tried annotation path and {image_dir}"
    )


def encode_png_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def get_question(row: pd.Series, question_field: str) -> str:
    value = row.get(question_field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"sample {row.get('index')} has no usable {question_field}")
    return value.strip()


def build_messages(question: str, image_data_uris: Sequence[str] | str) -> List[Dict[str, Any]]:
    """Match the CharXiv prompt while supporting one or several evidence images."""
    if isinstance(image_data_uris, str):
        image_data_uris = [image_data_uris]
    if not image_data_uris:
        raise ValueError("at least one model input image is required")
    content: List[Dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": uri, "detail": "high"}}
        for uri in image_data_uris
    ]
    content.append({"type": "text", "text": question})
    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def resolve_api_base(model_name: str, api_base: Optional[str]) -> str:
    resolved = api_base or MODEL_API_BASES.get(model_name)
    if not resolved:
        raise ValueError(
            f"no default API base for {model_name!r}; pass --api-base explicitly"
        )
    resolved = resolved.rstrip("/")
    suffix = "/chat/completions"
    if resolved.endswith(suffix):
        resolved = resolved[: -len(suffix)]
    return resolved


def resolve_served_model_name(model_name: str) -> str:
    return MODEL_SERVED_NAMES.get(model_name, model_name)


def call_model(
    *,
    model_name: str,
    api_base: str,
    api_key: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    timeout: float,
    retries: int,
    retry_wait: float,
) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout, max_retries=0)
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=MM_PROCESSOR_KWARGS,
            )
            elapsed = time.perf_counter() - started
            message = response.choices[0].message
            usage = getattr(response, "usage", None)
            return {
                "prediction": message.content or "",
                "reasoning_content": getattr(message, "reasoning_content", None),
                "finish_reason": getattr(response.choices[0], "finish_reason", None),
                "elapsed_seconds": elapsed,
                "attempts": attempt,
                "token_usage": {
                    "prompt": getattr(usage, "prompt_tokens", None),
                    "completion": getattr(usage, "completion_tokens", None),
                    "total": getattr(usage, "total_tokens", None),
                },
            }
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_wait * attempt)
    assert last_error is not None
    raise last_error


def extract_judge_prediction(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Derive an Excel-safe final response while retaining raw output in JSON.

    Older runs stored the full response in ``prediction``.  Newer entries also keep
    it under ``model_raw_output``.  A token-limited response is never treated as a
    valid answer, even if it happens to contain a closing thinking tag.
    """
    raw_output = str(result.get("model_raw_output", result.get("prediction", "")) or "")
    request_status = str(result.get("request_status", result.get("status", "not_run")))
    finish_reason = result.get("finish_reason")

    if request_status == "prepared":
        return {
            "answer_status": "prepared",
            "judge_prediction": NO_VALID_ANSWER_NOT_RUN,
            "model_raw_output": raw_output,
            "judge_prediction_trimmed": False,
        }
    if request_status == "failed":
        return {
            "answer_status": "request_failed",
            "judge_prediction": NO_VALID_ANSWER_FAILED,
            "model_raw_output": raw_output,
            "judge_prediction_trimmed": False,
        }
    if finish_reason == "length" or request_status == "truncated":
        return {
            "answer_status": "truncated",
            "judge_prediction": NO_VALID_ANSWER_TRUNCATED,
            "model_raw_output": raw_output,
            "judge_prediction_trimmed": False,
        }
    if not raw_output.strip():
        return {
            "answer_status": "empty",
            "judge_prediction": NO_VALID_ANSWER_EMPTY,
            "model_raw_output": raw_output,
            "judge_prediction_trimmed": False,
        }

    final_output = raw_output.rsplit("</think>", 1)[-1].strip()
    if not final_output:
        return {
            "answer_status": "empty",
            "judge_prediction": NO_VALID_ANSWER_EMPTY,
            "model_raw_output": raw_output,
            "judge_prediction_trimmed": False,
        }
    trimmed = len(final_output) > EXCEL_CELL_LIMIT
    if trimmed:
        # Keep the tail because the final answer is conventionally at the end.
        final_output = final_output[-EXCEL_CELL_LIMIT:]
    return {
        "answer_status": "valid",
        "judge_prediction": final_output,
        "model_raw_output": raw_output,
        "judge_prediction_trimmed": trimmed,
    }


def normalize_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(result)
    quality = extract_judge_prediction(result)
    normalized.update(quality)
    normalized["request_status"] = (
        "success" if quality["answer_status"] in {"valid", "truncated", "empty"}
        else quality["answer_status"]
    )
    if quality["answer_status"] == "valid":
        normalized["status"] = "success"
    elif quality["answer_status"] == "request_failed":
        normalized["status"] = "failed"
    else:
        normalized["status"] = quality["answer_status"]
    # ``prediction`` is the evaluator-facing field; raw content remains preserved.
    normalized["prediction"] = quality["judge_prediction"]
    return normalized


def summarize_answer_quality(
    results: Sequence[Mapping[str, Any]], requested_indices: Sequence[int]
) -> Dict[str, Any]:
    by_status: Dict[str, List[int]] = {
        "valid": [],
        "truncated": [],
        "empty": [],
        "request_failed": [],
        "prepared": [],
        "not_run": [],
    }
    result_map = {int(item["index"]): item for item in results}
    for index in requested_indices:
        result = result_map.get(int(index))
        if result is None:
            by_status["not_run"].append(int(index))
            continue
        status = extract_judge_prediction(result)["answer_status"]
        by_status.setdefault(status, []).append(int(index))

    completed = sorted(
        by_status["valid"] + by_status["truncated"] + by_status["empty"] + by_status["prepared"]
    )
    return {
        "requested_samples": len(requested_indices),
        "http_completed_samples": len(by_status["valid"] + by_status["truncated"] + by_status["empty"]),
        "valid_answer_samples": len(by_status["valid"]),
        "truncated_samples": len(by_status["truncated"]),
        "empty_answer_samples": len(by_status["empty"]),
        "request_failed_samples": len(by_status["request_failed"]),
        "prepared_samples": len(by_status["prepared"]),
        "not_run_samples": len(by_status["not_run"]),
        "valid_answer_rate": len(by_status["valid"]) / len(requested_indices) if requested_indices else 0.0,
        "completed_indices": completed,
        "valid_answer_indices": sorted(by_status["valid"]),
        "truncated_indices": sorted(by_status["truncated"]),
        "empty_answer_indices": sorted(by_status["empty"]),
        "request_failed_indices": sorted(by_status["request_failed"]),
        "prepared_indices": sorted(by_status["prepared"]),
        "not_run_indices": sorted(by_status["not_run"]),
        "protocol_complete": not by_status["request_failed"] and not by_status["not_run"],
    }


def build_prediction_dataframe(
    source_data: pd.DataFrame,
    requested_indices: Sequence[int],
    results: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Create a workbook accepted by ``CharXiv.evaluate``."""
    by_index = {int(result["index"]): normalize_result(result) for result in results}
    order = {index: position for position, index in enumerate(requested_indices)}
    selected = source_data[source_data["index"].astype(int).isin(requested_indices)].copy()
    selected["_order"] = selected["index"].astype(int).map(order)
    selected = selected.sort_values("_order").drop(columns=["_order"])
    if "image" in selected.columns:
        selected = selected.drop(columns=["image"])

    selected["prediction"] = [
        str(by_index.get(int(index), {}).get("prediction", "")) for index in selected["index"]
    ]
    selected["gt_mask_status"] = [
        str(by_index.get(int(index), {}).get("status", "not_run")) for index in selected["index"]
    ]
    selected["gt_mask_answer_status"] = [
        str(by_index.get(int(index), {}).get("answer_status", "not_run"))
        for index in selected["index"]
    ]
    selected["gt_mask_finish_reason"] = [
        str(by_index.get(int(index), {}).get("finish_reason", "")) for index in selected["index"]
    ]
    selected["gt_mask_visible_ratio"] = [
        by_index.get(int(index), {}).get("mask_statistics", {}).get("visible_ratio")
        for index in selected["index"]
    ]
    selected["gt_mask_image_path"] = [
        str(by_index.get(int(index), {}).get("masked_image_path", "")) for index in selected["index"]
    ]
    selected["evidence_condition"] = [
        str(by_index.get(int(index), {}).get("condition", "")) for index in selected["index"]
    ]
    selected["condition_image_path"] = [
        str(by_index.get(int(index), {}).get("condition_image_path", ""))
        for index in selected["index"]
    ]
    selected["model_input_image_count"] = [
        by_index.get(int(index), {}).get("model_input_image_count") for index in selected["index"]
    ]
    selected["model_input_image_paths_json"] = [
        json.dumps(
            by_index.get(int(index), {}).get("model_input_image_paths", []),
            ensure_ascii=False,
        )
        for index in selected["index"]
    ]
    selected["input_retained_ratio"] = [
        by_index.get(int(index), {})
        .get("mask_statistics", {})
        .get("input_retained_ratio")
        for index in selected["index"]
    ]
    return selected


def build_input_manifest(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Create a base64-free audit view of the exact multimodal input order."""
    manifest = []
    for result in results:
        paths = [str(path) for path in result.get("model_input_image_paths", [])]
        content = [
            {
                "position": position,
                "type": "image_url",
                "detail": "high",
                "local_path": path,
            }
            for position, path in enumerate(paths, start=1)
        ]
        question = str(result.get("question", ""))
        if question:
            content.append({"position": len(content) + 1, "type": "text", "text": question})
        manifest.append(
            {
                "index": int(result["index"]),
                "condition": str(result.get("condition", "")),
                "model_input_image_count": len(paths),
                "message_role": "user",
                "content": content,
            }
        )
    return manifest


def rebuild_existing_outputs(output_dir: Path, tsv_path: Optional[Path] = None) -> Dict[str, Any]:
    """Rebuild evaluator inputs from a completed run without modifying raw files."""
    assert_run_directory_writable(output_dir, operation="rebuild evaluator outputs")
    results_path = output_dir / "predictions.json"
    config_path = output_dir / "experiment_config.json"
    if not results_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"missing predictions.json or experiment_config.json in {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    resolved_tsv = Path(tsv_path or config["tsv_path"])
    source_data = pd.read_csv(resolved_tsv, sep="\t")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    requested_indices = [int(value) for value in config.get("requested_indices", [])]
    if not requested_indices:
        requested_indices = [int(item["index"]) for item in results]

    normalized_results = [normalize_result(item) for item in results]
    quality_report = summarize_answer_quality(results, requested_indices)
    quality_report.update(
        {
            "experiment_version": config.get("experiment_version", EXPERIMENT_VERSION),
            "model_name": config.get("model_name"),
            "condition": config.get("condition", "gt-mask-in"),
            "source_predictions_path": str(results_path.resolve()),
            "source_predictions_sha256": sha256_file(results_path),
            "rebuilt_at": utc_now(),
        }
    )
    normalized_path = output_dir / "predictions_normalized.json"
    judge_path = output_dir / "predictions_for_judge.xlsx"
    report_path = output_dir / "answer_quality_report.json"
    save_json_atomic(normalized_path, normalized_results)
    save_xlsx_atomic(
        judge_path,
        build_prediction_dataframe(source_data, requested_indices, normalized_results),
    )
    save_json_atomic(report_path, quality_report)
    print(
        f"Rebuilt {config.get('model_name')}: valid={quality_report['valid_answer_samples']}, "
        f"truncated={quality_report['truncated_samples']}, "
        f"empty={quality_report['empty_answer_samples']}, "
        f"request_failed={quality_report['request_failed_samples']}"
    )
    print(f"Judge workbook: {judge_path}")
    return quality_report


def prepare_output_directory(
    output_dir: Path,
    *,
    resume: bool,
    overwrite_existing: bool,
) -> None:
    assert_run_directory_writable(output_dir, operation="write experiment outputs")
    if resume and overwrite_existing:
        raise ValueError("--resume and --overwrite-existing are mutually exclusive")
    nonempty = output_dir.exists() and any(output_dir.iterdir())
    if nonempty and not resume and not overwrite_existing:
        raise FileExistsError(
            f"run directory is not empty: {output_dir}; use --resume, "
            "--overwrite-existing, or a different --output-dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def prepare_condition_sample(
    *,
    row: pd.Series,
    annotation_entry: Mapping[str, Any],
    image_dir: Path,
    condition: str,
    images_dir: Path,
    masks_dir: Path,
    roi_dir: Path,
    padding: float,
    mask_color: Tuple[int, int, int],
    random_seed: int,
    random_candidates: int,
) -> Dict[str, Any]:
    index = int(row["index"])
    image, source_path = load_source_image(row, annotation_entry, image_dir)
    gt_mask_in, gt_mask, statistics = build_gt_union_mask(
        image,
        annotation_entry["annotations"],
        padding=padding,
        background=mask_color,
    )
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    mask_path = masks_dir / f"q{index:04d}_union_mask.png"
    gt_mask.save(mask_path, format="PNG")

    condition_mask_path = mask_path
    roi_paths: List[str] = []
    if condition == "gt-roi":
        model_images, _, statistics = build_roi_inputs(
            image,
            annotation_entry["annotations"],
            padding=padding,
            background=mask_color,
        )
        overview_path = images_dir / f"q{index:04d}_gt_roi_overview.png"
        model_images[0].save(overview_path, format="PNG")
        sample_roi_dir = roi_dir / f"q{index:04d}"
        sample_roi_dir.mkdir(parents=True, exist_ok=True)
        for position, crop in enumerate(model_images[1:], start=1):
            crop_path = sample_roi_dir / f"roi_{position:03d}.png"
            crop.save(crop_path, format="PNG")
            roi_paths.append(str(crop_path.resolve()))
        primary_input_path = overview_path
        input_paths = [str(overview_path.resolve()), *roi_paths]
        retained_ratio = statistics["visible_ratio"]
    else:
        if condition == "full-image":
            condition_mask = Image.new("L", image.size, color=255)
        elif condition == "gt-mask-out":
            condition_mask = gt_mask
        elif condition == "gt-mask-in":
            condition_mask = gt_mask
        elif condition == "random-mask-in":
            condition_mask, random_statistics = shifted_random_control_mask(
                gt_mask,
                seed=random_seed + index,
                candidates=random_candidates,
            )
            statistics["random_control"] = random_statistics
            condition_mask_path = masks_dir / f"q{index:04d}_random_mask.png"
            condition_mask.save(condition_mask_path, format="PNG")
        else:
            raise ValueError(f"unsupported condition: {condition}")
        condition_image = apply_condition_mask(
            image,
            condition_mask,
            condition=condition,
            background=mask_color,
        )
        primary_input_path = images_dir / f"q{index:04d}_{condition.replace('-', '_')}.png"
        condition_image.save(primary_input_path, format="PNG")
        model_images = [condition_image]
        input_paths = [str(primary_input_path.resolve())]
        if condition == "full-image":
            retained_ratio = 1.0
        elif condition == "gt-mask-out":
            retained_ratio = 1.0 - statistics["visible_ratio"]
        else:
            retained_ratio = statistics["visible_ratio"]

    statistics.update(
        {
            "gt_evidence_pixels": statistics["visible_pixels"],
            "gt_evidence_ratio": statistics["visible_ratio"],
            "input_retained_ratio": retained_ratio,
            "condition": condition,
        }
    )
    return {
        "index": index,
        "source_image_path": source_path,
        "condition_image_path": str(primary_input_path.resolve()),
        "model_input_image_paths": input_paths,
        "model_input_image_count": len(input_paths),
        "roi_image_paths": roi_paths,
        # Backward-compatible field used by the existing GT Mask-In workbook.
        "masked_image_path": str(primary_input_path.resolve()),
        "binary_mask_path": str(mask_path.resolve()),
        "condition_mask_path": str(condition_mask_path.resolve()),
        "mask_statistics": statistics,
        "model_images": model_images,
    }


def run_evidence_condition(
    *,
    annotations_path: Path = DEFAULT_ANNOTATIONS,
    tsv_path: Path = DEFAULT_TSV_PATH,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    indices: Optional[Sequence[int]] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    api_base: Optional[str] = None,
    api_key: str = "EMPTY",
    output_dir: Optional[Path] = None,
    condition: str = "gt-mask-in",
    question_field: str = "question",
    mask_color: Tuple[int, int, int] = DEFAULT_MASK_COLOR,
    bbox_padding: float = 0,
    random_seed: int = DEFAULT_RANDOM_SEED,
    random_candidates: int = DEFAULT_RANDOM_CANDIDATES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    timeout: float = 900,
    retries: int = 5,
    retry_wait: float = 5,
    prepare_only: bool = False,
    resume: bool = False,
    overwrite_existing: bool = False,
) -> List[Dict[str, Any]]:
    if condition not in SUPPORTED_CONDITIONS:
        raise ValueError(f"unsupported condition: {condition!r}")
    if bbox_padding < 0 or bbox_padding > 1000:
        raise ValueError("bbox padding must be in [0,1000]")
    if random_candidates < 1:
        raise ValueError("random mask candidates must be at least 1")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if retries < 1:
        raise ValueError("retries must be at least 1")
    if timeout <= 0 or retry_wait < 0:
        raise ValueError("timeout must be positive and retry_wait non-negative")
    run_version = experiment_version(condition)
    output_dir = model_run_dir(
        model_name,
        output_dir,
        condition=condition,
        random_seed=random_seed,
    )
    prepare_output_directory(
        output_dir, resume=resume, overwrite_existing=overwrite_existing
    )
    annotations = load_annotations(annotations_path)
    data = pd.read_csv(tsv_path, sep="\t")
    data_indices = set(data["index"].astype(int))

    requested_indices = list(indices) if indices is not None else sorted(annotations)
    if len(requested_indices) != len(set(requested_indices)):
        raise ValueError("requested indices contain duplicates")
    missing_annotations = sorted(set(requested_indices) - set(annotations))
    missing_rows = sorted(set(requested_indices) - data_indices)
    if missing_annotations:
        raise ValueError(f"indices absent from annotations: {missing_annotations}")
    if missing_rows:
        raise ValueError(f"indices absent from CharXiv TSV: {missing_rows}")

    results_path = output_dir / "predictions.json"
    status_path = output_dir / "run_status.json"
    rebuilt_workbook = output_dir / "predictions_for_judge.xlsx"
    workbook_path = rebuilt_workbook if rebuilt_workbook.exists() else output_dir / "predictions.xlsx"
    manifest_path = output_dir / "experiment_config.json"
    input_manifest_path = output_dir / "model_input_manifest.json"
    images_dir = output_dir / ("masked_images" if condition == "gt-mask-in" else "input_images")
    masks_dir = output_dir / "binary_masks"
    roi_dir = output_dir / "roi_images"

    results: List[Dict[str, Any]] = []
    if resume and results_path.exists():
        loaded = json.loads(results_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"resume file must contain a list: {results_path}")
        results = loaded
        versions = {str(item.get("experiment_version")) for item in results}
        models = {str(item.get("model_name")) for item in results}
        if versions and versions != {run_version}:
            raise ValueError(f"cannot resume across experiment versions: {versions}")
        if models and models != {model_name}:
            raise ValueError(f"cannot resume across models: {models}")

    resolved_api_base = None if prepare_only else resolve_api_base(model_name, api_base)
    served_model_name = resolve_served_model_name(model_name)
    config = {
        "experiment_version": run_version,
        "condition": condition,
        "model_name": model_name,
        "served_model_name": served_model_name,
        "api_base": resolved_api_base,
        "annotations_path": str(annotations_path.resolve()),
        "annotations_sha256": sha256_file(annotations_path),
        "tsv_path": str(tsv_path.resolve()),
        "image_dir": str(image_dir.resolve()),
        "question_field": question_field,
        "mask_color_rgb": list(mask_color),
        "bbox_coordinate_space": "normalized_[0,1000]",
        "bbox_padding_normalized": bbox_padding,
        "random_seed": random_seed if condition == "random-mask-in" else None,
        "random_mask_candidates": random_candidates if condition == "random-mask-in" else None,
        "random_mask_method": (
            "minimum-overlap cyclic translation of the GT union mask"
            if condition == "random-mask-in"
            else None
        ),
        "mask_composition": "pixel_union",
        "preserve_canvas_size": True,
        "draw_annotations_on_model_input": False,
        "model_input_layout": (
            "gt-mask-in overview followed by one crop per GT annotation"
            if condition == "gt-roi"
            else "single image"
        ),
        "image_transport_format": "PNG",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout_seconds": timeout,
        "api_retries": retries,
        "retry_wait_seconds": retry_wait,
        "thinking_disabled": False,
        "requested_indices": requested_indices,
        "created_at": utc_now(),
    }
    if resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        immutable_keys = (
            "experiment_version",
            "condition",
            "model_name",
            "annotations_sha256",
            "question_field",
            "mask_color_rgb",
            "bbox_padding_normalized",
            "random_seed",
            "random_mask_candidates",
            "temperature",
            "max_tokens",
        )
        changed = [key for key in immutable_keys if previous.get(key) != config.get(key)]
        if changed:
            raise ValueError(f"cannot resume with changed experiment settings: {changed}")
        config["created_at"] = previous.get("created_at", config["created_at"])
        config["resumed_at"] = utc_now()
    save_json_atomic(manifest_path, config)

    initial_quality = summarize_answer_quality(results, requested_indices)
    completed = set(initial_quality["completed_indices"])
    status: Dict[str, Any] = {
        "experiment_version": run_version,
        "model_name": model_name,
        "condition": condition,
        "requested_indices": requested_indices,
        "completed_indices": sorted(completed),
        "valid_answer_indices": initial_quality["valid_answer_indices"],
        "truncated_indices": initial_quality["truncated_indices"],
        "empty_answer_indices": initial_quality["empty_answer_indices"],
        "failed_indices": initial_quality["request_failed_indices"],
        "prepare_only": prepare_only,
        "updated_at": utc_now(),
    }
    save_json_atomic(results_path, results)
    save_json_atomic(status_path, status)

    row_map = {int(row["index"]): row for _, row in data.iterrows()}
    for position, index in enumerate(requested_indices, start=1):
        if index in completed:
            print(f"[{position}/{len(requested_indices)}] #{index}: already complete; skipping")
            continue
        print(f"[{position}/{len(requested_indices)}] #{index}: preparing {condition}")
        row = row_map[index]
        try:
            prepared = prepare_condition_sample(
                row=row,
                annotation_entry=annotations[index],
                image_dir=image_dir,
                condition=condition,
                images_dir=images_dir,
                masks_dir=masks_dir,
                roi_dir=roi_dir,
                padding=bbox_padding,
                mask_color=mask_color,
                random_seed=random_seed,
                random_candidates=random_candidates,
            )
            model_images = prepared.pop("model_images")
            entry: Dict[str, Any] = {
                "index": index,
                "experiment_version": run_version,
                "condition": condition,
                "model_name": model_name,
                "served_model_name": served_model_name,
                "inst_category": int(row["inst_category"]),
                "question": get_question(row, question_field),
                "question_source_field": question_field,
                "dataset_question": str(row.get("question", "")),
                "gt_answer": str(row.get("answer", "")),
                "gt_annotations": annotations[index]["annotations"],
                **prepared,
            }
            if prepare_only:
                entry.update({"status": "prepared", "prediction": ""})
            else:
                print(
                    f"  retained={entry['mask_statistics']['input_retained_ratio']:.2%}, "
                    f"boxes={entry['mask_statistics']['gt_box_count']}, "
                    f"images={entry['model_input_image_count']}; "
                    f"calling {served_model_name}"
                )
                response_data = call_model(
                    model_name=served_model_name,
                    api_base=str(resolved_api_base),
                    api_key=api_key,
                    messages=build_messages(
                        entry["question"],
                        [encode_png_data_uri(image) for image in model_images],
                    ),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    retries=retries,
                    retry_wait=retry_wait,
                )
                raw_output = response_data.pop("prediction")
                entry.update(
                    {
                        "request_status": "success",
                        "model_raw_output": raw_output,
                        **response_data,
                    }
                )
                entry = normalize_result(entry)
                print(
                    f"  completed in {response_data['elapsed_seconds']:.1f}s; "
                    f"finish_reason={response_data['finish_reason']}"
                )
        except Exception as exc:
            entry = {
                "index": index,
                "experiment_version": run_version,
                "condition": condition,
                "model_name": model_name,
                "served_model_name": served_model_name,
                "status": "failed",
                "request_status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(f"  FAILED: {type(exc).__name__}: {exc}")

        results = [item for item in results if int(item["index"]) != index]
        results.append(entry)
        results.sort(key=lambda item: requested_indices.index(int(item["index"])))
        quality = summarize_answer_quality(results, requested_indices)
        completed = set(quality["completed_indices"])
        status.update(
            {
                "completed_indices": sorted(completed),
                "valid_answer_indices": quality["valid_answer_indices"],
                "truncated_indices": quality["truncated_indices"],
                "empty_answer_indices": quality["empty_answer_indices"],
                "failed_indices": quality["request_failed_indices"],
                "updated_at": utc_now(),
            }
        )
        save_json_atomic(results_path, results)
        save_json_atomic(status_path, status)
        save_json_atomic(input_manifest_path, build_input_manifest(results))
        prediction_data = build_prediction_dataframe(data, requested_indices, results)
        save_xlsx_atomic(workbook_path, prediction_data)

    status["finished_at"] = utc_now()
    quality = summarize_answer_quality(results, requested_indices)
    status.update(quality)
    status["successful_samples"] = (
        quality["prepared_samples"] if prepare_only else quality["valid_answer_samples"]
    )
    status["failed_samples"] = quality["request_failed_samples"]
    status["complete"] = quality["protocol_complete"]
    save_json_atomic(status_path, status)
    save_json_atomic(results_path, results)
    save_json_atomic(input_manifest_path, build_input_manifest(results))
    save_xlsx_atomic(
        workbook_path, build_prediction_dataframe(data, requested_indices, results)
    )
    print(
        f"Finished: valid={quality['valid_answer_samples']}, "
        f"truncated={quality['truncated_samples']}, "
        f"empty={quality['empty_answer_samples']}, "
        f"prepared={quality['prepared_samples']}; outputs: {output_dir}"
    )
    return results


def run_gt_mask_in(**kwargs: Any) -> List[Dict[str, Any]]:
    """Backward-compatible entry point for the original GT Mask-In experiment."""
    supplied_condition = kwargs.pop("condition", "gt-mask-in")
    if supplied_condition != "gt-mask-in":
        raise ValueError("run_gt_mask_in only supports condition='gt-mask-in'")
    return run_evidence_condition(condition="gt-mask-in", **kwargs)


def evaluate_predictions(
    output_dir: Path,
    *,
    judge_model: str,
    judge_nproc: int,
    judge_timeout: float = 120,
    judge_retry: int = 3,
    judge_max_tokens: int = 256,
    allow_incomplete: bool = False,
) -> pd.DataFrame:
    assert_run_directory_writable(output_dir, operation="evaluate predictions")
    status_path = output_dir / "run_status.json"
    rebuilt_workbook = output_dir / "predictions_for_judge.xlsx"
    workbook_path = rebuilt_workbook if rebuilt_workbook.exists() else output_dir / "predictions.xlsx"
    if not status_path.exists() or not workbook_path.exists():
        raise FileNotFoundError(
            "run_status.json and predictions_for_judge.xlsx (or predictions.xlsx) "
            "are required for evaluation"
        )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("prepare_only"):
        raise ValueError("prepare-only outputs contain no model predictions")
    if not status.get("complete") and not allow_incomplete:
        raise ValueError(
            "refusing to evaluate an incomplete run; resume it or pass "
            "--allow-incomplete-evaluation explicitly"
        )
    from vlmeval.dataset.charxiv import CharXiv

    print(f"Evaluation workbook: {workbook_path}")
    dataset = CharXiv(dataset="CharXiv_reasoning_val")
    return dataset.evaluate(
        str(workbook_path),
        model=judge_model,
        nproc=judge_nproc,
        timeout=judge_timeout,
        retry=judge_retry,
        max_tokens=judge_max_tokens,
    )


def validate_dry_run(
    *,
    annotations_path: Path,
    tsv_path: Path,
    image_dir: Path,
    indices: Optional[Sequence[int]],
    model_name: str,
    condition: str,
    random_seed: int,
    api_base: Optional[str],
    output_dir: Optional[Path],
) -> None:
    annotations = load_annotations(annotations_path)
    data = pd.read_csv(tsv_path, sep="\t")
    selected = list(indices) if indices is not None else sorted(annotations)
    missing_annotations = sorted(set(selected) - set(annotations))
    missing_rows = sorted(set(selected) - set(data["index"].astype(int)))
    local_sources = sum(
        1
        for index in selected
        if Path(annotations[index]["source_path"]).is_file()
        or (image_dir / f"{index}.jpg").is_file()
        or (image_dir / f"{index}.png").is_file()
    ) if not missing_annotations else 0
    print(f"Experiment: {experiment_version(condition)}")
    print(f"Condition: {condition}")
    print(f"Model: {model_name}")
    print(f"API base: {resolve_api_base(model_name, api_base)}")
    print(f"Annotations: {annotations_path} ({len(annotations)} images)")
    print(f"Selected indices: {len(selected)}")
    print(f"GT boxes: {sum(len(annotations[i]['annotations']) for i in selected if i in annotations)}")
    print(f"Local image sources resolved: {local_sources}/{len(selected)}")
    print(f"Missing annotations: {missing_annotations}")
    print(f"Missing TSV rows: {missing_rows}")
    print(
        "Output directory: "
        f"{model_run_dir(model_name, output_dir, condition=condition, random_seed=random_seed)}"
    )
    if missing_annotations or missing_rows:
        raise ValueError("dry-run validation failed")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run controlled evidence-condition experiments on annotated CharXiv samples."
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--tsv-path", type=Path, default=DEFAULT_TSV_PATH)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--condition",
        choices=SUPPORTED_CONDITIONS,
        default="gt-mask-in",
        help="Visual evidence condition. Existing GT Mask-In result directories are frozen in place.",
    )
    parser.add_argument("--api-base")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--question-field",
        choices=("question", "raw_question"),
        default="question",
        help="Use 'question' for comparability with standard CharXiv full-image runs.",
    )
    parser.add_argument("--mask-color", type=parse_mask_color, default=DEFAULT_MASK_COLOR)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--random-candidates",
        type=int,
        default=DEFAULT_RANDOM_CANDIDATES,
        help="Number of cyclic translations searched for the least GT-overlapping random mask.",
    )
    parser.add_argument(
        "--bbox-padding",
        type=float,
        default=0,
        help="Symmetric padding in the normalized [0,1000] coordinate space.",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-wait", type=float, default=5)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate masks and manifests without calling the model API.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rebuild-existing",
        action="store_true",
        help="Rebuild normalized results and judge workbook without model API calls.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--judge-model", default="qwen-plus")
    parser.add_argument("--judge-nproc", type=int, default=4)
    parser.add_argument("--judge-timeout", type=float, default=120)
    parser.add_argument("--judge-retry", type=int, default=3)
    parser.add_argument("--judge-max-tokens", type=int, default=256)
    parser.add_argument("--allow-incomplete-evaluation", action="store_true")
    parser.add_argument(
        "--freeze-existing",
        action="store_true",
        help="Freeze an existing run by recording hashes and rejecting future runner writes.",
    )
    parser.add_argument(
        "--verify-frozen",
        action="store_true",
        help="Verify every file recorded in an existing FROZEN.json manifest.",
    )
    parser.add_argument(
        "--validate-prepared",
        action="store_true",
        help="Validate prepared images, ROI ordering, masks, and multimodal input manifests.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    output_dir = model_run_dir(
        args.model_name,
        args.output_dir,
        condition=args.condition,
        random_seed=args.random_seed,
    )
    management_modes = sum(
        bool(value) for value in (args.freeze_existing, args.verify_frozen, args.validate_prepared)
    )
    if management_modes > 1:
        raise ValueError(
            "--freeze-existing, --verify-frozen, and --validate-prepared are mutually exclusive"
        )
    if args.freeze_existing:
        freeze_run_directory(
            output_dir,
            reason=(
                f"Completed {args.model_name} {args.condition} result frozen for "
                "immutable condition comparison."
            ),
        )
        return
    if args.verify_frozen:
        result = verify_frozen_directory(output_dir)
        if not result["valid"]:
            raise RuntimeError("frozen run integrity verification failed")
        return
    if args.validate_prepared:
        result = validate_prepared_outputs(output_dir)
        if not result["valid"]:
            raise RuntimeError("prepared input validation failed")
        return
    if args.dry_run:
        validate_dry_run(
            annotations_path=args.annotations,
            tsv_path=args.tsv_path,
            image_dir=args.image_dir,
            indices=args.indices,
            model_name=args.model_name,
            condition=args.condition,
            random_seed=args.random_seed,
            api_base=args.api_base,
            output_dir=args.output_dir,
        )
        return

    if args.rebuild_existing:
        rebuild_existing_outputs(output_dir, args.tsv_path)
        if args.evaluate:
            scores = evaluate_predictions(
                output_dir,
                judge_model=args.judge_model,
                judge_nproc=args.judge_nproc,
                judge_timeout=args.judge_timeout,
                judge_retry=args.judge_retry,
                judge_max_tokens=args.judge_max_tokens,
                allow_incomplete=args.allow_incomplete_evaluation,
            )
            print(scores.to_string(index=False))
        return

    run_evidence_condition(
        annotations_path=args.annotations,
        tsv_path=args.tsv_path,
        image_dir=args.image_dir,
        indices=args.indices,
        model_name=args.model_name,
        condition=args.condition,
        api_base=args.api_base,
        api_key=args.api_key,
        output_dir=output_dir,
        question_field=args.question_field,
        mask_color=args.mask_color,
        bbox_padding=args.bbox_padding,
        random_seed=args.random_seed,
        random_candidates=args.random_candidates,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        retries=args.retries,
        retry_wait=args.retry_wait,
        prepare_only=args.prepare_only,
        resume=args.resume,
        overwrite_existing=args.overwrite_existing,
    )
    if args.evaluate:
        scores = evaluate_predictions(
            output_dir,
            judge_model=args.judge_model,
            judge_nproc=args.judge_nproc,
            judge_timeout=args.judge_timeout,
            judge_retry=args.judge_retry,
            judge_max_tokens=args.judge_max_tokens,
            allow_incomplete=args.allow_incomplete_evaluation,
        )
        print(scores.to_string(index=False))


if __name__ == "__main__":
    main()

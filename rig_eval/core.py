"""Core metrics for the current (v1) RIG annotation format.

The annotation object's ``id`` is treated as the identity of one necessary
semantic clue.  Bounding boxes only generate spatial candidates; they do not
merge annotation objects, even when boxes overlap or one contains another.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.x1, self.y1, self.x2, self.y2)):
            raise ValueError("bbox coordinates must be finite")
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError(f"invalid bbox: {self.as_list()}")

    @classmethod
    def from_value(cls, value: Sequence[Any]) -> "BBox":
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError(f"bbox must contain four coordinates: {value!r}")
        return cls(*(float(v) for v in value))

    def as_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def intersection_area(self, other: "BBox") -> float:
        width = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        height = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        return width * height

    def iou(self, other: "BBox") -> float:
        intersection = self.intersection_area(other)
        union = self.area + other.area - intersection
        return intersection / union if union else 0.0

    def coverage_by(self, other: "BBox") -> float:
        """Fraction of this box covered by ``other``."""
        return self.intersection_area(other) / self.area


@dataclass(frozen=True)
class EvaluationConfig:
    iou_thresholds: Tuple[float, ...] = (0.3, 0.5)
    containment_threshold: float = 0.7
    max_pred_to_gt_area_ratio: float = 4.0
    gt_overlap_iou_threshold: float = 0.0
    semantic_policy: str = "overlap_review"

    def __post_init__(self) -> None:
        valid = {"spatial", "overlap_review", "review_all"}
        if self.semantic_policy not in valid:
            raise ValueError(f"semantic_policy must be one of {sorted(valid)}")
        if not self.iou_thresholds:
            raise ValueError("at least one IoU threshold is required")


def _sample_index_from_path(path: str) -> int:
    stem = Path(path.replace("\\", "/")).stem
    try:
        return int(stem)
    except ValueError as exc:
        raise ValueError(f"annotation image filename must be an integer: {path}") from exc


def load_annotations(path: Path) -> Dict[int, Dict[str, Any]]:
    """Load annotation schema v1 without requiring a schema upgrade."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if str(data.get("version")) != "1.0":
        raise ValueError(f"expected annotation version 1.0, got {data.get('version')!r}")

    samples: Dict[int, Dict[str, Any]] = {}
    seen_ids = set()
    for image in data.get("images", []):
        index = _sample_index_from_path(str(image.get("path", "")))
        if index in samples:
            raise ValueError(f"duplicate image index in annotations: {index}")
        elements = []
        for raw in image.get("annotations", []):
            element_id = str(raw.get("id", "")).strip()
            if not element_id:
                raise ValueError(f"sample {index} contains an annotation without id")
            if element_id in seen_ids:
                raise ValueError(f"duplicate annotation id: {element_id}")
            seen_ids.add(element_id)
            bbox = BBox(float(raw["x1"]), float(raw["y1"]), float(raw["x2"]), float(raw["y2"]))
            elements.append({
                "id": element_id,
                "description": str(raw.get("description", "")).strip(),
                "bbox": bbox,
            })
        if not elements:
            raise ValueError(f"sample {index} has no necessary evidence annotations")
        samples[index] = {
            "index": index,
            "image_path": str(image.get("path", "")),
            "elements": elements,
        }
    return samples


def load_predictions(path: Path) -> Dict[int, Dict[str, Any]]:
    """Load ``rig_results.json`` and assign stable IDs to predicted elements."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("results", data.get("samples", []))
    if not isinstance(data, list):
        raise ValueError("prediction file must be a list or contain a results/samples list")

    samples: Dict[int, Dict[str, Any]] = {}
    for raw_sample in data:
        index = int(raw_sample["index"])
        if index in samples:
            raise ValueError(f"duplicate prediction index: {index}")
        elements = []
        for position, raw in enumerate(raw_sample.get("elements", [])):
            try:
                bbox = BBox.from_value(raw["bbox"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid prediction bbox at sample {index}, element {position}") from exc
            elements.append({
                "id": str(raw.get("id") or f"pred_{position:03d}"),
                "description": str(raw.get("description", "")).strip(),
                "bbox": bbox,
            })
        samples[index] = {
            "index": index,
            "elements": elements,
            "answer_score": _answer_score(raw_sample),
            "question": str(raw_sample.get("question", "")),
            "inst_category": raw_sample.get("inst_category"),
            "category_name": raw_sample.get("category_name"),
        }
    return samples


def load_answer_scores(
    path: Path,
    index_column: str = "index",
    score_column: str = "score",
) -> Dict[int, float]:
    """Load answer scores from an Excel/CSV/TSV table, keyed by dataset index."""
    import pandas as pd

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        table = pd.read_excel(path)
    elif suffix == ".csv":
        table = pd.read_csv(path)
    elif suffix in {".tsv", ".txt"}:
        table = pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"unsupported answer-score file type: {path.suffix}")

    missing_columns = [name for name in (index_column, score_column) if name not in table.columns]
    if missing_columns:
        raise ValueError(f"answer-score table is missing columns: {missing_columns}")

    selected = table[[index_column, score_column]].dropna(subset=[index_column])
    duplicate_mask = selected[index_column].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_values = selected.loc[duplicate_mask, index_column].tolist()
        raise ValueError(f"duplicate answer-score indices: {duplicate_values[:10]}")

    scores: Dict[int, float] = {}
    for _, row in selected.iterrows():
        index = int(row[index_column])
        score = row[score_column]
        if pd.isna(score):
            continue
        scores[index] = float(score)
    return scores


def _answer_score(sample: Mapping[str, Any]) -> Optional[float]:
    for key in ("answer_score", "score"):
        value = sample.get(key)
        if value is not None and value != "":
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _pair_metrics(gt: BBox, pred: BBox) -> Dict[str, float]:
    intersection = gt.intersection_area(pred)
    return {
        "iou": gt.iou(pred),
        "gt_coverage": intersection / gt.area,
        "pred_coverage": intersection / pred.area,
        "pred_to_gt_area_ratio": pred.area / gt.area,
    }


def _is_candidate(metrics: Mapping[str, float], iou_threshold: float, config: EvaluationConfig) -> bool:
    strict_iou = metrics["iou"] >= iou_threshold
    controlled_containment = (
        metrics["gt_coverage"] >= config.containment_threshold
        and metrics["pred_to_gt_area_ratio"] <= config.max_pred_to_gt_area_ratio
    )
    return strict_iou or controlled_containment


def _gt_overlap_pairs(elements: Sequence[Mapping[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    pairs = []
    for left in range(len(elements)):
        for right in range(left + 1, len(elements)):
            a = elements[left]["bbox"]
            b = elements[right]["bbox"]
            intersection = a.intersection_area(b)
            if intersection <= 0:
                continue
            iou = a.iou(b)
            if iou <= threshold:
                continue
            pairs.append({
                "left_gt_id": elements[left]["id"],
                "right_gt_id": elements[right]["id"],
                "iou": iou,
                "left_coverage": a.coverage_by(b),
                "right_coverage": b.coverage_by(a),
            })
    return pairs


def audit_annotations(annotations: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    per_sample = []
    total_elements = 0
    total_pairs = 0
    for index in sorted(annotations):
        elements = annotations[index]["elements"]
        pairs = _gt_overlap_pairs(elements, 0.0)
        total_elements += len(elements)
        total_pairs += len(pairs)
        per_sample.append({
            "index": index,
            "n_gt": len(elements),
            "has_overlap": bool(pairs),
            "overlap_pair_count": len(pairs),
            "overlap_pairs": pairs,
        })
    counts = [row["n_gt"] for row in per_sample]
    return {
        "n_samples": len(per_sample),
        "n_elements": total_elements,
        "elements_per_sample": {
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "mean": mean(counts) if counts else 0.0,
        },
        "n_overlap_samples": sum(row["has_overlap"] for row in per_sample),
        "n_overlap_pairs": total_pairs,
        "samples": per_sample,
    }


def _candidate_pairs(
    gt_elements: Sequence[Mapping[str, Any]],
    pred_elements: Sequence[Mapping[str, Any]],
    iou_threshold: float,
    config: EvaluationConfig,
) -> List[Dict[str, Any]]:
    pairs = []
    for gt_pos, gt in enumerate(gt_elements):
        for pred_pos, pred in enumerate(pred_elements):
            metrics = _pair_metrics(gt["bbox"], pred["bbox"])
            if _is_candidate(metrics, iou_threshold, config):
                pairs.append({
                    "gt_pos": gt_pos,
                    "pred_pos": pred_pos,
                    "gt_id": gt["id"],
                    "pred_id": pred["id"],
                    **metrics,
                })
    return pairs


def _review_key(index: int, gt_id: str, pred_id: str) -> str:
    return f"{index}:{gt_id}:{pred_id}"


def load_review_decisions(path: Optional[Path]) -> Dict[str, bool]:
    if path is None or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    decisions: Dict[str, bool] = {}
    for row in data.get("decisions", []):
        decision = row.get("decision")
        if decision is None or decision == "":
            continue
        if not isinstance(decision, bool):
            raise ValueError("review decision must be true, false, or null")
        key = _review_key(int(row["index"]), str(row["gt_id"]), str(row["pred_id"]))
        decisions[key] = decision
    return decisions


def prepare_review(
    annotations: Mapping[int, Mapping[str, Any]],
    predictions: Mapping[int, Mapping[str, Any]],
    output_path: Path,
    config: EvaluationConfig,
    review_all: bool = False,
) -> Dict[str, Any]:
    """Create or refresh a JSON semantic-review form while preserving decisions."""
    existing_rows: Dict[str, Mapping[str, Any]] = {}
    if Path(output_path).exists():
        existing = json.loads(Path(output_path).read_text(encoding="utf-8"))
        for row in existing.get("decisions", []):
            existing_rows[_review_key(int(row["index"]), str(row["gt_id"]), str(row["pred_id"]))] = row

    threshold = min(config.iou_thresholds)
    rows = []
    for index in sorted(set(annotations) & set(predictions)):
        gt_elements = annotations[index]["elements"]
        pred_elements = predictions[index]["elements"]
        has_overlap = bool(_gt_overlap_pairs(gt_elements, config.gt_overlap_iou_threshold))
        if not review_all and not has_overlap:
            continue
        for pair in _candidate_pairs(gt_elements, pred_elements, threshold, config):
            gt = gt_elements[pair["gt_pos"]]
            pred = pred_elements[pair["pred_pos"]]
            key = _review_key(index, gt["id"], pred["id"])
            previous = existing_rows.get(key, {})
            rows.append({
                "index": index,
                "image_path": annotations[index].get("image_path", ""),
                "question": predictions[index].get("question", ""),
                "gt_id": gt["id"],
                "gt_description": gt["description"],
                "gt_bbox": gt["bbox"].as_list(),
                "pred_id": pred["id"],
                "pred_description": pred["description"],
                "pred_bbox": pred["bbox"].as_list(),
                "iou": round(pair["iou"], 6),
                "gt_coverage": round(pair["gt_coverage"], 6),
                "pred_to_gt_area_ratio": round(pair["pred_to_gt_area_ratio"], 6),
                "decision": previous.get("decision"),
                "notes": previous.get("notes", ""),
            })
    payload = {
        "format": "rig_semantic_review_v1",
        "instructions": (
            "Set decision=true only when the predicted item expresses the same necessary semantic clue as the GT item. "
            "Use false for spatially overlapping but semantically different items; leave null while undecided."
        ),
        "review_scope": "all_samples" if review_all else "gt_overlap_samples",
        "decisions": rows,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _maximum_weight_one_to_one(
    n_gt: int,
    n_pred: int,
    pairs: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    """Maximum-cardinality, then maximum-quality, one-to-one assignment."""
    size = max(n_gt, n_pred)
    if size == 0:
        return []
    pair_by_pos = {(int(p["gt_pos"]), int(p["pred_pos"])): p for p in pairs}
    weights = [[0.0] * size for _ in range(size)]
    for (gt_pos, pred_pos), pair in pair_by_pos.items():
        # The large base makes match count dominate match quality.
        weights[gt_pos][pred_pos] = 100.0 + float(pair["iou"]) + float(pair["gt_coverage"])

    max_weight = max(max(row) for row in weights)
    costs = [[max_weight - value for value in row] for row in weights]
    assignment = _hungarian_min(costs)
    matched = []
    for gt_pos, pred_pos in enumerate(assignment):
        pair = pair_by_pos.get((gt_pos, pred_pos))
        if pair is not None:
            matched.append(pair)
    return matched


def _hungarian_min(cost: Sequence[Sequence[float]]) -> List[int]:
    """Return column assignment for a square minimization matrix (O(n^3))."""
    n = len(cost)
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def _union_area(rectangles: Iterable[BBox]) -> float:
    rectangles = list(rectangles)
    if not rectangles:
        return 0.0
    xs = sorted({coord for box in rectangles for coord in (box.x1, box.x2)})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (box.y1, box.y2) for box in rectangles if box.x1 < right and box.x2 > left
        )
        if not intervals:
            continue
        covered = 0.0
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start > end:
                covered += end - start
                start, end = next_start, next_end
            else:
                end = max(end, next_end)
        covered += end - start
        area += (right - left) * covered
    return area


def _union_intersection_area(left: Sequence[BBox], right: Sequence[BBox]) -> float:
    intersections = []
    for a in left:
        for b in right:
            x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
            x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
            if x1 < x2 and y1 < y2:
                intersections.append(BBox(x1, y1, x2, y2))
    return _union_area(intersections)


def _review_required(policy: str, has_overlap: bool) -> bool:
    return policy == "review_all" or (policy == "overlap_review" and has_overlap)


def _evaluate_profile(
    index: int,
    gt_elements: Sequence[Mapping[str, Any]],
    pred_elements: Sequence[Mapping[str, Any]],
    has_overlap: bool,
    iou_threshold: float,
    config: EvaluationConfig,
    decisions: Mapping[str, bool],
) -> Dict[str, Any]:
    candidates = _candidate_pairs(gt_elements, pred_elements, iou_threshold, config)
    spatial_matches = _maximum_weight_one_to_one(len(gt_elements), len(pred_elements), candidates)
    needs_review = _review_required(config.semantic_policy, has_overlap)
    pending = []
    eligible = []
    for pair in candidates:
        key = _review_key(index, str(pair["gt_id"]), str(pair["pred_id"]))
        if not needs_review or config.semantic_policy == "spatial":
            eligible.append(pair)
        elif key not in decisions:
            pending.append(key)
        elif decisions[key]:
            eligible.append(pair)
    semantic_matches = _maximum_weight_one_to_one(len(gt_elements), len(pred_elements), eligible)
    finalized = not pending

    n_gt, n_pred = len(gt_elements), len(pred_elements)
    n_matched = len(semantic_matches)
    recall = n_matched / n_gt if n_gt else 1.0
    precision = n_matched / n_pred if n_pred else (1.0 if n_gt == 0 else 0.0)
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {
        "iou_threshold": iou_threshold,
        "finalized": finalized,
        "n_pending_decisions": len(pending),
        "spatial_upper_bound_matches": len(spatial_matches),
        "spatial_upper_bound_recall": len(spatial_matches) / n_gt if n_gt else 1.0,
        "n_matched": n_matched if finalized else None,
        "necessary_recall": recall if finalized else None,
        "precision": precision if finalized else None,
        "f1": f1 if finalized else None,
        "complete_evidence": (n_matched == n_gt) if finalized else None,
        "mean_matched_iou": (
            mean(float(pair["iou"]) for pair in semantic_matches)
            if finalized and semantic_matches else None
        ),
        "matches": [
            {
                "gt_id": pair["gt_id"],
                "pred_id": pair["pred_id"],
                "iou": pair["iou"],
                "gt_coverage": pair["gt_coverage"],
            }
            for pair in semantic_matches
        ] if finalized else [],
    }


def evaluate_predictions(
    annotations: Mapping[int, Mapping[str, Any]],
    predictions: Mapping[int, Mapping[str, Any]],
    config: EvaluationConfig,
    review_path: Optional[Path] = None,
    answer_scores: Optional[Mapping[int, float]] = None,
) -> Dict[str, Any]:
    decisions = load_review_decisions(review_path)
    sample_rows = []
    for index in sorted(annotations):
        gt_elements = annotations[index]["elements"]
        prediction = predictions.get(
            index,
            {
                "elements": [],
                "answer_score": None,
                "inst_category": None,
                "category_name": None,
            },
        )
        pred_elements = prediction["elements"]
        answer_score = (
            answer_scores.get(index)
            if answer_scores is not None and index in answer_scores
            else prediction.get("answer_score")
        )
        overlap_pairs = _gt_overlap_pairs(gt_elements, config.gt_overlap_iou_threshold)
        gt_boxes = [item["bbox"] for item in gt_elements]
        pred_boxes = [item["bbox"] for item in pred_elements]
        gt_union = _union_area(gt_boxes)
        pred_union = _union_area(pred_boxes)
        profiles = {
            _threshold_name(threshold): _evaluate_profile(
                index, gt_elements, pred_elements, bool(overlap_pairs), threshold, config, decisions
            )
            for threshold in config.iou_thresholds
        }
        sample_rows.append({
            "index": index,
            "has_prediction": index in predictions,
            "has_gt_overlap": bool(overlap_pairs),
            "n_gt_overlap_pairs": len(overlap_pairs),
            "n_gt": len(gt_elements),
            "n_pred": len(pred_elements),
            "answer_score": answer_score,
            "inst_category": prediction.get("inst_category"),
            "category_name": prediction.get("category_name"),
            "mean_best_iou": (
                mean(max((gt.iou(pred) for pred in pred_boxes), default=0.0) for gt in gt_boxes)
                if gt_boxes else 1.0
            ),
            "gt_union_area": gt_union,
            "pred_union_area": pred_union,
            "pred_union_area_ratio": pred_union / 1_000_000.0,
            "union_region_coverage": _union_intersection_area(gt_boxes, pred_boxes) / gt_union if gt_union else 1.0,
            "profiles": profiles,
        })

    aggregates = {
        name: _aggregate_profile(sample_rows, name)
        for name in (_threshold_name(t) for t in config.iou_thresholds)
    }
    return {
        "config": {
            "iou_thresholds": list(config.iou_thresholds),
            "containment_threshold": config.containment_threshold,
            "max_pred_to_gt_area_ratio": config.max_pred_to_gt_area_ratio,
            "semantic_policy": config.semantic_policy,
        },
        "summary": {
            "n_annotated_samples": len(annotations),
            "n_prediction_samples": sum(row["has_prediction"] for row in sample_rows),
            "n_missing_prediction_samples": sum(not row["has_prediction"] for row in sample_rows),
            "n_overlap_samples": sum(row["has_gt_overlap"] for row in sample_rows),
            "mean_union_region_coverage": mean(row["union_region_coverage"] for row in sample_rows) if sample_rows else 0.0,
            "mean_pred_count": mean(row["n_pred"] for row in sample_rows) if sample_rows else 0.0,
            "mean_pred_union_area_ratio": mean(row["pred_union_area_ratio"] for row in sample_rows) if sample_rows else 0.0,
            "mean_best_iou": mean(row["mean_best_iou"] for row in sample_rows) if sample_rows else 0.0,
            "profiles": aggregates,
            "by_inst_category": {
                str(category): {
                    name: _aggregate_profile(
                        [row for row in sample_rows if row["inst_category"] == category], name
                    )
                    for name in aggregates
                }
                for category in sorted({
                    row["inst_category"] for row in sample_rows if row["inst_category"] is not None
                }, key=str)
            },
        },
        "samples": sample_rows,
    }


def _threshold_name(value: float) -> str:
    return f"match_at_{value:g}"


def _aggregate_profile(samples: Sequence[Mapping[str, Any]], profile_name: str) -> Dict[str, Any]:
    finalized = [row for row in samples if row["profiles"][profile_name]["finalized"]]
    pending = [row for row in samples if not row["profiles"][profile_name]["finalized"]]
    profile_rows = [row["profiles"][profile_name] for row in finalized]

    def subset_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        profiles = [row["profiles"][profile_name] for row in rows if row["profiles"][profile_name]["finalized"]]
        return {
            "n_samples": len(profiles),
            "macro_necessary_recall": mean(p["necessary_recall"] for p in profiles) if profiles else None,
            "complete_evidence_rate": mean(float(p["complete_evidence"]) for p in profiles) if profiles else None,
        }

    answer_known = [row for row in finalized if row.get("answer_score") is not None]
    answer_correct = [row for row in answer_known if float(row["answer_score"]) > 0]
    answer_incorrect = [row for row in answer_known if float(row["answer_score"]) <= 0]
    return {
        "n_finalized_samples": len(finalized),
        "n_pending_samples": len(pending),
        "pending_indices": [row["index"] for row in pending],
        "macro_necessary_recall": mean(p["necessary_recall"] for p in profile_rows) if profile_rows else None,
        "micro_necessary_recall": (
            sum(p["n_matched"] for p in profile_rows) / sum(row["n_gt"] for row in finalized)
            if finalized and sum(row["n_gt"] for row in finalized) else None
        ),
        "macro_precision": mean(p["precision"] for p in profile_rows) if profile_rows else None,
        "mean_matched_iou": mean(
            p["mean_matched_iou"] for p in profile_rows if p["mean_matched_iou"] is not None
        ) if any(p["mean_matched_iou"] is not None for p in profile_rows) else None,
        "complete_evidence_rate": mean(float(p["complete_evidence"]) for p in profile_rows) if profile_rows else None,
        "spatial_upper_bound_recall": mean(
            row["profiles"][profile_name]["spatial_upper_bound_recall"] for row in samples
        ) if samples else None,
        "overlap_subset": subset_stats([row for row in finalized if row["has_gt_overlap"]]),
        "non_overlap_subset": subset_stats([row for row in finalized if not row["has_gt_overlap"]]),
        "answer_conditioned": {
            "n_with_answer_score": len(answer_known),
            "correct": subset_stats(answer_correct),
            "incorrect": subset_stats(answer_incorrect),
        },
    }

"""Finalize the five-question H2 adaptive-gate semantic adjudication.

This script is intentionally tied to the frozen candidate sets produced by the
27B repeated-single@6 and static-question-plan@6 calibration runs.  It refuses
to write decisions if the candidate identities change, so regenerating a
review form cannot silently alter the adjudicated population.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from rig_eval import EvaluationConfig, evaluate_predictions, load_annotations, load_predictions


ROOT = Path(__file__).resolve().parent
ANNOTATIONS = ROOT.parent / "RIG-Annotating-main/annotation of RIG/Charxiv_reasoning_val/annotation_1.json"
BASE = ROOT / "test_case/RIG_Charxiv/h2_grounding/feasibility/adaptive-gate"
INDICES = (1, 17, 8, 244, 346)

RUNS = {
    "repeated-single@6": {
        "directory": BASE / "Qwen3.5-27B__repeated-single-at6",
        "candidate_count": 131,
        "candidate_key_sha256": "866a132858299983ae7eed7b6af5688f290068194fc40f41ac39c79c68876a84",
    },
    "static-question-plan@6": {
        "directory": BASE / "Qwen3.5-27B__static-question-plan-at6",
        "candidate_count": 63,
        "candidate_key_sha256": "4ce6efb2812da6b549a138d877e6fef94dc5e3fd6a9251eee8e9055feda641b2",
    },
}

Key = Tuple[int, str, str]


# Every candidate not listed here was manually reviewed as expressing the same
# necessary semantic clue as its GT item.  The explicit false set captures the
# recurring ambiguity classes and is guarded by the complete candidate digest.
FALSE_DECISIONS: Dict[str, Dict[Key, str]] = {
    "repeated-single@6": {
        (1, "729e3599", "pred_020"): "Wrong series: red lambda_L=0.20, not blue lambda_L=0.30.",
        (1, "729e3599", "pred_032"): "Wrong series: red lambda_L=0.20, not blue lambda_L=0.30.",
        (1, "c638eef1", "pred_021"): "Wrong series: blue lambda_L=0.30, not red lambda_L=0.20.",
        (1, "c638eef1", "pred_033"): "Wrong series: blue lambda_L=0.30, not red lambda_L=0.20.",
        (1, "f46f7a34", "pred_020"): "Wrong series: red lambda_L=0.20, not green lambda_L=0.13.",
        (1, "f46f7a34", "pred_032"): "Wrong series: red lambda_L=0.20, not green lambda_L=0.13.",
        (8, "d8e00dbd", "pred_008"): "Prediction identifies the Xception text-label region, not the Xception data point.",
        (244, "2e6f8d03", "pred_093"): "One non-zero curve is not the full first-coordinate-2 curve set.",
        (244, "3abc2185", "pred_090"): "The single (0,0) curve is not the full first-coordinate-0 curve set.",
        (244, "45403871", "pred_101"): "One non-zero curve is not the full first-coordinate-6 curve set.",
        (244, "4fab7f1a", "pred_091"): "The single (1,0) curve is not the full first-coordinate-1 curve set.",
        (244, "67368efe", "pred_105"): "One non-zero curve is not the full first-coordinate-8 curve set.",
        (244, "7bd9af33", "pred_103"): "One non-zero curve is not the full first-coordinate-7 curve set.",
        (244, "94e5ccfe", "pred_095"): "One non-zero curve is not the full first-coordinate-3 curve set.",
        (244, "a2bd8746", "pred_098"): "The single (5,0) curve is not the full first-coordinate-5 curve set.",
        (244, "a2bd8746", "pred_099"): "One non-zero curve is not the full first-coordinate-5 curve set.",
        (244, "a6d4048d", "pred_097"): "One non-zero curve is not the full first-coordinate-4 curve set.",
        (244, "d0f1f68f", "pred_002"): "Prediction identifies the axis/ticks but not the x-axis label text.",
    },
    "static-question-plan@6": {
        (244, "2e6f8d03", "pred_045"): "The single (2,0) curve is not the full first-coordinate-2 curve set.",
        (244, "3abc2185", "pred_043"): "The single (0,0) curve is not the full first-coordinate-0 curve set.",
        (244, "45403871", "pred_053"): "The single (6,0) curve is not the full first-coordinate-6 curve set.",
        (244, "4fab7f1a", "pred_044"): "The single (1,0) curve is not the full first-coordinate-1 curve set.",
        (244, "94e5ccfe", "pred_047"): "The single (3,0) curve is not the full first-coordinate-3 curve set.",
        (244, "a2bd8746", "pred_051"): "The single (5,0) curve is not the full first-coordinate-5 curve set.",
    },
}


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def candidate_key(row: Mapping[str, Any]) -> Key:
    return int(row["index"]), str(row["gt_id"]), str(row["pred_id"])


def key_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    text = "\n".join(
        f"{index}|{gt_id}|{pred_id}"
        for index, gt_id, pred_id in (candidate_key(row) for row in rows)
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_record(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def review_note(row: Mapping[str, Any]) -> str:
    gt_description = str(row.get("gt_description", "")).lower()
    pred_description = str(row.get("pred_description", "")).lower()
    if "lines of cells" in gt_description and ("chart" in pred_description or "plot area" in pred_description):
        return (
            "Accepted as the same grouped-curve evidence: the controlled panel region contains the complete "
            "first-coordinate curve set. Its spatial breadth remains penalized by union-area metrics."
        )
    return "Same necessary semantic clue."


def finalize_review(name: str, spec: Mapping[str, Any]) -> Dict[str, Any]:
    path = Path(spec["directory"]) / "semantic_review.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("decisions", [])
    if len(rows) != int(spec["candidate_count"]):
        raise RuntimeError(f"{name}: expected {spec['candidate_count']} candidates, found {len(rows)}")
    digest = key_digest(rows)
    if digest != spec["candidate_key_sha256"]:
        raise RuntimeError(f"{name}: candidate-key digest changed: {digest}")

    candidate_keys = {candidate_key(row) for row in rows}
    unknown_false_keys = set(FALSE_DECISIONS[name]) - candidate_keys
    if unknown_false_keys:
        raise RuntimeError(f"{name}: false-decision keys absent from candidate set: {sorted(unknown_false_keys)}")

    for row in rows:
        key = candidate_key(row)
        reason = FALSE_DECISIONS[name].get(key)
        row["decision"] = reason is None
        row["notes"] = reason or review_note(row)

    n_false = len(FALSE_DECISIONS[name])
    payload["adjudication"] = {
        "protocol": "H2 five-question overlap-review adjudication v1",
        "indices_with_manual_review": [1, 8, 244],
        "source_material": ["GT description", "prediction description", "GT/prediction bbox", "source image"],
        "decision_rule": (
            "True only for the same necessary semantic clue. Wrong-series overlaps, label/point substitutions, "
            "and a single curve substituted for a required full curve set are false. A controlled subplot region "
            "covering the complete named curve set is semantically accepted and separately penalized by area."
        ),
        "candidate_count": len(rows),
        "candidate_key_sha256": digest,
        "true_count": len(rows) - n_false,
        "false_count": n_false,
        "pending_count": 0,
    }
    atomic_json(path, payload)
    return payload["adjudication"]


def macro_f1(samples: Sequence[Mapping[str, Any]], profile: str) -> float:
    values = []
    for sample in samples:
        row = sample["profiles"][profile]
        recall, precision = float(row["necessary_recall"]), float(row["precision"])
        values.append(2 * recall * precision / (recall + precision) if recall + precision else 0.0)
    return mean(values) if values else 0.0


def reviewed_report(directory: Path) -> Dict[str, Any]:
    annotations = load_annotations(ANNOTATIONS)
    annotations = {index: annotations[index] for index in INDICES}
    predictions = load_predictions(directory / "rig_results.json")
    if set(predictions) != set(INDICES):
        raise RuntimeError(f"{directory.name}: predictions do not exactly cover {list(INDICES)}")
    report = evaluate_predictions(
        annotations,
        predictions,
        EvaluationConfig(iou_thresholds=(0.3, 0.5), semantic_policy="overlap_review"),
        directory / "semantic_review.json",
    )
    for profile in ("match_at_0.3", "match_at_0.5"):
        report["summary"]["profiles"][profile]["macro_f1"] = macro_f1(report["samples"], profile)
    output = directory / "evaluation_report_reviewed.json"
    atomic_json(output, report)
    return report


def compact_summary(report: Mapping[str, Any]) -> Dict[str, Any]:
    summary = report["summary"]
    result: Dict[str, Any] = {
        "n_samples": summary["n_annotated_samples"],
        "mean_pred_count": summary["mean_pred_count"],
        "mean_pred_union_area_ratio": summary["mean_pred_union_area_ratio"],
        "profiles": {},
    }
    for profile in ("match_at_0.3", "match_at_0.5"):
        row = summary["profiles"][profile]
        result["profiles"][profile] = {
            "n_finalized_samples": row["n_finalized_samples"],
            "n_pending_samples": row["n_pending_samples"],
            "macro_necessary_recall": row["macro_necessary_recall"],
            "micro_necessary_recall": row["micro_necessary_recall"],
            "macro_precision": row["macro_precision"],
            "macro_f1": row["macro_f1"],
            "complete_evidence_rate": row["complete_evidence_rate"],
            "spatial_upper_bound_recall": row["spatial_upper_bound_recall"],
        }
    return result


def paired_recall_analysis(
    repeated_report: Mapping[str, Any],
    static_report: Mapping[str, Any],
    profile: str = "match_at_0.3",
    *,
    resamples: int = 10_000,
    seed: int = 20260903,
) -> Dict[str, Any]:
    repeated = {
        int(row["index"]): float(row["profiles"][profile]["necessary_recall"])
        for row in repeated_report["samples"]
    }
    static = {
        int(row["index"]): float(row["profiles"][profile]["necessary_recall"])
        for row in static_report["samples"]
    }
    if set(repeated) != set(static):
        raise RuntimeError("paired reports do not cover the same question indices")
    rows = [
        {
            "index": index,
            "repeated_single": repeated[index],
            "static_question_plan": static[index],
            "delta_static_minus_repeated": static[index] - repeated[index],
        }
        for index in sorted(repeated)
    ]
    values = [row["delta_static_minus_repeated"] for row in rows]
    generator = random.Random(seed)
    bootstrap = sorted(
        mean(generator.choice(values) for _ in values)
        for _ in range(resamples)
    )
    lower = bootstrap[int(0.025 * resamples)]
    upper = bootstrap[int(0.975 * resamples) - 1]
    return {
        "profile": profile,
        "paired_unit": "question",
        "per_question": rows,
        "mean_delta": mean(values),
        "bootstrap": {
            "method": "paired question-level percentile bootstrap",
            "resamples": resamples,
            "seed": seed,
            "confidence_level": 0.95,
            "interval": [lower, upper],
        },
    }


def inference_cost(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "calls": len(rows),
        "valid_calls": sum(row.get("status") == "valid" for row in rows),
        "prompt_tokens": sum(int((row.get("token_usage") or {}).get("prompt") or 0) for row in rows),
        "completion_tokens": sum(int((row.get("token_usage") or {}).get("completion") or 0) for row in rows),
        "total_tokens": sum(int((row.get("token_usage") or {}).get("total") or 0) for row in rows),
        "elapsed_seconds": sum(float(row.get("elapsed_seconds") or 0.0) for row in rows),
    }


def add_costs(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: left[key] + right[key] for key in left}


def main() -> int:
    adjudications = {name: finalize_review(name, spec) for name, spec in RUNS.items()}
    reports = {name: reviewed_report(Path(spec["directory"])) for name, spec in RUNS.items()}
    summary = {
        "experiment": "H2 27B five-question adaptive-gate reviewed comparison",
        "indices": list(INDICES),
        "semantic_policy": "overlap_review",
        "adjudications": adjudications,
        "runs": {name: compact_summary(report) for name, report in reports.items()},
    }
    repeated = summary["runs"]["repeated-single@6"]["profiles"]["match_at_0.3"]
    static = summary["runs"]["static-question-plan@6"]["profiles"]["match_at_0.3"]
    summary["primary_delta_static_minus_repeated"] = {
        "macro_necessary_recall": static["macro_necessary_recall"] - repeated["macro_necessary_recall"],
        "macro_precision": static["macro_precision"] - repeated["macro_precision"],
        "macro_f1": static["macro_f1"] - repeated["macro_f1"],
    }
    summary["paired_primary_analysis"] = paired_recall_analysis(
        reports["repeated-single@6"], reports["static-question-plan@6"]
    )
    repeated_steps = json.loads(
        (Path(RUNS["repeated-single@6"]["directory"]) / "step_outputs.json").read_text(encoding="utf-8")
    )
    static_steps = json.loads(
        (Path(RUNS["static-question-plan@6"]["directory"]) / "step_outputs.json").read_text(encoding="utf-8")
    )
    static_plans = json.loads(
        (BASE / "Qwen3.5-27B__static-question-plan-at6__plan-only/plans.json").read_text(encoding="utf-8")
    )
    repeated_cost = inference_cost(repeated_steps)
    static_grounding_cost = inference_cost(static_steps)
    static_planning_cost = inference_cost(static_plans)
    summary["costs"] = {
        "repeated_single_grounding": repeated_cost,
        "static_question_planning": static_planning_cost,
        "static_question_grounding": static_grounding_cost,
        "static_question_total": add_costs(static_planning_cost, static_grounding_cost),
        "image_calls": {"repeated-single@6": 30, "static-question-plan@6": 30},
    }
    output = BASE / "Qwen3.5-27B__adaptive-gate-5-reviewed-summary.json"
    atomic_json(output, summary)
    frozen_assets = [
        Path(RUNS["repeated-single@6"]["directory"]) / "rig_results.json",
        Path(RUNS["repeated-single@6"]["directory"]) / "semantic_review.json",
        Path(RUNS["repeated-single@6"]["directory"]) / "evaluation_report_reviewed.json",
        Path(RUNS["static-question-plan@6"]["directory"]) / "rig_results.json",
        Path(RUNS["static-question-plan@6"]["directory"]) / "semantic_review.json",
        Path(RUNS["static-question-plan@6"]["directory"]) / "evaluation_report_reviewed.json",
        BASE / "Qwen3.5-27B__static-question-plan-at6__plan-only/plans.json",
        output,
    ]
    manifest = {
        "format": "H2 adaptive-gate five-question adjudication manifest v1",
        "scope": "reviewed calibration assets; not a 41-question formal-run freeze",
        "files": [file_record(path) for path in frozen_assets],
    }
    manifest_path = BASE / "Qwen3.5-27B__adaptive-gate-5-reviewed-manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

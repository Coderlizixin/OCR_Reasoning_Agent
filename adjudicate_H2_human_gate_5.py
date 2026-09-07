"""Finalize and compare the five-question Human@6 adaptive-gate run.

The script is locked to the reviewed Human candidate population.  It refuses
to write if any candidate identity changes, then evaluates Human@6 with the
same overlap-review rules already used for Repeated@6 and Static@6.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from adjudicate_H2_adaptive_gate_5 import (
    ANNOTATIONS,
    BASE,
    INDICES,
    compact_summary,
    key_digest,
    macro_f1,
)
from rig_eval import EvaluationConfig, evaluate_predictions, load_annotations, load_predictions


HUMAN_DIR = BASE / "Qwen3.5-27B__human-image-aware-plan-at6"
CANDIDATE_COUNT = 44
CANDIDATE_KEY_SHA256 = "dcb65d8e25ff63c92c9d161a6302e65f843353d51d7a0a3be4baf1cf52128e91"
FALSE: Dict[Tuple[int, str, str], str] = {
    (1, "c638eef1", "pred_009"): "Wrong series: blue lambda_L=0.30, not red lambda_L=0.20.",
    (1, "729e3599", "pred_007"): "Wrong series: red lambda_L=0.20, not blue lambda_L=0.30.",
}


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_review() -> Dict[str, Any]:
    path = HUMAN_DIR / "semantic_review.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("decisions", [])
    if len(rows) != CANDIDATE_COUNT or key_digest(rows) != CANDIDATE_KEY_SHA256:
        raise RuntimeError("Human@6 semantic candidate population changed; refusing to adjudicate")
    keys = {(int(r["index"]), str(r["gt_id"]), str(r["pred_id"])) for r in rows}
    if not set(FALSE) <= keys:
        raise RuntimeError("A pre-reviewed false candidate is missing")
    for row in rows:
        key = int(row["index"]), str(row["gt_id"]), str(row["pred_id"])
        reason = FALSE.get(key)
        row["decision"] = reason is None
        if reason:
            row["notes"] = reason
        elif "lines of cells" in str(row["gt_description"]).lower() and "subplot" in str(row["pred_description"]).lower():
            row["notes"] = (
                "Accepted as the same grouped-curve evidence: the controlled subplot region contains the "
                "named first-coordinate curve set; spatial breadth remains penalized by union area."
            )
        else:
            row["notes"] = "Same necessary semantic clue."
    payload["adjudication"] = {
        "protocol": "H2 five-question overlap-review adjudication v1",
        "indices_with_manual_review": [1, 8, 244],
        "source_material": ["GT description", "prediction description", "GT/prediction bbox", "source image"],
        "decision_rule": (
            "True only for the same necessary semantic clue. Wrong-series overlaps are false. A controlled "
            "subplot region containing the complete named curve set is accepted and separately penalized by area."
        ),
        "candidate_count": len(rows),
        "candidate_key_sha256": CANDIDATE_KEY_SHA256,
        "true_count": len(rows) - len(FALSE),
        "false_count": len(FALSE),
        "pending_count": 0,
    }
    atomic_json(path, payload)
    return payload["adjudication"]


def reviewed_report(predictions_path: Path) -> Dict[str, Any]:
    all_annotations = load_annotations(ANNOTATIONS)
    annotations = {index: all_annotations[index] for index in INDICES}
    predictions = load_predictions(predictions_path)
    if set(predictions) != set(INDICES):
        raise RuntimeError(f"prediction indices differ from the five-question cohort: {sorted(predictions)}")
    report = evaluate_predictions(
        annotations,
        predictions,
        EvaluationConfig(iou_thresholds=(0.3, 0.5), semantic_policy="overlap_review"),
        HUMAN_DIR / "semantic_review.json",
    )
    for profile in ("match_at_0.3", "match_at_0.5"):
        report["summary"]["profiles"][profile]["macro_f1"] = macro_f1(report["samples"], profile)
    return report


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_bootstrap(human: Mapping[int, float], other: Mapping[int, float]) -> Dict[str, Any]:
    differences = [human[index] - other[index] for index in INDICES]
    rng = random.Random(20260903)
    draws = [mean(rng.choices(differences, k=len(differences))) for _ in range(10_000)]
    return {
        "paired_unit": "question",
        "resamples": 10_000,
        "seed": 20260903,
        "mean_delta_human_minus_other": mean(differences),
        "bootstrap_95_ci": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "per_question": {str(index): human[index] - other[index] for index in INDICES},
    }


def main() -> int:
    adjudication = finalize_review()
    human_report = reviewed_report(HUMAN_DIR / "rig_results.json")
    atomic_json(HUMAN_DIR / "evaluation_report_reviewed.json", human_report)

    reports = {
        "repeated-single@6": json.loads((BASE / "Qwen3.5-27B__repeated-single-at6/evaluation_report_reviewed.json").read_text(encoding="utf-8")),
        "static-question-plan@6": json.loads((BASE / "Qwen3.5-27B__static-question-plan-at6/evaluation_report_reviewed.json").read_text(encoding="utf-8")),
        "human-image-aware-plan@6": human_report,
    }
    prefixes = []
    for k in range(1, 7):
        report = reviewed_report(HUMAN_DIR / f"prefix_results/k{k}/rig_results.json")
        row = compact_summary(report)
        prefixes.append({"k": k, **row})

    human_recall = {
        int(row["index"]): float(row["profiles"]["match_at_0.3"]["necessary_recall"])
        for row in human_report["samples"]
    }
    comparisons = {}
    for name in ("repeated-single@6", "static-question-plan@6"):
        other = {
            int(row["index"]): float(row["profiles"]["match_at_0.3"]["necessary_recall"])
            for row in reports[name]["samples"]
        }
        comparisons[name] = paired_bootstrap(human_recall, other)

    step_rows = json.loads((HUMAN_DIR / "step_outputs.json").read_text(encoding="utf-8"))
    summary = {
        "experiment": "H2 27B five-question adaptive-gate reviewed comparison including Human@6",
        "indices": list(INDICES),
        "semantic_policy": "overlap_review",
        "human_adjudication": adjudication,
        "runs": {name: compact_summary(report) for name, report in reports.items()},
        "human_prefixes": prefixes,
        "paired_recall_0_3": comparisons,
        "human_grounding_cost": {
            "visual_calls": len(step_rows),
            "total_tokens": sum((row.get("token_usage") or {}).get("total") or 0 for row in step_rows),
            "elapsed_seconds": sum(float(row.get("elapsed_seconds") or 0) for row in step_rows),
            "human_planning_time": "unavailable",
        },
        "interpretation_limits": [
            "Five-question feasibility calibration, not a formal 41-question result.",
            "Human planning time is unavailable and excluded from cost comparisons.",
            "Human@6 is image-aware open-loop planning, not closed-loop adaptive planning.",
        ],
    }
    output = BASE / "Qwen3.5-27B__adaptive-gate-5-human-reviewed-summary.json"
    atomic_json(output, summary)
    manifest = {
        "files": [
            {"path": str(path.relative_to(BASE)).replace("\\", "/"), "size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (
                HUMAN_DIR / "rig_results.json",
                HUMAN_DIR / "step_outputs.json",
                HUMAN_DIR / "semantic_review.json",
                HUMAN_DIR / "evaluation_report_reviewed.json",
                output,
            )
        ]
    }
    atomic_json(BASE / "Qwen3.5-27B__adaptive-gate-5-human-reviewed-manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

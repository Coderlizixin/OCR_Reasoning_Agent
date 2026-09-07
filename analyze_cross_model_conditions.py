"""Build a paired 9B/27B summary from immutable condition reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest


DEFAULT_BOOTSTRAP_SAMPLES = 200_000
DEFAULT_BOOTSTRAP_SEED = 20260920
REPORT_VERSION = "cross-model-condition-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_ci(
    values: Sequence[float], *, samples: int, seed: int, confidence: float = 0.95
) -> list[float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array):
        raise ValueError("paired bootstrap requires a non-empty one-dimensional vector")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    batch_size = min(20_000, samples)
    cursor = 0
    while cursor < samples:
        count = min(batch_size, samples - cursor)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        estimates[cursor : cursor + count] = array[indices].mean(axis=1)
        cursor += count
    alpha = 1.0 - confidence
    return [
        float(np.quantile(estimates, alpha / 2)),
        float(np.quantile(estimates, 1 - alpha / 2)),
    ]


def load_report(path: Path) -> Dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    required = {"model_name", "sample_count", "sources", "overall", "per_item"}
    missing = sorted(required - set(report))
    if missing:
        raise ValueError(f"report {path} lacks fields: {missing}")
    return report


def verify_frozen_sources(report: Mapping[str, Any]) -> list[Dict[str, Any]]:
    verified = []
    for source in report["sources"]:
        marker = Path(source["freeze_manifest_path"])
        workbook = Path(source["judge_workbook_path"])
        if sha256_file(marker) != source["freeze_manifest_sha256"]:
            raise RuntimeError(f"freeze manifest changed since report creation: {marker}")
        if sha256_file(workbook) != source["judge_workbook_sha256"]:
            raise RuntimeError(f"judge workbook changed since report creation: {workbook}")
        payload = json.loads(marker.read_text(encoding="utf-8"))
        run_dir = Path(source["run_directory"])
        mismatches = []
        for item in payload.get("files", []):
            file_path = run_dir / item["path"]
            if not file_path.is_file():
                mismatches.append({"path": item["path"], "reason": "missing"})
            elif file_path.stat().st_size != int(item["size_bytes"]):
                mismatches.append({"path": item["path"], "reason": "size_changed"})
            elif sha256_file(file_path) != item["sha256"]:
                mismatches.append({"path": item["path"], "reason": "content_changed"})
        if mismatches:
            raise RuntimeError(f"frozen source integrity failed in {run_dir}: {mismatches[:5]}")
        verified.append(
            {
                "condition_id": source["condition_id"],
                "judge_workbook_path": str(workbook.resolve()),
                "judge_workbook_sha256": source["judge_workbook_sha256"],
                "freeze_manifest_path": str(marker.resolve()),
                "freeze_manifest_sha256": source["freeze_manifest_sha256"],
                "frozen_file_count": int(source["frozen_file_count"]),
                "integrity_verified": True,
            }
        )
    return verified


def by_id(rows: Iterable[Mapping[str, Any]], key: str) -> Dict[str, Mapping[str, Any]]:
    return {str(row[key]): row for row in rows}


def score(item: Mapping[str, Any], condition_id: str) -> float:
    return float(item[f"{condition_id}_score"])


def exact_mcnemar(first: Sequence[float], second: Sequence[float]) -> Dict[str, Any]:
    first_array = np.asarray(first, dtype=int)
    second_array = np.asarray(second, dtype=int)
    first_only = int(np.sum((first_array == 1) & (second_array == 0)))
    second_only = int(np.sum((first_array == 0) & (second_array == 1)))
    discordant = first_only + second_only
    p_value = float(binomtest(first_only, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "paired_test": "exact_mcnemar_binomial",
        "first_only": first_only,
        "second_only": second_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-9b", type=Path, required=True)
    parser.add_argument("--report-27b", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_paths = [args.report_9b.resolve(), args.report_27b.resolve()]
    reports = [load_report(path) for path in report_paths]
    model_names = [str(report["model_name"]) for report in reports]
    if len(set(model_names)) != 2:
        raise ValueError(f"expected two distinct models, got {model_names}")
    if [int(report["sample_count"]) for report in reports] != [41, 41]:
        raise ValueError("cross-model closure requires exactly 41 samples per model")

    per_model = {name: report for name, report in zip(model_names, reports)}
    small_name, large_name = model_names
    item_maps = {
        name: {int(item["index"]): item for item in report["per_item"]}
        for name, report in per_model.items()
    }
    expected_indices = sorted(item_maps[small_name])
    if sorted(item_maps[large_name]) != expected_indices or len(expected_indices) != 41:
        raise ValueError("the two reports do not contain the same 41 indices")

    condition_rows = {name: by_id(report["overall"], "condition_id") for name, report in per_model.items()}
    condition_ids = list(condition_rows[small_name])
    if list(condition_rows[large_name]) != condition_ids:
        raise ValueError("condition order or membership differs between reports")

    verified_sources = {
        name: verify_frozen_sources(report) for name, report in per_model.items()
    }
    overall = []
    for offset, condition_id in enumerate(condition_ids):
        small_values = [score(item_maps[small_name][index], condition_id) for index in expected_indices]
        large_values = [score(item_maps[large_name][index], condition_id) for index in expected_indices]
        paired = np.asarray(large_values) - np.asarray(small_values)
        ci = bootstrap_ci(
            paired,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + offset,
        )
        item = {
            "condition_id": condition_id,
            "display_name": condition_rows[small_name][condition_id]["display_name"],
            "models": {
                small_name: dict(condition_rows[small_name][condition_id]),
                large_name: dict(condition_rows[large_name][condition_id]),
            },
            "delta_27b_minus_9b": float(paired.mean()),
            "delta_27b_minus_9b_pp": float(paired.mean() * 100),
            "bootstrap_ci_95": ci,
            "bootstrap_ci_95_pp": [value * 100 for value in ci],
        }
        if condition_id != "random_mask_in_mean":
            item.update(exact_mcnemar(large_values, small_values))
        else:
            item.update({"paired_test": "not_applicable_to_three_seed_mean", "p_value": None})
        overall.append(item)

    pair_maps = {
        name: by_id(report["pairwise_comparisons"], "comparison_id")
        for name, report in per_model.items()
    }
    comparison_ids = list(pair_maps[small_name])
    if list(pair_maps[large_name]) != comparison_ids:
        raise ValueError("pairwise comparison membership differs between reports")
    effects = []
    for offset, comparison_id in enumerate(comparison_ids, start=len(condition_ids)):
        small_pair = pair_maps[small_name][comparison_id]
        large_pair = pair_maps[large_name][comparison_id]
        treatment_id = str(small_pair["treatment_id"])
        reference_id = str(small_pair["reference_id"])
        if (treatment_id, reference_id) != (
            str(large_pair["treatment_id"]),
            str(large_pair["reference_id"]),
        ):
            raise ValueError(f"comparison definition differs for {comparison_id}")
        interactions = []
        for index in expected_indices:
            small_item = item_maps[small_name][index]
            large_item = item_maps[large_name][index]
            small_effect = score(small_item, treatment_id) - score(small_item, reference_id)
            large_effect = score(large_item, treatment_id) - score(large_item, reference_id)
            interactions.append(large_effect - small_effect)
        ci = bootstrap_ci(
            interactions,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + offset,
        )
        effects.append(
            {
                "comparison_id": comparison_id,
                "treatment_id": treatment_id,
                "reference_id": reference_id,
                "models": {
                    small_name: dict(small_pair),
                    large_name: dict(large_pair),
                },
                "interaction_27b_minus_9b": float(np.mean(interactions)),
                "interaction_27b_minus_9b_pp": float(np.mean(interactions) * 100),
                "bootstrap_ci_95": ci,
                "bootstrap_ci_95_pp": [value * 100 for value in ci],
            }
        )

    per_item = []
    for index in expected_indices:
        small_item = item_maps[small_name][index]
        large_item = item_maps[large_name][index]
        item = {
            "index": index,
            "inst_category": int(small_item["inst_category"]),
            "category_name": small_item["category_name"],
            "conditions": {},
        }
        for condition_id in condition_ids:
            item["conditions"][condition_id] = {
                small_name: {
                    "score": score(small_item, condition_id),
                    "status": small_item.get(f"{condition_id}_status"),
                },
                large_name: {
                    "score": score(large_item, condition_id),
                    "status": large_item.get(f"{condition_id}_status"),
                },
                "delta_27b_minus_9b": score(large_item, condition_id)
                - score(small_item, condition_id),
            }
        per_item.append(item)

    categories = []
    for name, report in per_model.items():
        for row in report["category_statistics"]:
            categories.append({"model_name": name, **row})

    input_reports = []
    for name, path, report in zip(model_names, report_paths, reports):
        input_reports.append(
            {
                "model_name": name,
                "path": str(path),
                "sha256": sha256_file(path),
                "generated_at": report.get("generated_at"),
                "source_count": len(report["sources"]),
                "frozen_file_count": sum(
                    int(source["frozen_file_count"]) for source in report["sources"]
                ),
                "all_sources_frozen_and_verified": True,
            }
        )

    output = {
        "report_version": REPORT_VERSION,
        "generated_at": utc_now(),
        "models": model_names,
        "sample_count": 41,
        "pairing": "same 41 CharXiv question indices; no independent-sample pooling",
        "fixed_completion_token_budget": 12288,
        "truncation_policy": "finish_reason=length counts as an incorrect model outcome",
        "bootstrap": {
            "method": "question-level paired percentile bootstrap",
            "samples": args.bootstrap_samples,
            "base_seed": args.bootstrap_seed,
            "confidence_level": 0.95,
        },
        "input_reports": input_reports,
        "verified_frozen_sources": verified_sources,
        "overall_by_condition": overall,
        "hypothesis_effects": effects,
        "category_statistics": categories,
        "per_item": per_item,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_json.resolve()}")
    print(
        json.dumps(
            {
                "models": model_names,
                "samples": len(expected_indices),
                "conditions": len(overall),
                "hypothesis_effects": len(effects),
                "verified_frozen_sources": sum(map(len, verified_sources.values())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Build an auditable comparison report for CharXiv evidence conditions.

The script reads frozen judge workbooks, verifies their SHA-256 manifests, computes
paired statistics, and writes a JSON report.  Excel authoring is intentionally kept
in ``build_condition_comparison_workbook.mjs`` so the workbook uses the bundled
artifact-tool runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EVIDENCE_ROOT = SCRIPT_DIR / "test_case/RIG_Charxiv/evidence_conditions"
DEFAULT_GT_MASK_ROOT = SCRIPT_DIR / "test_case/RIG_Charxiv/gt_mask_in"
DEFAULT_RANDOM_SEEDS = (20260901, 20260902, 20260903)
DEFAULT_BOOTSTRAP_SAMPLES = 200_000
DEFAULT_BOOTSTRAP_SEED = 20260902
REPORT_VERSION = "condition-comparison-v1"

CATEGORY_NAMES = {
    1: "Text-in-Chart",
    2: "Text-in-General",
    3: "Number-in-Chart",
    4: "Number-in-General",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    result = "".join(character if character in allowed else "_" for character in value)
    return result.strip("._")


def source_specs(
    *,
    model_name: str,
    evidence_root: Path,
    gt_mask_root: Path,
    random_seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    model = safe_name(model_name)
    sources: List[Dict[str, Any]] = [
        {
            "condition_id": "full_image",
            "display_name": "Full Image",
            "run_dir": evidence_root / "full-image" / f"{model}__evidence-conditions-v1",
            "workbook_name": "predictions_qwen-plus.xlsx",
            "random_seed": None,
        },
        {
            "condition_id": "gt_mask_in",
            "display_name": "GT Mask-In",
            "run_dir": gt_mask_root / f"{model}__gt-mask-in-v1",
            "workbook_name": "predictions_for_judge_qwen-plus.xlsx",
            "random_seed": None,
        },
        {
            "condition_id": "gt_mask_out",
            "display_name": "GT Mask-Out",
            "run_dir": evidence_root / "gt-mask-out" / f"{model}__evidence-conditions-v1",
            "workbook_name": "predictions_qwen-plus.xlsx",
            "random_seed": None,
        },
    ]
    for position, seed in enumerate(random_seeds, start=1):
        sources.append(
            {
                "condition_id": f"random_mask_in_{position}",
                "display_name": f"Random Mask-In (seed {seed})",
                "run_dir": (
                    evidence_root
                    / "random-mask-in"
                    / f"{model}__evidence-conditions-v1__seed-{seed}"
                ),
                "workbook_name": "predictions_qwen-plus.xlsx",
                "random_seed": seed,
            }
        )
    sources.append(
        {
            "condition_id": "gt_roi",
            "display_name": "GT ROI",
            "run_dir": evidence_root / "gt-roi" / f"{model}__evidence-conditions-v1",
            "workbook_name": "predictions_qwen-plus.xlsx",
            "random_seed": None,
        }
    )
    return sources


def verify_frozen_manifest(run_dir: Path) -> Dict[str, Any]:
    marker = run_dir / "FROZEN.json"
    if not marker.is_file():
        raise FileNotFoundError(f"missing FROZEN.json: {run_dir}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    mismatches = []
    for item in payload.get("files", []):
        path = run_dir / item["path"]
        if not path.is_file():
            mismatches.append({"path": item["path"], "reason": "missing"})
            continue
        if path.stat().st_size != int(item["size_bytes"]):
            mismatches.append({"path": item["path"], "reason": "size_changed"})
            continue
        if sha256_file(path) != item["sha256"]:
            mismatches.append({"path": item["path"], "reason": "content_changed"})
    if mismatches:
        raise RuntimeError(f"frozen run integrity failure in {run_dir}: {mismatches[:5]}")
    return {
        "freeze_manifest_path": str(marker.resolve()),
        "freeze_manifest_sha256": sha256_file(marker),
        "frozen_at": payload.get("frozen_at"),
        "frozen_file_count": int(payload.get("file_count", len(payload.get("files", [])))),
        "integrity_verified": True,
    }


def load_condition(spec: Mapping[str, Any]) -> tuple[pd.DataFrame, Dict[str, Any]]:
    run_dir = Path(spec["run_dir"])
    workbook = run_dir / str(spec["workbook_name"])
    if not workbook.is_file():
        raise FileNotFoundError(f"missing judge workbook: {workbook}")
    freeze = verify_frozen_manifest(run_dir)
    data = pd.read_excel(workbook)
    required = {"index", "inst_category", "score", "gt_mask_answer_status", "gt_mask_finish_reason"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{workbook} lacks required columns: {missing}")
    data = data.copy()
    data["index"] = pd.to_numeric(data["index"], errors="raise").astype(int)
    data["inst_category"] = pd.to_numeric(data["inst_category"], errors="raise").astype(int)
    data["score"] = pd.to_numeric(data["score"], errors="coerce").fillna(0).astype(int)
    if not set(data["score"]).issubset({0, 1}):
        raise ValueError(f"score must be binary in {workbook}")
    if data["index"].duplicated().any():
        raise ValueError(f"duplicate index in {workbook}")
    metadata = {
        **{key: spec[key] for key in ("condition_id", "display_name", "random_seed")},
        "run_directory": str(run_dir.resolve()),
        "judge_workbook_path": str(workbook.resolve()),
        "judge_workbook_sha256": sha256_file(workbook),
        **freeze,
    }
    return data.sort_values("index").reset_index(drop=True), metadata


def overall_row(condition_id: str, display_name: str, data: pd.DataFrame) -> Dict[str, Any]:
    valid = data["gt_mask_answer_status"].eq("valid")
    truncated = data["gt_mask_answer_status"].eq("truncated")
    total = len(data)
    correct = int(data["score"].sum())
    valid_count = int(valid.sum())
    return {
        "condition_id": condition_id,
        "display_name": display_name,
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "valid_answers": valid_count,
        "valid_answer_rate": valid_count / total,
        "truncated": int(truncated.sum()),
        "truncation_rate": float(truncated.mean()),
        "conditional_accuracy": float(data.loc[valid, "score"].mean()) if valid_count else None,
    }


def category_rows(condition_id: str, display_name: str, data: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    for category_id, category_name in CATEGORY_NAMES.items():
        subset = data[data["inst_category"].eq(category_id)]
        rows.append(
            {
                "condition_id": condition_id,
                "display_name": display_name,
                "inst_category": category_id,
                "category_name": category_name,
                "correct": int(subset["score"].sum()),
                "total": len(subset),
                "accuracy": float(subset["score"].mean()),
            }
        )
    return rows


def bootstrap_ci(
    differences: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> List[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    estimates = differences[indices].mean(axis=1)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def paired_comparison(
    *,
    comparison_id: str,
    treatment_id: str,
    reference_id: str,
    frames: Mapping[str, pd.DataFrame],
    bootstrap_samples: int,
    bootstrap_seed: int,
    random_mean_ids: Sequence[str] = (),
) -> Dict[str, Any]:
    treatment = frames[treatment_id][["index", "score"]].rename(columns={"score": "treatment"})
    if random_mean_ids:
        reference = frames[random_mean_ids[0]][["index", "score"]].rename(
            columns={"score": random_mean_ids[0]}
        )
        for condition_id in random_mean_ids[1:]:
            reference = reference.merge(
                frames[condition_id][["index", "score"]].rename(columns={"score": condition_id}),
                on="index",
                validate="one_to_one",
            )
        reference["reference"] = reference[list(random_mean_ids)].mean(axis=1)
        reference = reference[["index", "reference"]]
    else:
        reference = frames[reference_id][["index", "score"]].rename(columns={"score": "reference"})
    paired = treatment.merge(reference, on="index", validate="one_to_one")
    differences = (paired["treatment"] - paired["reference"]).to_numpy(dtype=float)
    result: Dict[str, Any] = {
        "comparison_id": comparison_id,
        "treatment_id": treatment_id,
        "reference_id": reference_id,
        "delta_accuracy": float(differences.mean()),
        "delta_accuracy_pp": float(differences.mean() * 100),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_ci_95": bootstrap_ci(
            differences,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
    }
    result["bootstrap_ci_95_pp"] = [value * 100 for value in result["bootstrap_ci_95"]]
    if random_mean_ids:
        result.update(
            {
                "paired_test": "not_applicable_to_fractional_random_seed_mean",
                "mcnemar_exact_p": None,
                "treatment_only_indices": [],
                "reference_only_indices": [],
                "both_correct_indices": [],
                "both_wrong_indices": [],
            }
        )
    else:
        treatment_only = paired[
            paired["treatment"].eq(1) & paired["reference"].eq(0)
        ]["index"].astype(int).tolist()
        reference_only = paired[
            paired["treatment"].eq(0) & paired["reference"].eq(1)
        ]["index"].astype(int).tolist()
        both_correct = paired[
            paired["treatment"].eq(1) & paired["reference"].eq(1)
        ]["index"].astype(int).tolist()
        both_wrong = paired[
            paired["treatment"].eq(0) & paired["reference"].eq(0)
        ]["index"].astype(int).tolist()
        discordant = len(treatment_only) + len(reference_only)
        p_value = (
            float(binomtest(min(len(treatment_only), len(reference_only)), discordant, 0.5).pvalue)
            if discordant
            else 1.0
        )
        result.update(
            {
                "paired_test": "exact_mcnemar_binomial",
                "mcnemar_exact_p": p_value,
                "treatment_only_indices": treatment_only,
                "reference_only_indices": reference_only,
                "both_correct_indices": both_correct,
                "both_wrong_indices": both_wrong,
            }
        )
    return result


def build_per_item(frames: Mapping[str, pd.DataFrame], random_ids: Sequence[str]) -> List[Dict[str, Any]]:
    first = next(iter(frames.values()))
    items = pd.DataFrame(
        {
            "index": first["index"],
            "inst_category": first["inst_category"],
            "category_name": first["inst_category"].map(CATEGORY_NAMES),
        }
    )
    for condition_id, data in frames.items():
        columns = data[["index", "score", "gt_mask_answer_status", "gt_mask_finish_reason"]].rename(
            columns={
                "score": f"{condition_id}_score",
                "gt_mask_answer_status": f"{condition_id}_status",
                "gt_mask_finish_reason": f"{condition_id}_finish_reason",
            }
        )
        items = items.merge(columns, on="index", validate="one_to_one")
    items["random_mask_in_mean_score"] = items[
        [f"{condition_id}_score" for condition_id in random_ids]
    ].mean(axis=1)
    items["gt_in_minus_full"] = items["gt_mask_in_score"] - items["full_image_score"]
    items["full_minus_gt_out"] = items["full_image_score"] - items["gt_mask_out_score"]
    items["gt_in_minus_random_mean"] = (
        items["gt_mask_in_score"] - items["random_mask_in_mean_score"]
    )
    items["gt_roi_minus_gt_in"] = items["gt_roi_score"] - items["gt_mask_in_score"]
    return json.loads(items.to_json(orient="records"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen3.5-9B")
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--gt-mask-root", type=Path, default=DEFAULT_GT_MASK_ROOT)
    parser.add_argument("--random-seeds", type=int, nargs="+", default=list(DEFAULT_RANDOM_SEEDS))
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_EVIDENCE_ROOT / "condition_comparison_report.json",
    )
    parser.add_argument(
        "--hash-manifest",
        type=Path,
        default=DEFAULT_EVIDENCE_ROOT / "condition_hash_manifest.json",
        help="Consolidated immutable-file SHA-256 inventory for all source runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = source_specs(
        model_name=args.model_name,
        evidence_root=args.evidence_root,
        gt_mask_root=args.gt_mask_root,
        random_seeds=args.random_seeds,
    )
    frames: Dict[str, pd.DataFrame] = {}
    sources = []
    for spec in specs:
        data, metadata = load_condition(spec)
        frames[str(spec["condition_id"])] = data
        sources.append(metadata)
    expected_indices = frames["full_image"]["index"].tolist()
    for condition_id, data in frames.items():
        if data["index"].tolist() != expected_indices:
            raise ValueError(f"index set/order mismatch for {condition_id}")

    random_ids = [f"random_mask_in_{position}" for position in range(1, len(args.random_seeds) + 1)]
    overall = []
    categories = []
    for spec in specs:
        condition_id = str(spec["condition_id"])
        display_name = str(spec["display_name"])
        overall.append(overall_row(condition_id, display_name, frames[condition_id]))
        categories.extend(category_rows(condition_id, display_name, frames[condition_id]))

    random_accuracy_values = [
        next(row["accuracy"] for row in overall if row["condition_id"] == condition_id)
        for condition_id in random_ids
    ]
    overall.append(
        {
            "condition_id": "random_mask_in_mean",
            "display_name": "Random Mask-In (3-seed mean)",
            "correct": float(np.mean([frames[condition_id]["score"].sum() for condition_id in random_ids])),
            "total": len(expected_indices),
            "accuracy": float(np.mean(random_accuracy_values)),
            "valid_answers": float(
                np.mean(
                    [frames[condition_id]["gt_mask_answer_status"].eq("valid").sum() for condition_id in random_ids]
                )
            ),
            "valid_answer_rate": float(
                np.mean(
                    [frames[condition_id]["gt_mask_answer_status"].eq("valid").mean() for condition_id in random_ids]
                )
            ),
            "truncated": float(
                np.mean(
                    [
                        frames[condition_id]["gt_mask_answer_status"].eq("truncated").sum()
                        for condition_id in random_ids
                    ]
                )
            ),
            "truncation_rate": float(
                np.mean(
                    [
                        frames[condition_id]["gt_mask_answer_status"].eq("truncated").mean()
                        for condition_id in random_ids
                    ]
                )
            ),
            "conditional_accuracy": float(
                sum(frames[condition_id]["score"].sum() for condition_id in random_ids)
                / sum(
                    frames[condition_id]["gt_mask_answer_status"].eq("valid").sum()
                    for condition_id in random_ids
                )
            ),
        }
    )

    pair_specs = [
        ("gt_in_minus_full", "gt_mask_in", "full_image", ()),
        ("full_minus_gt_out", "full_image", "gt_mask_out", ()),
        ("gt_roi_minus_gt_in", "gt_roi", "gt_mask_in", ()),
    ]
    pair_specs.extend(
        (f"gt_in_minus_{condition_id}", "gt_mask_in", condition_id, ())
        for condition_id in random_ids
    )
    pair_specs.extend(
        [
            ("gt_in_minus_random_mean", "gt_mask_in", "random_mask_in_mean", tuple(random_ids)),
            ("gt_out_minus_random_mean", "gt_mask_out", "random_mask_in_mean", tuple(random_ids)),
        ]
    )
    comparisons = []
    for offset, (comparison_id, treatment_id, reference_id, random_mean_ids) in enumerate(pair_specs):
        comparisons.append(
            paired_comparison(
                comparison_id=comparison_id,
                treatment_id=treatment_id,
                reference_id=reference_id,
                frames=frames,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed + offset,
                random_mean_ids=random_mean_ids,
            )
        )

    generated_at = utc_now()
    hash_manifest_sources = []
    for source in sources:
        freeze_manifest_path = Path(source["freeze_manifest_path"])
        freeze_payload = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
        hash_manifest_sources.append(
            {
                "condition_id": source["condition_id"],
                "display_name": source["display_name"],
                "random_seed": source["random_seed"],
                "run_directory": source["run_directory"],
                "freeze_manifest_path": source["freeze_manifest_path"],
                "freeze_manifest_sha256": source["freeze_manifest_sha256"],
                "frozen_at": source["frozen_at"],
                "file_count": source["frozen_file_count"],
                "files": freeze_payload.get("files", []),
            }
        )
    hash_manifest = {
        "manifest_version": "condition-hash-manifest-v1",
        "generated_at": generated_at,
        "model_name": args.model_name,
        "hash_algorithm": "SHA-256",
        "all_sources_frozen_and_verified": True,
        "source_count": len(hash_manifest_sources),
        "total_file_count": sum(source["file_count"] for source in hash_manifest_sources),
        "sources": hash_manifest_sources,
    }
    args.hash_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.hash_manifest.write_text(
        json.dumps(hash_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "report_version": REPORT_VERSION,
        "generated_at": generated_at,
        "model_name": args.model_name,
        "sample_count": len(expected_indices),
        "fixed_completion_token_budget": 12288,
        "truncation_policy": "finish_reason=length counts as an incorrect model outcome",
        "category_mapping": {str(key): value for key, value in CATEGORY_NAMES.items()},
        "random_seeds": list(args.random_seeds),
        "bootstrap": {
            "method": "sample-level paired percentile bootstrap",
            "samples": args.bootstrap_samples,
            "base_seed": args.bootstrap_seed,
            "confidence_level": 0.95,
        },
        "sources": sources,
        "overall": overall,
        "pairwise_comparisons": comparisons,
        "category_statistics": categories,
        "per_item": build_per_item(frames, random_ids),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_json.resolve()}")
    print(f"Wrote {args.hash_manifest.resolve()}")
    print(
        json.dumps(
            {
                "model": args.model_name,
                "samples": len(expected_indices),
                "conditions": len(overall),
                "comparisons": len(comparisons),
                "all_sources_frozen_and_verified": all(
                    source["integrity_verified"] for source in sources
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

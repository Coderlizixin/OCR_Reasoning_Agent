"""Command-line entry point for RIG necessary-evidence evaluation."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from rig_eval import (
    EvaluationConfig,
    audit_annotations,
    evaluate_predictions,
    load_annotations,
    load_answer_scores,
    load_predictions,
    prepare_review,
)


DEFAULT_ANNOTATIONS = Path("../RIG-Annotating-main/annotation of RIG/Charxiv_reasoning_val/annotation_1.json")
DEFAULT_PREDICTIONS = Path("test_case/RIG_Charxiv/rig_results.json")
DEFAULT_REVIEW = Path("test_case/RIG_Charxiv/semantic_review.json")
DEFAULT_REPORT = Path("test_case/RIG_Charxiv/evaluation_report.json")


def _config(args: argparse.Namespace) -> EvaluationConfig:
    return EvaluationConfig(
        iou_thresholds=tuple(args.iou_thresholds),
        containment_threshold=args.containment_threshold,
        max_pred_to_gt_area_ratio=args.max_area_ratio,
        semantic_policy=getattr(args, "semantic_policy", "overlap_review"),
    )


def _add_matching_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.3, 0.5])
    parser.add_argument("--containment-threshold", type=float, default=0.7)
    parser.add_argument("--max-area-ratio", type=float, default=4.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="validate and summarize annotation_1.json")
    audit.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    audit.add_argument("--output", type=Path)
    audit.add_argument("--expect-samples", type=int, default=41)

    review = subparsers.add_parser("prepare-review", help="create semantic pair review JSON")
    review.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    review.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    review.add_argument("--output", type=Path, default=DEFAULT_REVIEW)
    review.add_argument("--review-all", action="store_true")
    _add_matching_args(review)

    evaluate = subparsers.add_parser("evaluate", help="calculate evidence metrics")
    evaluate.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    evaluate.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    evaluate.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    evaluate.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    evaluate.add_argument(
        "--answer-scores",
        type=Path,
        help="Optional Excel/CSV/TSV containing answer scores joined by dataset index.",
    )
    evaluate.add_argument("--answer-index-column", default="index")
    evaluate.add_argument("--answer-score-column", default="score")
    evaluate.add_argument(
        "--semantic-policy",
        choices=["spatial", "overlap_review", "review_all"],
        default="overlap_review",
    )
    _add_matching_args(evaluate)
    return parser


def _write_or_print(payload: dict, output: Path = None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"Saved: {output}")


def main(argv: Sequence[str] = None) -> int:
    args = build_parser().parse_args(argv)
    annotations = load_annotations(args.annotations)

    if args.command == "audit":
        if args.expect_samples and len(annotations) != args.expect_samples:
            raise SystemExit(
                f"Expected {args.expect_samples} formal samples, found {len(annotations)} in {args.annotations}"
            )
        _write_or_print(audit_annotations(annotations), args.output)
        return 0

    predictions = load_predictions(args.predictions)
    config = _config(args)
    if args.command == "prepare-review":
        payload = prepare_review(annotations, predictions, args.output, config, review_all=args.review_all)
        print(f"Saved {len(payload['decisions'])} review pairs: {args.output}")
        return 0

    answer_scores = None
    if args.answer_scores:
        answer_scores = load_answer_scores(
            args.answer_scores,
            index_column=args.answer_index_column,
            score_column=args.answer_score_column,
        )
        missing_score_indices = sorted(set(annotations) - set(answer_scores))
        if missing_score_indices:
            raise SystemExit(
                "Answer-score table does not cover all annotated samples; "
                f"missing indices: {missing_score_indices}"
            )
        print(
            f"Loaded answer scores for all {len(annotations)} annotated indices "
            f"from {args.answer_scores}"
        )
    report = evaluate_predictions(
        annotations,
        predictions,
        config,
        args.review,
        answer_scores=answer_scores,
    )
    if args.answer_scores:
        report["config"]["answer_scores"] = {
            "source": str(args.answer_scores),
            "index_column": args.answer_index_column,
            "score_column": args.answer_score_column,
            "join": "dataset_index",
        }
    _write_or_print(report, args.output)
    summary = report["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

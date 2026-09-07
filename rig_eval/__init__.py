"""Evaluation utilities for Reasoning Image Grounding (RIG)."""

from .core import (
    BBox,
    EvaluationConfig,
    audit_annotations,
    evaluate_predictions,
    load_annotations,
    load_answer_scores,
    load_predictions,
    prepare_review,
)

__all__ = [
    "BBox",
    "EvaluationConfig",
    "audit_annotations",
    "evaluate_predictions",
    "load_annotations",
    "load_answer_scores",
    "load_predictions",
    "prepare_review",
]

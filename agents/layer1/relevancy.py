"""
agents/layer1/relevancy.py
===========================
Layer 1 — Relevancy check step.

Inspects SpeciesNet detection output to decide whether the image
contains wildlife or should be discarded.

Converted from Relevancy.txt (Semantic Kernel → plain Python function).
Logic and thresholds are identical to the reference implementation.
"""
from __future__ import annotations

import logging
from agents.layer1.schemas import RelevancyResult

logger = logging.getLogger(__name__)

# ── Thresholds & label sets ───────────────────────────────────────────────────

DETECTION_SCORE_THRESHOLD = 0.15

NON_WILDLIFE_LABELS: frozenset[str] = frozenset({
    "blank", "vehicle", "human", "person", "no_cv_result",
})


def check_relevancy(prediction: dict) -> RelevancyResult:
    """
    Inspect SpeciesNet bounding-box detections to decide if the image is
    wildlife-relevant.

    SpeciesNet prediction structure:
        {
            "detections": [
                {"label": "animal", "conf": 0.93, "bbox": [...], "category": "1"},
                ...
            ],
            "classifications": {"classes": [...], "scores": [...]}
        }

    Returns ``RelevancyResult(is_relevant=True)`` when:
      1. At least one detection exists.
      2. Top detection label is NOT in NON_WILDLIFE_LABELS.
      3. Top detection confidence >= DETECTION_SCORE_THRESHOLD (0.15).

    Falls back to checking classifications if detections list is absent.
    """
    detections: list[dict] = prediction.get("detections") or []

    # ── No detections at all — fall back to classifications presence ──────────
    if not detections:
        classifications = prediction.get("classifications") or {}
        has_classes = bool(classifications.get("classes"))
        if has_classes:
            logger.debug("No detections but classifications present — treating as relevant.")
            return RelevancyResult(
                is_relevant=True,
                reason="No detections but classifications present — treating as relevant.",
            )
        logger.debug("No detections and no classifications — blank or corrupted frame.")
        return RelevancyResult(
            is_relevant=False,
            reason="No detections and no classifications — frame may be blank or corrupted.",
        )

    # Detections are sorted by confidence descending by SpeciesNet
    top: dict = detections[0]
    top_label = str(top.get("label", "")).lower().strip()
    top_score = float(top.get("conf", 0.0))

    # ── Explicitly non-wildlife detection ─────────────────────────────────────
    if top_label in NON_WILDLIFE_LABELS:
        logger.debug("Top detection is non-wildlife: label=%s", top_label)
        return RelevancyResult(
            is_relevant=False,
            reason=f"Top detection is '{top_label}' — not a wildlife subject.",
            top_detection_label=top_label,
            top_detection_score=top_score,
        )

    # ── Detection below confidence floor ──────────────────────────────────────
    if top_score < DETECTION_SCORE_THRESHOLD:
        logger.debug("Detection below threshold: conf=%.2f", top_score)
        return RelevancyResult(
            is_relevant=False,
            reason=(
                f"Detection confidence {top_score:.1%} is below the "
                f"{DETECTION_SCORE_THRESHOLD:.0%} minimum threshold."
            ),
            top_detection_label=top_label,
            top_detection_score=top_score,
        )

    # ── Wildlife detected with sufficient confidence ───────────────────────────
    logger.debug("Wildlife detected: label=%s conf=%.2f", top_label, top_score)
    return RelevancyResult(
        is_relevant=True,
        reason=f"Detected '{top_label}' with {top_score:.1%} confidence.",
        top_detection_label=top_label,
        top_detection_score=top_score,
    )

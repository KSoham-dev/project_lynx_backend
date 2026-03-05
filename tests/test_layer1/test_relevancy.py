"""
tests/test_layer1/test_relevancy.py
=====================================
Unit tests for agents/layer1/relevancy.py — all 5 paths of check_relevancy().
"""
import pytest
from agents.layer1.relevancy import check_relevancy, DETECTION_SCORE_THRESHOLD


def _make_prediction(detections=None, classifications=None):
    """Build a minimal SpeciesNet prediction dict."""
    pred = {}
    if detections is not None:
        pred["detections"] = detections
    if classifications is not None:
        pred["classifications"] = classifications
    return pred


# ── No detections ─────────────────────────────────────────────────────────────

def test_no_detections_no_classifications_is_not_relevant():
    result = check_relevancy(_make_prediction(detections=[]))
    assert result.is_relevant is False
    assert "blank or corrupted" in result.reason.lower()


def test_no_detections_with_classifications_is_relevant():
    pred = _make_prediction(
        detections=[],
        classifications={"classes": ["animalia;…;felidae;panthera;tigris"], "scores": [0.92]},
    )
    result = check_relevancy(pred)
    assert result.is_relevant is True


def test_missing_detections_key_treated_as_empty():
    """prediction has no 'detections' key at all."""
    pred = {"classifications": {"classes": ["animalia;…;rattus;rattus"], "scores": [0.7]}}
    result = check_relevancy(pred)
    assert result.is_relevant is True


# ── Non-wildlife labels ───────────────────────────────────────────────────────

@pytest.mark.parametrize("label", ["blank", "vehicle", "human", "person", "no_cv_result"])
def test_non_wildlife_label_is_not_relevant(label):
    pred = _make_prediction(
        detections=[{"label": label, "conf": 0.95, "bbox": [0, 0, 1, 1]}]
    )
    result = check_relevancy(pred)
    assert result.is_relevant is False
    assert label in result.reason


# ── Low confidence ────────────────────────────────────────────────────────────

def test_detection_below_threshold_is_not_relevant():
    pred = _make_prediction(
        detections=[{"label": "animal", "conf": DETECTION_SCORE_THRESHOLD - 0.01, "bbox": []}]
    )
    result = check_relevancy(pred)
    assert result.is_relevant is False
    assert "threshold" in result.reason.lower()


# ── Wildlife detected ─────────────────────────────────────────────────────────

def test_animal_above_threshold_is_relevant():
    pred = _make_prediction(
        detections=[{"label": "animal", "conf": 0.93, "bbox": [0.1, 0.1, 0.9, 0.9]}]
    )
    result = check_relevancy(pred)
    assert result.is_relevant is True
    assert result.top_detection_label == "animal"
    assert result.top_detection_score == pytest.approx(0.93)


def test_exactly_at_threshold_is_relevant():
    pred = _make_prediction(
        detections=[{"label": "animal", "conf": DETECTION_SCORE_THRESHOLD, "bbox": []}]
    )
    result = check_relevancy(pred)
    assert result.is_relevant is True

"""
tests/test_predict.py
---------------------
Tests for:
  POST /predict           (single-image classification)
  POST /predict/batch     (multi-image classification)
"""

from unittest.mock import MagicMock, patch

import pytest

import main

# ── Helpers ────────────────────────────────────────────────────────────────────

VALID_PAYLOAD = {
    "image_url": "https://example.com/lion.jpg",
    "latitude": -1.29,
    "longitude": 36.82,
}

VALID_PAYLOAD_NO_GPS = {"image_url": "https://example.com/elephant.jpg"}


def _prediction(classes=None, scores=None) -> dict:
    return {
        "classifications": {
            "classes": classes or ["Panthera leo;lion"],
            "scores": scores or [0.95],
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# POST /predict
# ══════════════════════════════════════════════════════════════════════════════


class TestPredictSingle:

    def test_503_when_model_not_loaded(self, client_no_model):
        resp = client_no_model.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 503
        assert "not loaded" in resp.json()["detail"].lower()

    def test_200_on_valid_request(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_response_schema(self, client):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "image_url" in body
        assert "top_prediction" in body
        assert "classifications" in body
        assert "raw" in body

    def test_image_url_echoed(self, client):
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert body["image_url"] == VALID_PAYLOAD["image_url"]

    def test_top_prediction_is_first_class(self, client, mock_model):
        mock_model.predict.return_value = {"predictions": [_prediction(["Elephas maximus;elephant"])]}
        body = client.post("/predict", json=VALID_PAYLOAD).json()
        assert body["top_prediction"] == "Elephas maximus;elephant"

    def test_classifications_list_structure(self, client, mock_model):
        mock_model.predict.return_value = {
            "predictions": [_prediction(["Cls A", "Cls B"], [0.8, 0.2])]
        }
        classifications = client.post("/predict", json=VALID_PAYLOAD).json()["classifications"]
        assert len(classifications) == 2
        assert classifications[0]["label"] == "Cls A"
        assert isinstance(classifications[0]["score"], float)

    def test_scores_rounded_to_6dp(self, client, mock_model):
        mock_model.predict.return_value = {
            "predictions": [_prediction(["X"], [0.123456789])]
        }
        score = client.post("/predict", json=VALID_PAYLOAD).json()["classifications"][0]["score"]
        assert len(str(score).split(".")[-1]) <= 6

    def test_request_without_gps(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD_NO_GPS)
        assert resp.status_code == 200

    def test_422_missing_image_url(self, client):
        resp = client.post("/predict", json={"latitude": 1.0})
        assert resp.status_code == 422

    def test_422_empty_body(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_500_on_model_exception(self, client, mock_model):
        mock_model.predict.side_effect = RuntimeError("GPU OOM")
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 500
        assert "Prediction error" in resp.json()["detail"]
        mock_model.predict.side_effect = None  # reset for other tests

    def test_500_on_empty_predictions(self, client, mock_model):
        mock_model.predict.return_value = {"predictions": []}
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 500

    def test_gps_forwarded_to_model(self, client, mock_model):
        mock_model.predict.return_value = {"predictions": [_prediction()]}
        client.post("/predict", json=VALID_PAYLOAD)
        call_args = mock_model.predict.call_args
        instance = call_args.kwargs["instances_dict"]["instances"][0]
        assert instance["latitude"] == VALID_PAYLOAD["latitude"]
        assert instance["longitude"] == VALID_PAYLOAD["longitude"]

    def test_gps_omitted_when_not_provided(self, client, mock_model):
        mock_model.predict.return_value = {"predictions": [_prediction()]}
        client.post("/predict", json=VALID_PAYLOAD_NO_GPS)
        instance = mock_model.predict.call_args.kwargs["instances_dict"]["instances"][0]
        assert "latitude" not in instance
        assert "longitude" not in instance


# ══════════════════════════════════════════════════════════════════════════════
# POST /predict/batch
# ══════════════════════════════════════════════════════════════════════════════

BATCH_PAYLOAD = [
    {"image_url": "https://example.com/lion.jpg", "latitude": -1.29, "longitude": 36.82},
    {"image_url": "https://example.com/elephant.jpg"},
]


class TestPredictBatch:

    def test_503_when_model_not_loaded(self, client_no_model):
        resp = client_no_model.post("/predict/batch", json=BATCH_PAYLOAD)
        assert resp.status_code == 503

    def test_400_on_empty_list(self, client):
        resp = client.post("/predict/batch", json=[])
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_200_on_valid_batch(self, client, mock_model):
        mock_model.predict.return_value = {
            "predictions": [_prediction(), _prediction(["Loxodonta africana;elephant"])]
        }
        resp = client.post("/predict/batch", json=BATCH_PAYLOAD)
        assert resp.status_code == 200

    def test_response_is_list(self, client, mock_model):
        mock_model.predict.return_value = {
            "predictions": [_prediction(), _prediction()]
        }
        body = client.post("/predict/batch", json=BATCH_PAYLOAD).json()
        assert isinstance(body, list)

    def test_response_length_matches_input(self, client, mock_model):
        mock_model.predict.return_value = {
            "predictions": [_prediction(), _prediction()]
        }
        body = client.post("/predict/batch", json=BATCH_PAYLOAD).json()
        assert len(body) == len(BATCH_PAYLOAD)

    def test_each_item_has_image_url(self, client, mock_model):
        mock_model.predict.return_value = {
            "predictions": [_prediction(), _prediction()]
        }
        body = client.post("/predict/batch", json=BATCH_PAYLOAD).json()
        for item, req in zip(body, BATCH_PAYLOAD):
            assert item["image_url"] == req["image_url"]

    def test_500_on_model_exception(self, client, mock_model):
        mock_model.predict.side_effect = ValueError("bad input")
        resp = client.post("/predict/batch", json=BATCH_PAYLOAD)
        assert resp.status_code == 500
        mock_model.predict.side_effect = None

    def test_single_image_batch(self, client, mock_model):
        mock_model.predict.return_value = {"predictions": [_prediction()]}
        resp = client.post("/predict/batch", json=[VALID_PAYLOAD])
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_all_instances_forwarded(self, client, mock_model):
        mock_model.predict.return_value = {
            "predictions": [_prediction(), _prediction()]
        }
        client.post("/predict/batch", json=BATCH_PAYLOAD)
        instances = mock_model.predict.call_args.kwargs["instances_dict"]["instances"]
        assert len(instances) == 2
        assert instances[0]["filepath"] == BATCH_PAYLOAD[0]["image_url"]
        assert instances[1]["filepath"] == BATCH_PAYLOAD[1]["image_url"]

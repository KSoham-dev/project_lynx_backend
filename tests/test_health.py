"""
tests/test_health.py
--------------------
Tests for:
  GET /          (root / welcome)
  GET /health    (model-loaded status)
"""

import main


def test_root_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_root_body(client):
    body = client.get("/").json()
    assert body["status"] == "ok"
    assert "message" in body
    assert "Prahari" in body["message"]


def test_health_model_loaded(client):
    """With mock model injected the flag must be True."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_health_model_not_loaded(client_no_model):
    """When main.model is None the flag must be False."""
    resp = client_no_model.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is False


def test_root_content_type(client):
    resp = client.get("/")
    assert "application/json" in resp.headers["content-type"]


def test_health_content_type(client):
    resp = client.get("/health")
    assert "application/json" in resp.headers["content-type"]

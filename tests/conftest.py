"""
tests/conftest.py
-----------------
Shared fixtures for the Prahari test suite.

Strategy
--------
* The FastAPI lifespan tries to `from speciesnet import SpeciesNet` and load a
  heavy PyTorch model.  We stub the entire `speciesnet` package in sys.modules
  before any import so the lifespan succeeds without touching real weights.
* Two clients are provided:
    - `client`          – app running with a mock SpeciesNet model injected
    - `client_no_model` – app where main.model is explicitly None (→ 503 paths)
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Stub out `speciesnet` before main.py is imported ──────────────────────────
_speciesnet_stub = types.ModuleType("speciesnet")
_speciesnet_stub.SpeciesNet = MagicMock()          # type: ignore[attr-defined]
sys.modules.setdefault("speciesnet", _speciesnet_stub)

import main  # noqa: E402  must come AFTER the stub


# ── Reusable mock model ────────────────────────────────────────────────────────
def _make_mock_model(
    classes: list[str] | None = None,
    scores: list[float] | None = None,
) -> MagicMock:
    """Return a MagicMock that mimics SpeciesNet.predict() output."""
    classes = classes or ["Panthera leo;lion", "Panthera tigris;tiger"]
    scores = scores or [0.92, 0.05]

    prediction = {
        "classifications": {
            "classes": classes,
            "scores": scores,
        },
        "extra_field": "ignored",
    }
    mock = MagicMock()
    mock.predict.return_value = {"predictions": [prediction]}
    return mock


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mock_model() -> MagicMock:
    """A single mock model instance reused across the whole session."""
    return _make_mock_model()


@pytest.fixture()
def client(mock_model: MagicMock) -> TestClient:
    """
    TestClient with a mock SpeciesNet model already loaded.
    The lifespan is bypassed by patching `main.model` directly and
    suppressing the lifespan via `raise_server_exceptions=False` on startup.
    """
    with patch.object(main, "model", mock_model):
        # We skip the real lifespan by not using a context manager on TestClient
        # (pass `raise_server_exceptions` to avoid lifespan errors in unit tests)
        yield TestClient(main.app, raise_server_exceptions=True)


@pytest.fixture()
def client_no_model() -> TestClient:
    """TestClient where the model has NOT been loaded (main.model is None)."""
    with patch.object(main, "model", None):
        yield TestClient(main.app, raise_server_exceptions=True)


@pytest.fixture()
def mock_pipeline_result() -> dict:
    """Canonical run_pipeline() return value used across species-traits tests."""
    return {
        "assessment_id": 12345,
        "year_published": 2022,
        "scientific_name": "Panthera leo",
        "common_names": ["Lion"],
        "category": "Vulnerable",
        "url": "https://www.iucnredlist.org/species/15951/115130419",
        "sis_taxon_id": 15951,
        "photo_url": "https://inaturalist.org/photos/lion.jpg",
        "photo_credit": "(c) Example Photographer",
        "lifespan_years": "10–14 years",
        "mass": "120–250 kg",
        "length": "1.4–2.0 m",
        "short_description": (
            "Panthera leo is a large social felid native to sub-Saharan Africa "
            "and classified as Vulnerable by the IUCN."
        ),
        "human_risk_level": "Very High",
        "human_threat_level": "High",
        "fun_fact_1": "Lions are the only truly social wild cats.",
        "fun_fact_2": "A lion's roar can be heard 8 km away.",
        "fun_fact_3": "Female lions do ~90 % of pride hunting.",
    }

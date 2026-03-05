"""
tests/test_species_traits.py
----------------------------
Tests for:
  GET /species/traits?scientific_name=<name>

run_pipeline() is fully mocked here; its internals are tested in
test_pipeline.py.
"""

from unittest.mock import patch

import pytest


class TestSpeciesTraitsRoute:

    # ── Validation errors ──────────────────────────────────────────────────────

    def test_422_missing_scientific_name(self, client):
        resp = client.get("/species/traits")
        assert resp.status_code == 422

    def test_422_name_too_short(self, client):
        # min_length=3 enforced by Query
        resp = client.get("/species/traits", params={"scientific_name": "ab"})
        assert resp.status_code == 422

    # ── Happy path ─────────────────────────────────────────────────────────────

    def test_200_on_valid_name(self, client, mock_pipeline_result):
        with patch("main.run_pipeline", return_value=mock_pipeline_result):
            resp = client.get("/species/traits", params={"scientific_name": "Panthera leo"})
        assert resp.status_code == 200

    def test_response_schema_fields(self, client, mock_pipeline_result):
        with patch("main.run_pipeline", return_value=mock_pipeline_result):
            body = client.get("/species/traits", params={"scientific_name": "Panthera leo"}).json()

        expected_keys = {
            "assessment_id", "year_published", "scientific_name", "common_names",
            "category", "url", "sis_taxon_id", "photo_url", "photo_credit",
            "lifespan_years", "mass", "length", "short_description",
            "human_risk_level", "human_threat_level",
            "fun_fact_1", "fun_fact_2", "fun_fact_3",
        }
        assert expected_keys.issubset(body.keys())

    def test_scientific_name_in_response(self, client, mock_pipeline_result):
        with patch("main.run_pipeline", return_value=mock_pipeline_result):
            body = client.get("/species/traits", params={"scientific_name": "Panthera leo"}).json()
        assert body["scientific_name"] == "Panthera leo"

    def test_common_names_is_list(self, client, mock_pipeline_result):
        with patch("main.run_pipeline", return_value=mock_pipeline_result):
            body = client.get("/species/traits", params={"scientific_name": "Panthera leo"}).json()
        assert isinstance(body["common_names"], list)

    def test_human_risk_level_valid_value(self, client, mock_pipeline_result):
        with patch("main.run_pipeline", return_value=mock_pipeline_result):
            body = client.get("/species/traits", params={"scientific_name": "Panthera leo"}).json()
        assert body["human_risk_level"] in {"Very High", "High", "Caution", "Low"}

    def test_fun_facts_present(self, client, mock_pipeline_result):
        with patch("main.run_pipeline", return_value=mock_pipeline_result):
            body = client.get("/species/traits", params={"scientific_name": "Panthera leo"}).json()
        assert body["fun_fact_1"]
        assert body["fun_fact_2"]
        assert body["fun_fact_3"]

    def test_name_with_spaces(self, client, mock_pipeline_result):
        """Spaces in scientific name must be handled (URL-encoded by requests)."""
        with patch("main.run_pipeline", return_value=mock_pipeline_result) as mock_run:
            client.get("/species/traits", params={"scientific_name": "Caridina typus"})
        mock_run.assert_called_once_with("Caridina typus")

    def test_pipeline_called_with_exact_name(self, client, mock_pipeline_result):
        with patch("main.run_pipeline", return_value=mock_pipeline_result) as mock_run:
            client.get("/species/traits", params={"scientific_name": "Panthera leo"})
        mock_run.assert_called_once_with("Panthera leo")

    # ── Error paths ────────────────────────────────────────────────────────────

    def test_404_when_blob_not_found(self, client):
        with patch("main.run_pipeline", side_effect=FileNotFoundError("No blob found")):
            resp = client.get("/species/traits", params={"scientific_name": "Unknown species"})
        assert resp.status_code == 404
        assert "No blob found" in resp.json()["detail"]

    def test_503_on_missing_azure_config(self, client):
        with patch("main.run_pipeline", side_effect=RuntimeError("Set AZURE_STORAGE_ACCOUNT_URL")):
            resp = client.get("/species/traits", params={"scientific_name": "Panthera leo"})
        assert resp.status_code == 503

    def test_500_on_unexpected_pipeline_error(self, client):
        with patch("main.run_pipeline", side_effect=Exception("unexpected")):
            resp = client.get("/species/traits", params={"scientific_name": "Panthera leo"})
        assert resp.status_code == 500
        assert "Pipeline error" in resp.json()["detail"]

    # ── Extra fields from LLM are surfaced (model_config extra=allow) ─────────

    def test_extra_llm_fields_passed_through(self, client, mock_pipeline_result):
        enriched = {**mock_pipeline_result, "diet": "Carnivore", "habitat": "Savanna"}
        with patch("main.run_pipeline", return_value=enriched):
            body = client.get("/species/traits", params={"scientific_name": "Panthera leo"}).json()
        assert body.get("diet") == "Carnivore"
        assert body.get("habitat") == "Savanna"

    # ── Content-type ──────────────────────────────────────────────────────────

    def test_content_type_json(self, client, mock_pipeline_result):
        with patch("main.run_pipeline", return_value=mock_pipeline_result):
            resp = client.get("/species/traits", params={"scientific_name": "Panthera leo"})
        assert "application/json" in resp.headers["content-type"]

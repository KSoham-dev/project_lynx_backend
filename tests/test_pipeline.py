"""
tests/test_pipeline.py
----------------------
Unit tests for every function in pipeline/species_traits.py.

All network calls (Wikipedia, iNaturalist), Azure Blob, and Groq are mocked.
The in-process cache is cleared before each test to prevent cross-test
contamination.
"""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

import pipeline.species_traits as pt


# ── Helpers to reset global cache ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_cache():
    """Wipe the TTLCache before every test."""
    with pt._cache_lock:
        pt._cache.clear()
    yield
    with pt._cache_lock:
        pt._cache.clear()


# ── IUCN JSON fixture ──────────────────────────────────────────────────────────

@pytest.fixture()
def iucn_json() -> dict:
    return {
        "assessment_id": 99,
        "year_published": 2022,
        "taxon": {
            "scientific_name": "Panthera leo",
            "common_names": [
                {"name": "Lion", "main": True},
                {"name": "African Lion", "main": False},
            ],
        },
        "red_list_category": {"description": {"en": "Vulnerable"}},
        "references": [],
        "url": "https://iucnredlist.org/species/15951",
        "sis_taxon_id": 15951,
    }


@pytest.fixture()
def groq_traits() -> dict:
    return {
        "lifespan_years": "10–14 years",
        "mass": "120–250 kg",
        "length": "1.4–2.0 m",
        "short_description": "A large felid, Vulnerable on IUCN.",
        "human_risk_level": "Very High",
        "human_threat_level": "High",
        "fun_fact_1": "Only social wild cat.",
        "fun_fact_2": "Roar carries 8 km.",
        "fun_fact_3": "Females do 90 % of hunting.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Pure helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalize:
    def test_spaces_removed(self):
        assert pt._normalize("Caridina typus") == "Caridinatypus"

    def test_multiple_spaces(self):
        assert pt._normalize("A b c") == "Abc"

    def test_no_spaces(self):
        assert pt._normalize("Caridinatypus") == "Caridinatypus"

    def test_all_spaces_removed_including_edges(self):
        # _normalize removes ALL spaces; run_pipeline strips before calling it
        assert pt._normalize(" X y ") == "Xy"


class TestSafeGet:
    def test_simple_key(self):
        assert pt._safe_get({"a": "v"}, "a") == "v"

    def test_nested_keys(self):
        assert pt._safe_get({"a": {"b": {"c": 42}}}, "a", "b", "c") == 42

    def test_missing_key_returns_default(self):
        assert pt._safe_get({}, "a", default="X") == "X"

    def test_none_default_when_unspecified(self):
        assert pt._safe_get({}, "missing") is None

    def test_intermediate_none(self):
        assert pt._safe_get({"a": None}, "a", "b") is None

    def test_index_error(self):
        assert pt._safe_get([], 0, default="d") == "d"

    def test_falsy_value_replaced_by_default(self):
        # empty string is falsy → returns default
        assert pt._safe_get({"k": ""}, "k", default="fallback") == "fallback"


# ══════════════════════════════════════════════════════════════════════════════
# _wiki_extract
# ══════════════════════════════════════════════════════════════════════════════

class TestWikiExtract:

    def _mock_session(self, search_titles, extract_text):
        """Build a mock _http_session that returns controlled Wikipedia API responses."""
        session = MagicMock()

        search_response = MagicMock()
        search_response.json.return_value = ["q", search_titles, [], []]

        page_response = MagicMock()
        page_response.json.return_value = {
            "query": {
                "pages": {
                    "12345": {"extract": extract_text}
                }
            }
        }
        session.get.side_effect = [search_response, page_response]
        return session

    def test_returns_extract_on_success(self):
        with patch.object(pt, "_http_session", self._mock_session(["Panthera leo"], "Lion text.")):
            result = pt._wiki_extract("Panthera_leo")
        assert result == "Lion text."

    def test_returns_not_found_when_no_search_results(self):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = ["q", [], [], []]
        session.get.return_value = resp
        with patch.object(pt, "_http_session", session):
            result = pt._wiki_extract("Nonexistent_species")
        assert result == "Not found"

    def test_returns_not_found_on_network_error(self):
        session = MagicMock()
        session.get.side_effect = ConnectionError("timeout")
        with patch.object(pt, "_http_session", session):
            result = pt._wiki_extract("Anything")
        assert result == "Not found"

    def test_returns_not_found_when_extract_is_missing(self):
        session = MagicMock()
        search_resp = MagicMock()
        search_resp.json.return_value = ["q", ["Title"], [], []]
        page_resp = MagicMock()
        page_resp.json.return_value = {"query": {"pages": {"1": {}}}}
        session.get.side_effect = [search_resp, page_resp]
        with patch.object(pt, "_http_session", session):
            result = pt._wiki_extract("Title")
        assert result == "Not found"


# ══════════════════════════════════════════════════════════════════════════════
# _inaturalist_photo
# ══════════════════════════════════════════════════════════════════════════════

class TestINaturalistPhoto:

    def _mock_session(self, results):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"results": results}
        session.get.return_value = resp
        return session

    def test_returns_url_and_credit(self):
        results = [{"default_photo": {"medium_url": "https://img.com/a.jpg", "attribution": "(c) X"}}]
        with patch.object(pt, "_http_session", self._mock_session(results)):
            url, credit = pt._inaturalist_photo("Panthera leo")
        assert url == "https://img.com/a.jpg"
        assert credit == "(c) X"

    def test_returns_none_when_no_results(self):
        with patch.object(pt, "_http_session", self._mock_session([])):
            url, credit = pt._inaturalist_photo("Unknown species")
        assert url is None
        assert credit is None

    def test_returns_none_on_network_error(self):
        session = MagicMock()
        session.get.side_effect = ConnectionError("down")
        with patch.object(pt, "_http_session", session):
            url, credit = pt._inaturalist_photo("X")
        assert url is None
        assert credit is None

    def test_returns_none_when_no_default_photo(self):
        results = [{"default_photo": None}]
        with patch.object(pt, "_http_session", self._mock_session(results)):
            url, credit = pt._inaturalist_photo("X")
        assert url is None
        assert credit is None


# ══════════════════════════════════════════════════════════════════════════════
# _fetch_blob
# ══════════════════════════════════════════════════════════════════════════════

class TestFetchBlob:

    def _make_blob_client(self, blob_names: list[str], json_content: dict):
        """Mock BlobServiceClient that lists blobs and returns json_content."""
        blob_item = MagicMock()
        blob_item.name = blob_names[0] if blob_names else None

        container_client = MagicMock()
        container_client.list_blobs.return_value = iter([blob_item] if blob_names else [])

        downloader = MagicMock()
        downloader.readall.return_value = json.dumps(json_content).encode()
        blob_client = MagicMock()
        blob_client.download_blob.return_value = downloader
        container_client.get_blob_client.return_value = blob_client

        service = MagicMock()
        service.get_container_client.return_value = container_client
        return service

    def test_returns_parsed_json(self, iucn_json):
        service = self._make_blob_client(
            ["Pantheraleo_VU_MAMMALIA_leo_FELIDAE_2022"],
            iucn_json,
        )
        with patch.object(pt, "_build_blob_service_client", return_value=service):
            result = pt._fetch_blob("Pantheraleo")
        assert result["assessment_id"] == iucn_json["assessment_id"]

    def test_raises_file_not_found_when_no_blob(self):
        service = self._make_blob_client([], {})
        with patch.object(pt, "_build_blob_service_client", return_value=service):
            with pytest.raises(FileNotFoundError, match="Pantheraleo"):
                pt._fetch_blob("Pantheraleo")

    def test_passes_prefix_to_list_blobs(self, iucn_json):
        service = self._make_blob_client(
            ["Caridinatypus_LC_MALACOSTRACA_typus_ATYIDAE_2013"],
            iucn_json,
        )
        with patch.object(pt, "_build_blob_service_client", return_value=service):
            pt._fetch_blob("Caridinatypus")
        container_client = service.get_container_client.return_value
        container_client = service.get_container_client.return_value
        container_client.list_blobs.assert_called_once()
        used_prefix = container_client.list_blobs.call_args.kwargs.get(
            "name_starts_with",
            container_client.list_blobs.call_args.args[0] if container_client.list_blobs.call_args.args else "",
        )
        assert used_prefix.endswith("Caridinatypus")

    def test_only_first_blob_is_downloaded(self, iucn_json):
        """Even if multiple blobs match, only the first is downloaded."""
        blob1, blob2 = MagicMock(), MagicMock()
        blob1.name = "Pantheraleo_VU_2020"
        blob2.name = "Pantheraleo_VU_2022"

        container_client = MagicMock()
        container_client.list_blobs.return_value = iter([blob1, blob2])
        downloader = MagicMock()
        downloader.readall.return_value = json.dumps(iucn_json).encode()
        blob_client = MagicMock()
        blob_client.download_blob.return_value = downloader
        container_client.get_blob_client.return_value = blob_client

        service = MagicMock()
        service.get_container_client.return_value = container_client

        with patch.object(pt, "_build_blob_service_client", return_value=service):
            pt._fetch_blob("Pantheraleo")

        container_client.get_blob_client.assert_called_once_with("Pantheraleo_VU_2020")


# ══════════════════════════════════════════════════════════════════════════════
# _call_groq
# ══════════════════════════════════════════════════════════════════════════════

class TestCallGroq:

    def test_returns_parsed_traits(self, groq_traits):
        mock_groq = MagicMock()
        mock_groq.return_value.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(groq_traits)

        with patch.object(pt, "Groq", mock_groq):
            result = pt._call_groq({"scientific_name": "Panthera leo"})

        assert result["lifespan_years"] == groq_traits["lifespan_years"]
        assert result["human_risk_level"] == groq_traits["human_risk_level"]

    def test_uses_json_object_format(self, groq_traits):
        mock_groq = MagicMock()
        mock_groq.return_value.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(groq_traits)

        with patch.object(pt, "Groq", mock_groq):
            pt._call_groq({})

        call_kwargs = mock_groq.return_value.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_propagates_exception_on_bad_json(self):
        mock_groq = MagicMock()
        mock_groq.return_value.chat.completions.create.return_value.choices[
            0
        ].message.content = "not json {{{"

        with patch.object(pt, "Groq", mock_groq):
            with pytest.raises(json.JSONDecodeError):
                pt._call_groq({})


# ══════════════════════════════════════════════════════════════════════════════
# run_pipeline  (end-to-end with everything mocked)
# ══════════════════════════════════════════════════════════════════════════════

class TestRunPipeline:

    @pytest.fixture()
    def patched_pipeline(self, iucn_json, groq_traits):
        """Context that stubs all external I/O for run_pipeline."""
        blob_item = MagicMock()
        blob_item.name = "Pantheraleo_VU_MAMMALIA_leo_FELIDAE_2022"
        container_client = MagicMock()
        container_client.list_blobs.return_value = iter([blob_item])
        downloader = MagicMock()
        downloader.readall.return_value = json.dumps(iucn_json).encode()
        blob_client = MagicMock()
        blob_client.download_blob.return_value = downloader
        container_client.get_blob_client.return_value = blob_client
        service = MagicMock()
        service.get_container_client.return_value = container_client

        mock_groq = MagicMock()
        mock_groq.return_value.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps(groq_traits)

        wiki_resp1 = MagicMock()
        wiki_resp1.json.return_value = ["q", ["Panthera leo"], [], []]
        wiki_resp2 = MagicMock()
        wiki_resp2.json.return_value = {
            "query": {"pages": {"1": {"extract": "Lion is a big cat."}}}
        }
        inat_resp = MagicMock()
        inat_resp.json.return_value = {
            "results": [
                {"default_photo": {"medium_url": "https://img/lion.jpg", "attribution": "(c) X"}}
            ]
        }

        session = MagicMock()
        # thread pool submits wiki and inat concurrently; session.get call order:
        # wiki call 1, wiki call 2, inaturalist call  (order may vary by thread)
        session.get.side_effect = [wiki_resp1, wiki_resp2, inat_resp]

        with (
            patch.object(pt, "_build_blob_service_client", return_value=service),
            patch.object(pt, "Groq", mock_groq),
            patch.object(pt, "_http_session", session),
        ):
            yield

    def test_returns_dict(self, patched_pipeline):
        result = pt.run_pipeline("Panthera leo")
        assert isinstance(result, dict)

    def test_scientific_name_in_result(self, patched_pipeline):
        result = pt.run_pipeline("Panthera leo")
        assert result["scientific_name"] == "Panthera leo"

    def test_traits_merged_into_result(self, patched_pipeline, groq_traits):
        result = pt.run_pipeline("Panthera leo")
        assert result["lifespan_years"] == groq_traits["lifespan_years"]
        assert result["human_risk_level"] == groq_traits["human_risk_level"]

    def test_photo_url_in_result(self, patched_pipeline):
        result = pt.run_pipeline("Panthera leo")
        assert result["photo_url"] == "https://img/lion.jpg"

    def test_wiki_extract_stripped_from_result(self, patched_pipeline):
        """wiki_extract is used for the LLM prompt but must NOT appear in output."""
        result = pt.run_pipeline("Panthera leo")
        assert "wiki_extract" not in result

    def test_result_cached_after_first_call(self, patched_pipeline):
        pt.run_pipeline("Panthera leo")
        # cache should be populated
        with pt._cache_lock:
            assert "panthera leo" in pt._cache

    def test_cache_hit_returns_same_result(self, patched_pipeline):
        first = pt.run_pipeline("Panthera leo")
        # populate with known value then call again
        second = pt.run_pipeline("Panthera leo")
        assert first == second

    def test_cache_hit_skips_blob_call(self, patched_pipeline):
        """Second call with same name must not call _fetch_blob."""
        pt.run_pipeline("Panthera leo")
        with patch.object(pt, "_fetch_blob", side_effect=AssertionError("should not call blob")):
            # Should NOT raise because result comes from cache
            pt.run_pipeline("Panthera leo")

    def test_cache_key_is_lowercased(self, patched_pipeline):
        pt.run_pipeline("Panthera Leo")   # mixed case
        with pt._cache_lock:
            assert "panthera leo" in pt._cache

    def test_groq_failure_returns_partial_result(self, iucn_json):
        """If Groq throws, run_pipeline returns IUCN/photo data without traits."""
        blob_item = MagicMock()
        blob_item.name = "Pantheraleo_VU_2022"
        container_client = MagicMock()
        container_client.list_blobs.return_value = iter([blob_item])
        downloader = MagicMock()
        downloader.readall.return_value = json.dumps(iucn_json).encode()
        blob_client = MagicMock()
        blob_client.download_blob.return_value = downloader
        container_client.get_blob_client.return_value = blob_client
        service = MagicMock()
        service.get_container_client.return_value = container_client

        wiki_resp1 = MagicMock()
        wiki_resp1.json.return_value = ["q", ["Panthera leo"], [], []]
        wiki_resp2 = MagicMock()
        wiki_resp2.json.return_value = {"query": {"pages": {"1": {"extract": "..."}}}}
        inat_resp = MagicMock()
        inat_resp.json.return_value = {"results": []}
        session = MagicMock()
        session.get.side_effect = [wiki_resp1, wiki_resp2, inat_resp]

        mock_groq = MagicMock()
        mock_groq.return_value.chat.completions.create.side_effect = RuntimeError("Groq down")

        with (
            patch.object(pt, "_build_blob_service_client", return_value=service),
            patch.object(pt, "Groq", mock_groq),
            patch.object(pt, "_http_session", session),
        ):
            result = pt.run_pipeline("Panthera leo")

        assert result["scientific_name"] == "Panthera leo"
        assert "lifespan_years" not in result  # traits absent but no crash

    def test_raises_file_not_found_on_missing_blob(self):
        container_client = MagicMock()
        container_client.list_blobs.return_value = iter([])
        service = MagicMock()
        service.get_container_client.return_value = container_client

        with patch.object(pt, "_build_blob_service_client", return_value=service):
            with pytest.raises(FileNotFoundError):
                pt.run_pipeline("Unknown nomen")

    def test_cache_is_thread_safe(self):
        """Concurrent reads from a pre-seeded cache must not corrupt state."""
        seed = {"scientific_name": "Panthera leo", "lifespan_years": "12 years"}
        with pt._cache_lock:
            pt._cache["panthera leo"] = seed

        errors: list[Exception] = []
        results: list[dict] = []

        def call():
            try:
                r = pt.run_pipeline("Panthera leo")
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 8
        assert all(r == seed for r in results)

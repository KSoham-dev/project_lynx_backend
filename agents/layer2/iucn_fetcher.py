"""
agents/layer1/iucn_fetcher.py
==============================
Layer 1 — iucn_raw_data tool implementation.

Fetches raw IUCN assessment data directly from Azure Blob Storage without
running the Groq / Wikipedia / iNaturalist enrichment pipeline.

Used for:
  EXPLORE      — alongside context_text (species traits + raw IUCN)
  ENCYCLOPEDIA — standalone (raw IUCN only)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _extract_iucn_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract and normalise key IUCN assessment fields from the raw blob JSON."""
    taxon: dict = raw.get("taxon") or {}
    common_names = taxon.get("common_names") or []
    main_names = [c["name"] for c in common_names if c.get("main")]

    def _safe(d: Any, *keys: str, default: Any = None) -> Any:
        for k in keys:
            try:
                d = d[k]
            except (KeyError, TypeError, IndexError):
                return default
        return d or default

    return {
        "assessment_id":         raw.get("assessment_id"),
        "year_published":        raw.get("year_published"),
        "scientific_name":       taxon.get("scientific_name"),
        "common_names":          main_names,
        "red_list_category":     _safe(raw, "red_list_category", "description", "en"),
        "red_list_code":         _safe(raw, "red_list_category", "code"),
        "population_trend":      _safe(raw, "population", "trend", "description", "en"),
        "population_size":       _safe(raw, "population", "size"),
        "geographic_range":      _safe(raw, "geographic_range", "description"),
        "habitats": [
            {
                "name":       h.get("description", {}).get("en"),
                "suitability": _safe(h, "suitability", "description", "en"),
                "season":     _safe(h, "season", "description", "en"),
            }
            for h in (raw.get("habitats") or [])[:5]
        ],
        "threats": [
            {
                "title":  t.get("description", {}).get("en"),
                "timing": _safe(t, "timing", "description", "en"),
                "scope":  _safe(t, "scope", "description", "en"),
                "impact": _safe(t, "impact", "description", "en"),
            }
            for t in (raw.get("threats") or [])[:8]
        ],
        "conservation_actions": [
            a.get("description", {}).get("en")
            for a in (raw.get("conservation_actions") or [])[:6]
        ],
        "use_trade": [
            {
                "description": u.get("description", {}).get("en"),
                "purpose":     _safe(u, "purpose", "description", "en"),
            }
            for u in (raw.get("use_trade") or [])[:4]
        ],
        "url":          raw.get("url"),
        "sis_taxon_id": raw.get("sis_taxon_id"),
        "references":   (raw.get("references") or [])[:10],
    }


async def get_iucn_raw(scientific_name: str) -> dict[str, Any]:
    """
    Fetch and extract raw IUCN data for a species from Azure Blob Storage.

    Parameters
    ----------
    scientific_name:
        Scientific name, e.g. "Panthera tigris".

    Returns
    -------
    dict with extracted IUCN fields (no Groq enrichment).

    Raises
    ------
    FileNotFoundError
        When no matching blob is found.
    RuntimeError
        When Azure Blob Storage is not configured.
    """
    from pipeline.species_traits import _fetch_blob, _normalize

    normalized = _normalize(scientific_name.strip())
    logger.info("[iucn_raw_data] Fetching blob for %r (prefix=%r)", scientific_name, normalized)
    raw: dict[str, Any] = await asyncio.to_thread(_fetch_blob, normalized)
    result = _extract_iucn_fields(raw)
    logger.info(
        "[iucn_raw_data] Done: species=%r threats=%d habitats=%d",
        scientific_name, len(result.get("threats") or []), len(result.get("habitats") or []),
    )
    return result

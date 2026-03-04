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
    """Return the raw IUCN blob data exactly as stored — no field filtering or transformation."""
    return raw


# ── Read-only accessors for the raw blob nested structure ─────────────────────
# These helpers let callers read fields without duplicating nested-key logic.
# They never modify or copy the data.

def iucn_scientific_name(raw: dict[str, Any]) -> str:
    return ((raw.get("taxon") or {}).get("scientific_name") or "").strip()

def iucn_common_names(raw: dict[str, Any]) -> list[str]:
    entries = (raw.get("taxon") or {}).get("common_names") or []
    return [c["name"] for c in entries if c.get("main")]

def iucn_taxon_field(raw: dict[str, Any], key: str) -> Any:
    """Read any field directly from the taxon sub-object (genus, family, kingdom, …)."""
    return (raw.get("taxon") or {}).get(key)

def iucn_red_list_category(raw: dict[str, Any]) -> str:
    rl = raw.get("red_list_category") or {}
    return ((rl.get("description") or {}).get("en") or "").strip()

def iucn_red_list_code(raw: dict[str, Any]) -> str:
    return ((raw.get("red_list_category") or {}).get("code") or "").strip().upper()

def iucn_population_trend(raw: dict[str, Any]) -> str:
    trend = ((raw.get("population") or {}).get("trend") or {})
    return ((trend.get("description") or {}).get("en") or "").strip()

def iucn_threat_titles(raw: dict[str, Any], limit: int = 4) -> list[str]:
    """Return the English description of the top threats (for prompt context)."""
    return [
        ((t.get("description") or {}).get("en") or "")
        for t in (raw.get("threats") or [])[:limit]
        if (t.get("description") or {}).get("en")
    ]

def iucn_habitat_names(raw: dict[str, Any], limit: int = 4) -> list[str]:
    """Return the English description of the top habitats (for prompt context)."""
    return [
        ((h.get("description") or {}).get("en") or "")
        for h in (raw.get("habitats") or [])[:limit]
        if (h.get("description") or {}).get("en")
    ]


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

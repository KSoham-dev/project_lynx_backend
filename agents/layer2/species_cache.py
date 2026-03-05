"""
agents/layer2/species_cache.py
================================
Cosmos DB read/write helpers for the ``species_context_cache`` container.

Two cache buckets in the same container, distinguished by a prefix on the id:
    traits:<normalised_name>   — full run_pipeline() output (reporter)
    iucn:<normalised_name>     — get_iucn_raw() output   (species_info / explorer)

TTL: 24 hours (86 400 s) — set at item level; the container must have
     DefaultTimeToLive = -1 (item-level TTL enabled).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from agents.state.containers import StateContainers
from agents.state.cosmos_client import get_state_container

logger = logging.getLogger(__name__)

_TTL_24H = 86_400


def _cache_id(prefix: str, scientific_name: str) -> str:
    """Stable document id: '<prefix>:<genus_species_lowercase_underscored>'."""
    key = scientific_name.strip().lower().replace(" ", "_")
    return f"{prefix}:{key}"


# ── Read ──────────────────────────────────────────────────────────────────────

async def read_species_cache(prefix: str, scientific_name: str) -> Optional[dict[str, Any]]:
    """Return cached payload or None on miss / error."""
    doc_id = _cache_id(prefix, scientific_name)
    try:
        container = await get_state_container(StateContainers.SPECIES_CONTEXT_CACHE)
        doc = await container.read_item(item=doc_id, partition_key=doc_id)
        logger.info("[species_cache] HIT  prefix=%s name=%r", prefix, scientific_name)
        return doc.get("payload")
    except Exception:
        logger.debug("[species_cache] MISS prefix=%s name=%r", prefix, scientific_name)
        return None


# ── Write ─────────────────────────────────────────────────────────────────────

async def write_species_cache(
    prefix: str,
    scientific_name: str,
    payload: dict[str, Any],
) -> None:
    """Upsert payload into species_context_cache. Non-fatal on error."""
    doc_id = _cache_id(prefix, scientific_name)
    doc = {
        "id":             doc_id,
        "scientific_name": scientific_name,
        "prefix":         prefix,
        "payload":        payload,
        "ttl":            _TTL_24H,
    }
    try:
        container = await get_state_container(StateContainers.SPECIES_CONTEXT_CACHE)
        await container.upsert_item(body=doc)
        logger.info("[species_cache] WRITE OK prefix=%s name=%r", prefix, scientific_name)
    except Exception as exc:
        logger.warning(
            "[species_cache] WRITE FAILED prefix=%s name=%r: %s",
            prefix, scientific_name, exc, exc_info=True,
        )

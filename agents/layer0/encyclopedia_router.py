"""
agents/layer0/encyclopedia_router.py
=====================================
Encyclopedia-specific routes.

Routes
------
GET /encyclopedia/random
    Return a fully-enriched encyclopedia entry for a randomly selected species
    from the IUCN blob storage container.

    The route is intentionally lightweight — no LLM calls, no session
    management, no Layer 1 tool selection.  It reads directly from blob
    storage and enriches with an iNaturalist photo, mirroring the output
    shape of the regular encyclopedia flow in /agent/chat.

Implementation notes
--------------------
* Blob names are cached in memory for ``_BLOB_CACHE_TTL`` seconds (1 hour)
  so we don't list the entire container on every request.
* After picking a random blob name the raw JSON is fetched directly via its
  full blob name (no prefix search needed), parsed, and passed straight to
  EncyclopediaAgent which attaches the iNaturalist photo — exactly as the
  normal chat flow does.
* On any transient Azure error the cache is left intact so the next request
  can retry without a re-list.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from agents.layer2.inaturalist import get_inaturalist_photo
from agents.layer2.iucn_fetcher import (
    iucn_red_list_category,
    iucn_scientific_name,
    is_iucn_not_found,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/encyclopedia", tags=["encyclopedia"])

# ── In-memory blob-name cache ─────────────────────────────────────────────────
_BLOB_CACHE_TTL: int = 3600  # 1 hour

_cached_blob_names: list[str] = []
_cache_loaded_at: float = 0.0
_cache_lock = asyncio.Lock()


async def _get_blob_names() -> list[str]:
    """
    Return the cached list of all IUCN blob names, refreshing when stale.

    The list is loaded lazily on the first call and then refreshed at most
    once per ``_BLOB_CACHE_TTL`` seconds.  The lock prevents a thundering-herd
    on startup / after cache expiry.
    """
    global _cached_blob_names, _cache_loaded_at

    now = time.monotonic()
    if _cached_blob_names and (now - _cache_loaded_at) < _BLOB_CACHE_TTL:
        return _cached_blob_names

    async with _cache_lock:
        # Re-check under the lock — another coroutine may have refreshed between
        # the check above and acquiring the lock.
        now = time.monotonic()
        if _cached_blob_names and (now - _cache_loaded_at) < _BLOB_CACHE_TTL:
            return _cached_blob_names

        logger.info("[encyclopedia/random] Refreshing blob name cache …")
        names = await asyncio.to_thread(_list_all_blob_names)
        if not names:
            raise RuntimeError("No blobs found in the IUCN species container.")
        _cached_blob_names = names
        _cache_loaded_at = time.monotonic()
        logger.info("[encyclopedia/random] Blob name cache loaded: %d entries", len(names))
        return _cached_blob_names


def _list_all_blob_names() -> list[str]:
    """
    Synchronous: enumerate all blob names in the IUCN container (runs in a thread).

    Respects the same AZURE_STORAGE_CONTAINER_NAME and AZURE_STORAGE_BLOB_FOLDER
    env vars as ``_fetch_blob`` in pipeline/species_traits.py so the folder
    structure is handled transparently.
    """
    from pipeline.species_traits import _build_blob_service_client

    container = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "iucn-species")
    folder    = os.getenv("AZURE_STORAGE_BLOB_FOLDER", "").strip("/")
    prefix    = f"{folder}/" if folder else ""

    service          = _build_blob_service_client()
    container_client = service.get_container_client(container)

    names: list[str] = [blob.name for blob in container_client.list_blobs(name_starts_with=prefix)]
    return names


def _fetch_blob_by_name(blob_name: str) -> dict[str, Any]:
    """
    Synchronous: download a specific blob by its exact name and return parsed JSON.

    Unlike ``_fetch_blob`` in species_traits.py this does *not* do a prefix search
    — it fetches the exact blob we already know about from the name cache.
    """
    from pipeline.species_traits import _build_blob_service_client

    container = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "iucn-species")
    service   = _build_blob_service_client()
    content   = (
        service
        .get_container_client(container)
        .get_blob_client(blob_name)
        .download_blob()
        .readall()
    )
    return json.loads(content)


# ── Route ─────────────────────────────────────────────────────────────────────

@router.get(
    "/random",
    summary="Surprise Me — random species from encyclopedia",
    response_description=(
        "Fully-enriched IUCN data for a randomly selected species, "
        "identical in shape to the encyclopedia entry returned by /agent/chat."
    ),
)
async def get_random_species() -> dict[str, Any]:
    """
    Pick a random species from the IUCN blob storage container and return its
    full encyclopedia entry (IUCN data + iNaturalist photo).

    The response shape matches the ``data`` field returned by
    ``POST /agent/chat`` with ``query_type=encyclopedia``.
    """
    # ── 1. Get (or refresh) the blob name list ────────────────────────────────
    try:
        blob_names = await _get_blob_names()
    except RuntimeError as exc:
        logger.error("[encyclopedia/random] Blob list unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))

    # ── 2. Pick a random blob ─────────────────────────────────────────────────
    blob_name = random.choice(blob_names)
    logger.info("[encyclopedia/random] Selected blob: %s", blob_name)

    # ── 3. Fetch the raw IUCN JSON ────────────────────────────────────────────
    try:
        iucn_data: dict[str, Any] = await asyncio.to_thread(_fetch_blob_by_name, blob_name)
    except Exception as exc:
        logger.error(
            "[encyclopedia/random] Failed to fetch blob %r: %s", blob_name, exc, exc_info=True
        )
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch species data from storage: {exc}",
        )

    # ── 4. Guard: blob should not be a "not-in-database" sentinel ────────────
    if is_iucn_not_found(iucn_data):
        # Should never happen for a blob that physically exists, but be safe.
        logger.warning("[encyclopedia/random] Blob returned not-in-database dict; retrying once")
        blob_name = random.choice(blob_names)
        try:
            iucn_data = await asyncio.to_thread(_fetch_blob_by_name, blob_name)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    # ── 5. Enrich with iNaturalist photo ─────────────────────────────────────
    scientific_name: str = iucn_scientific_name(iucn_data) or ""
    photo_url, photo_credit = await get_inaturalist_photo(scientific_name)

    result: dict[str, Any] = {
        **iucn_data,
        "photo_url":    photo_url    or "Not available",
        "photo_credit": photo_credit or "Not available",
    }

    logger.info(
        "[encyclopedia/random] Done: species=%r category=%r photo=%s",
        scientific_name,
        iucn_red_list_category(iucn_data),
        "yes" if photo_url else "no",
    )
    return result

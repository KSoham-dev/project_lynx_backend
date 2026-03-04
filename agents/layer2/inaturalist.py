"""
agents/layer2/inaturalist.py
==============================
Shared iNaturalist photo utility — reusable by any Layer 2 agent.

Provides a single async function that queries the iNaturalist Taxa API
and returns the default photo URL + attribution for a species.

Usage
-----
    from agents.layer2.inaturalist import get_inaturalist_photo
    photo_url, photo_credit = await get_inaturalist_photo("Panthera tigris")
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import requests as _requests

logger = logging.getLogger(__name__)

# Shared HTTP session — connection pooled, thread-safe
_session = _requests.Session()
_session.headers.update({"User-Agent": "Prahari/1.0"})


def _fetch_photo_sync(scientific_name: str) -> tuple[Optional[str], Optional[str]]:
    """
    Synchronous iNaturalist fetch — runs inside a thread via asyncio.to_thread.

    Returns
    -------
    (medium_url, attribution) or (None, None) on any failure / no results.
    """
    try:
        resp = _session.get(
            "https://api.inaturalist.org/v1/taxa",
            params={"q": scientific_name},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            logger.debug("[inaturalist] No results for %r", scientific_name)
            return None, None

        photo = results[0].get("default_photo") or {}
        url   = photo.get("medium_url")
        credit = photo.get("attribution")
        logger.info("[inaturalist] Photo found for %r: %s", scientific_name, url)
        return url, credit
    except Exception as exc:
        logger.warning("[inaturalist] Fetch failed for %r: %s", scientific_name, exc)
        return None, None


async def get_inaturalist_photo(
    scientific_name: str,
) -> tuple[Optional[str], Optional[str]]:
    """
    Async wrapper around the synchronous iNaturalist Taxa API query.

    Returns
    -------
    (medium_url, attribution)
        medium_url   – publicly accessible JPEG URL, or None
        attribution  – credit string, or None
    """
    return await asyncio.to_thread(_fetch_photo_sync, scientific_name)

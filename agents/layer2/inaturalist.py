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

        # Prefer a CC-licensed photo over an all-rights-reserved one.
        # Walk every result's default_photo and pick the first that has
        # "cc" in its license_code or attribution (case-insensitive).
        # Fall back to the very first photo if nothing CC is found.
        def _is_cc(photo: dict) -> bool:
            license_code = (photo.get("license_code") or "").lower()
            attribution  = (photo.get("attribution")  or "").lower()
            return "cc" in license_code or "cc" in attribution

        def _is_all_rights_reserved(photo: dict) -> bool:
            return "all rights reserved" in (photo.get("attribution") or "").lower()

        chosen_photo: dict | None = None

        for result in results:
            p = result.get("default_photo") or {}
            if not p.get("medium_url"):
                continue
            if _is_cc(p) and not _is_all_rights_reserved(p):
                chosen_photo = p
                break

        if chosen_photo is None:
            # No CC-licensed photo found anywhere in the results
            logger.info(
                "[inaturalist] No CC-licensed photo found for %r — returning no photo",
                scientific_name,
            )
            return None, None

        url    = chosen_photo.get("medium_url")
        credit = chosen_photo.get("attribution")
        default_photo = results[0].get("default_photo") or {}
        if chosen_photo is not default_photo and _is_all_rights_reserved(default_photo):
            logger.info(
                "[inaturalist] Default photo for %r is all-rights-reserved; "
                "using CC-licensed fallback: %s",
                scientific_name, url,
            )
        else:
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

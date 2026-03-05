"""
agents/layer1/location.py
==========================
Layer 1 — location_context tool implementation.

Reverse-geocodes GPS coordinates using Nominatim (OpenStreetMap) and
caches results in Cosmos DB (prahari-state/location_cache, TTL 30 days).

Uses pygeodesy for geohash encoding (pure Python, no C extension).
Geohash precision 5 ≈ 5 km² cell.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
from pygeodesy.geohash import encode as _geohash_encode  # type: ignore[import]

from agents.state.containers import StateContainers
from agents.state.cosmos_client import get_state_container

logger = logging.getLogger(__name__)

_TTL_30D = 2_592_000           # 30 days in seconds
_GEOHASH_PRECISION = 5         # ≈ 5 km² per cell
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
_USER_AGENT = "Prahari/1.0 (wildlife-field-assistant)"

_PROTECTED_TAGS = {
    "nature_reserve", "national_park", "protected_area",
    "wildlife_sanctuary", "forest", "tiger_reserve",
}


def _geohash(lat: float, lon: float) -> str:
    return _geohash_encode(lat, lon, precision=_GEOHASH_PRECISION)


def _parse_nominatim(data: dict) -> dict[str, Any]:
    """Extract structured fields from a Nominatim reverse-geocode response."""
    addr: dict = data.get("address") or {}

    district: Optional[str] = (
        addr.get("county")
        or addr.get("district")
        or addr.get("city_district")
        or addr.get("municipality")
    )
    state:   Optional[str] = addr.get("state")
    country: Optional[str] = addr.get("country")

    # Protected area from extra_tags or address tags
    extra_tags: dict = data.get("extratags") or {}
    protected_area: Optional[str] = extra_tags.get("name") or None

    if not protected_area:
        for tag in _PROTECTED_TAGS:
            if tag in addr:
                protected_area = addr[tag]
                break

    # Fallback: check display_name for known protected area phrases
    display: str = data.get("display_name") or ""
    if not protected_area:
        for phrase in ["National Park", "Wildlife Sanctuary", "Tiger Reserve", "Nature Reserve"]:
            if phrase in display:
                for part in display.split(","):
                    if phrase in part:
                        protected_area = part.strip()
                        break
                break

    parts = [p for p in [protected_area, district, state] if p]
    formatted = ", ".join(parts) if parts else display[:80]

    return {
        "district":       district,
        "state":          state,
        "country":        country,
        "protected_area": protected_area,
        "formatted":      formatted,
        "display_name":   display,
    }


async def get_location_context(latitude: float, longitude: float) -> dict[str, Any]:
    """
    Reverse-geocode GPS coordinates and cache the result.

    Parameters
    ----------
    latitude, longitude: WGS-84 decimal degrees.

    Returns
    -------
    dict with: district, state, country, protected_area, geohash, formatted.
    """
    geo = _geohash(latitude, longitude)
    logger.info("[location_context] START: lat=%s lon=%s geohash=%s", latitude, longitude, geo)

    # ── Get container ────────────────────────────────────────────────────────
    container: Optional[Any] = None
    try:
        container = await get_state_container(StateContainers.LOCATION_CACHE)
    except Exception as exc:
        logger.warning("[location_context] Could not get location_cache container: %s", exc)

    # ── Cache read ───────────────────────────────────────────────────────────
    if container is not None:
        try:
            doc = await container.read_item(item=geo, partition_key=geo)
            logger.info("[location_context] Cache HIT: geohash=%s", geo)
            return doc.get("result", {})
        except Exception:
            logger.debug("[location_context] Cache MISS: geohash=%s", geo)

    # ── Nominatim reverse geocode ────────────────────────────────────────────
    logger.info("[location_context] Calling Nominatim for lat=%s lon=%s", latitude, longitude)
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            _NOMINATIM_URL,
            params={"lat": latitude, "lon": longitude, "format": "json", "extratags": 1},
            headers={"User-Agent": _USER_AGENT},
        )
        resp.raise_for_status()
        nominatim_data = resp.json()

    parsed = _parse_nominatim(nominatim_data)
    result: dict[str, Any] = {**parsed, "geohash": geo, "latitude": latitude, "longitude": longitude}
    logger.info(
        "[location_context] DONE: district=%r state=%r protected_area=%r",
        result.get("district"), result.get("state"), result.get("protected_area"),
    )

    # ── Cache write ──────────────────────────────────────────────────────────
    if container is not None:
        try:
            cache_doc = {"id": geo, "geohash": geo, "result": result, "ttl": _TTL_30D}
            await container.upsert_item(body=cache_doc)
            logger.info("[location_context] Cache WRITE OK: geohash=%s", geo)
        except Exception as exc:
            logger.warning("[location_context] Cache write failed for %s: %s", geo, exc, exc_info=True)

    return result

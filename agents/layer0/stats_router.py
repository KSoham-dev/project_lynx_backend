"""
agents/layer0/stats_router.py
==============================
GET /stats/reports — Dashboard statistics derived from the reports container.

Designed for the frontend homepage. Returns a single response with all the
figures needed to populate summary cards, charts, and a recent-activity feed
without requiring multiple round-trips.

All aggregation is done in Python after a single cross-partition scan of the
reports container so the Cosmos index stays simple.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from agents.state.containers import DataContainers
from agents.state.cosmos_client import get_data_container

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stats", tags=["Stats"])


# ── Response models ───────────────────────────────────────────────────────────

class UniqueSpecies(BaseModel):
    scientific_name: str
    common_name: Optional[str] = None
    sighting_count: int
    red_list_category: Optional[str] = None


class SpeciesSighting(BaseModel):
    scientific_name: str
    common_name: Optional[str] = None
    count: int
    red_list_category: Optional[str] = None


class LocationStat(BaseModel):
    state: Optional[str] = None
    district: Optional[str] = None
    count: int


class ProtectedAreaStat(BaseModel):
    name: str
    count: int


class RecentSighting(BaseModel):
    report_id: str
    scientific_name: Optional[str] = None
    common_name: Optional[str] = None
    red_list_category: Optional[str] = None
    risk_level: Optional[str] = None
    severity: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_formatted: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    created_at: Optional[str] = None
    image_url: Optional[str] = None
    sighting_count: int = 1    # total sightings of this species in the radius


class ReportStatsResponse(BaseModel):
    # ── Totals ────────────────────────────────────────────────────────────────
    total_reports: int
    total_wildlife_sightings: int           # alias exposed to frontend
    total_species_in_database: int          # full model catalogue size
    reports_last_7_days: int
    reports_last_30_days: int

    # ── Species breakdown ─────────────────────────────────────────────────────
    unique_species: list[UniqueSpecies]             # all distinct species in radius, sorted by sighting count
    unique_species_count: int

    # ── Location breakdowns ───────────────────────────────────────────────────
    sightings_by_state: list[LocationStat]          # all states
    sightings_by_district: list[LocationStat]       # top-15 districts
    top_protected_areas: list[ProtectedAreaStat]    # named protected areas only

    # ── Classification distributions ─────────────────────────────────────────
    risk_level_distribution: dict[str, int]         # LOW/MEDIUM/HIGH/CRITICAL
    red_list_distribution: dict[str, int]           # CR/EN/VU/NT/LC/DD/…
    severity_distribution: dict[str, int]           # LOW/MEDIUM/HIGH/CRITICAL
    status_distribution: dict[str, int]             # submitted/draft/…

    # ── Recent activity feed ──────────────────────────────────────────────────
    recent_sightings: list[RecentSighting]          # 10 most recent reports

    # ── Global top sighted ───────────────────────────────────────────────────
    top_sighted_species: Optional[UniqueSpecies] = None   # #1 species across entire database


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _parse_dt(val: Any) -> Optional[datetime]:
    if not val:
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _location(doc: dict) -> dict:
    return doc.get("location") or {}


_EARTH_RADIUS_KM = 6_371.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two GPS coordinates."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return _EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Route ─────────────────────────────────────────────────────────────────────

@router.get("/reports", response_model=ReportStatsResponse)
async def get_report_stats(
    latitude:  float = Query(..., description="Centre latitude for radius filter (decimal degrees)."),
    longitude: float = Query(..., description="Centre longitude for radius filter (decimal degrees)."),
    radius_km: float = Query(50.0, ge=1.0, le=2000.0, description="Radius in kilometres. Only sightings within this circle are included."),
) -> ReportStatsResponse:
    """
    Aggregate statistics for wildlife sighting reports within a given radius.

    The frontend must supply the user's current GPS location and a radius.
    Only reports whose stored ``location.latitude`` / ``location.longitude``
    fall within ``radius_km`` of the given centre are included in all counts,
    distributions, and the recent-activity feed.
    """
    try:
        container = await get_data_container(DataContainers.REPORTS)
        query = "SELECT * FROM c"
        raw_docs: list[dict] = []
        async for page in container.query_items(query=query):
            raw_docs.append(page)
    except Exception as exc:
        logger.error("Stats: failed to query reports container: %s", exc)
        raise HTTPException(status_code=503, detail="Could not fetch report data.") from exc

    # ── Radius filter ─────────────────────────────────────────────────────────
    filtered_docs: list[dict] = []
    for doc in raw_docs:
        loc = _location(doc)
        doc_lat = loc.get("latitude")
        doc_lon = loc.get("longitude")
        if doc_lat is None or doc_lon is None:
            continue  # exclude sightings with no GPS data
        if _haversine_km(latitude, longitude, doc_lat, doc_lon) <= radius_km:
            filtered_docs.append(doc)

    logger.info(
        "Stats: %d/%d docs within %.1f km of (%.4f, %.4f)",
        len(filtered_docs), len(raw_docs), radius_km, latitude, longitude,
    )
    all_docs = raw_docs          # full unfiltered set — used for global top species
    raw_docs = filtered_docs

    now = _utcnow()
    cutoff_7d  = now - timedelta(days=7)
    cutoff_30d = now - timedelta(days=30)

    total = len(all_docs)        # total reports across entire database (not radius-filtered)
    count_7d  = 0
    count_30d = 0

    # Aggregation buckets
    species_counter:   Counter[str] = Counter()
    species_meta:      dict[str, dict] = {}          # scientific_name → {common_name, red_list}
    state_counter:     Counter[str] = Counter()
    district_counter:  Counter[str] = Counter()
    protected_counter: Counter[str] = Counter()
    risk_counter:      Counter[str] = Counter()
    red_list_counter:  Counter[str] = Counter()
    severity_counter:  Counter[str] = Counter()
    status_counter:    Counter[str] = Counter()

    dated_docs: list[tuple[datetime, dict]] = []

    for doc in raw_docs:
        created = _parse_dt(doc.get("created_at"))
        if created:
            if created >= cutoff_7d:
                count_7d  += 1
            if created >= cutoff_30d:
                count_30d += 1
            dated_docs.append((created, doc))

        # Species — normalise to "Genus species" (title-case genus, lower epithet)
        # so "panthera leo", "Panthera Leo", and "Panthera leo" all collapse to
        # the same key and appear as a single species on the frontend.
        raw_sname = (doc.get("scientific_name") or "").strip()
        if raw_sname:
            parts = raw_sname.split()
            sname = (parts[0].capitalize() + " " + " ".join(p.lower() for p in parts[1:])).strip() if len(parts) >= 2 else raw_sname.capitalize()
            species_counter[sname] += 1
            if sname not in species_meta:
                species_meta[sname] = {
                    "common_name":       doc.get("common_name"),
                    "red_list_category": doc.get("red_list_category"),
                }

        # Location
        loc = _location(doc)
        state = (loc.get("state") or "").strip()
        district = (loc.get("district") or "").strip()
        protected = (loc.get("protected_area") or "").strip()
        if state:
            state_counter[state] += 1
        if district:
            district_counter[district] += 1
        if protected:
            protected_counter[protected] += 1

        # Risk / classification
        risk = (doc.get("risk_level") or "").strip()
        red  = (doc.get("red_list_category") or "").strip()
        sev  = (doc.get("severity") or "").strip()
        stat = (doc.get("status") or "").strip()
        if risk:
            risk_counter[risk] += 1
        if red:
            red_list_counter[red] += 1
        if sev:
            severity_counter[sev] += 1
        if stat:
            status_counter[stat] += 1

    # ── Unique species in radius (all, sorted by sighting count desc) ──────────
    unique_species = [
        UniqueSpecies(
            scientific_name=name,
            common_name=species_meta[name]["common_name"],
            sighting_count=cnt,
            red_list_category=species_meta[name]["red_list_category"],
        )
        for name, cnt in species_counter.most_common()  # no limit — all unique species
    ]

    # ── Location stats ────────────────────────────────────────────────────────
    sightings_by_state = [
        LocationStat(state=s, count=c)
        for s, c in sorted(state_counter.items(), key=lambda x: -x[1])
    ]
    sightings_by_district = [
        LocationStat(district=d, count=c)
        for d, c in district_counter.most_common(15)
    ]
    top_protected_areas = [
        ProtectedAreaStat(name=p, count=c)
        for p, c in protected_counter.most_common(10)
    ]

    # ── Recent activity — one card per unique species, most recent sighting ──
    # dated_docs is already sorted newest-first; iterating in order means the
    # first time we see a species key it is always the most recent occurrence.
    dated_docs.sort(key=lambda x: x[0], reverse=True)
    recent_sightings: list[RecentSighting] = []
    seen_species: set[str] = set()
    for _, doc in dated_docs:
        raw = (doc.get("scientific_name") or "").strip()
        parts = raw.split()
        key = (parts[0].capitalize() + " " + " ".join(p.lower() for p in parts[1:])).strip() if len(parts) >= 2 else raw.capitalize()
        if key in seen_species:
            continue  # already have a card for this species
        seen_species.add(key)
        loc = _location(doc)
        recent_sightings.append(
            RecentSighting(
                report_id=doc.get("id", ""),
                scientific_name=key or None,
                common_name=doc.get("common_name"),
                red_list_category=doc.get("red_list_category"),
                risk_level=doc.get("risk_level"),
                severity=doc.get("severity"),
                latitude=loc.get("latitude"),
                longitude=loc.get("longitude"),
                location_formatted=loc.get("formatted"),
                state=loc.get("state"),
                district=loc.get("district"),
                created_at=doc.get("created_at"),
                image_url=doc.get("image_url"),
                sighting_count=species_counter.get(key, 1),
            )
        )

    # ── Global top sighted species (whole database, ignores radius) ───────────
    global_species_counter: Counter[str] = Counter()
    global_species_meta: dict[str, dict] = {}
    for doc in all_docs:
        raw_sname = (doc.get("scientific_name") or "").strip()
        if raw_sname:
            parts = raw_sname.split()
            sname = (parts[0].capitalize() + " " + " ".join(p.lower() for p in parts[1:])).strip() if len(parts) >= 2 else raw_sname.capitalize()
            global_species_counter[sname] += 1
            if sname not in global_species_meta:
                global_species_meta[sname] = {
                    "common_name":       doc.get("common_name"),
                    "red_list_category": doc.get("red_list_category"),
                }

    top_sighted_species: Optional[UniqueSpecies] = None
    if global_species_counter:
        top_name, top_count = global_species_counter.most_common(1)[0]
        top_sighted_species = UniqueSpecies(
            scientific_name=top_name,
            common_name=global_species_meta[top_name]["common_name"],
            sighting_count=top_count,
            red_list_category=global_species_meta[top_name]["red_list_category"],
        )

    return ReportStatsResponse(
        total_reports=total,
        total_wildlife_sightings=total,
        total_species_in_database=4677,
        reports_last_7_days=count_7d,
        reports_last_30_days=count_30d,
        unique_species=unique_species,
        unique_species_count=len(species_counter),
        sightings_by_state=sightings_by_state,
        sightings_by_district=sightings_by_district,
        top_protected_areas=top_protected_areas,
        risk_level_distribution=dict(risk_counter),
        red_list_distribution=dict(red_list_counter),
        severity_distribution=dict(severity_counter),
        status_distribution=dict(status_counter),
        recent_sightings=recent_sightings,
        top_sighted_species=top_sighted_species,
    )

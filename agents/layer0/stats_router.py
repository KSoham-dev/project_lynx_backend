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
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.state.containers import DataContainers
from agents.state.cosmos_client import get_data_container

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stats", tags=["Stats"])


# ── Response models ───────────────────────────────────────────────────────────

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


class ReportStatsResponse(BaseModel):
    # ── Totals ────────────────────────────────────────────────────────────────
    total_reports: int
    total_wildlife_sightings: int           # alias exposed to frontend
    total_species_in_database: int          # full model catalogue size
    reports_last_7_days: int
    reports_last_30_days: int

    # ── Species breakdown ─────────────────────────────────────────────────────
    top_species: list[SpeciesSighting]              # top-10 most sighted species
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


# ── Route ─────────────────────────────────────────────────────────────────────

@router.get("/reports", response_model=ReportStatsResponse)
async def get_report_stats() -> ReportStatsResponse:
    """
    Aggregate statistics across all wildlife sighting reports.

    Returns totals, breakdowns by species / location / risk, and a
    recent-activity feed — everything the frontend homepage needs in one call.
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

    now = _utcnow()
    cutoff_7d  = now - timedelta(days=7)
    cutoff_30d = now - timedelta(days=30)

    total = len(raw_docs)
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

        # Species
        sname = (doc.get("scientific_name") or "").strip()
        if sname:
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

    # ── Top species ───────────────────────────────────────────────────────────
    top_species = [
        SpeciesSighting(
            scientific_name=name,
            common_name=species_meta[name]["common_name"],
            count=cnt,
            red_list_category=species_meta[name]["red_list_category"],
        )
        for name, cnt in species_counter.most_common(10)
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

    # ── Recent activity (10 newest) ───────────────────────────────────────────
    dated_docs.sort(key=lambda x: x[0], reverse=True)
    recent_sightings = []
    for _, doc in dated_docs[:10]:
        loc = _location(doc)
        recent_sightings.append(
            RecentSighting(
                report_id=doc.get("id", ""),
                scientific_name=doc.get("scientific_name"),
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
            )
        )

    return ReportStatsResponse(
        total_reports=total,
        total_wildlife_sightings=total,
        total_species_in_database=4677,
        reports_last_7_days=count_7d,
        reports_last_30_days=count_30d,
        top_species=top_species,
        unique_species_count=len(species_counter),
        sightings_by_state=sightings_by_state,
        sightings_by_district=sightings_by_district,
        top_protected_areas=top_protected_areas,
        risk_level_distribution=dict(risk_counter),
        red_list_distribution=dict(red_list_counter),
        severity_distribution=dict(severity_counter),
        status_distribution=dict(status_counter),
        recent_sightings=recent_sightings,
    )

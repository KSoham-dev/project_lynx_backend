"""
agents/layer2/reporter.py
==========================
Layer 2 — Reporter Agent (REPORT query type)

Called by the router after the orchestrator. Uses tool_results to get the
identified species name, then calls run_pipeline() directly to fetch the
full SpeciesTraitsResponse, and saves a ReportDocument to Cosmos DB.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Optional

from agents.enums import IncidentSeverity, QueryType, ReportStatus
from agents.layer0.receiver import AgentRequest
from agents.layer2.species_cache import read_species_cache, write_species_cache
from agents.state.data_store import upsert_report

logger = logging.getLogger(__name__)


class ReporterAgent:
    async def run(
        self,
        request: AgentRequest,
        tool_results: dict[str, Any],
    ) -> dict[str, Any]:
        """
        1. Extract best scientific_name from tool_results.
        2. Call run_pipeline(scientific_name) → SpeciesTraitsResponse dict.
        3. Save ReportDocument to Cosmos DB.
        4. Return SpeciesTraitsResponse dict for frontend.
        """
        text_data:  dict = tool_results.get("analyse_text") or {}
        image_data: dict = tool_results.get("analyse_image") or {}
        loc_data:   dict = tool_results.get("location_context") or {}

        scientific_name: Optional[str] = (
            image_data.get("scientific_name")
            or (text_data.get("scientific_name") if isinstance(text_data, dict) else None)
        )

        # ── Fetch species traits (IUCN blob + Groq enrichment) ──────────────
        traits_data: dict[str, Any] = {}
        if scientific_name and scientific_name.lower() not in ("unknown", "unidentified"):
            # ── Check Cosmos cache first ──────────────────────────────────────
            cached = await read_species_cache("traits", scientific_name)
            if cached:
                traits_data = cached
                logger.info("[reporter] Using cached traits for %r", scientific_name)
            else:
                try:
                    from pipeline.species_traits import run_pipeline
                    logger.info("[reporter] Running species_traits pipeline for %r", scientific_name)
                    traits_data = await asyncio.to_thread(run_pipeline, scientific_name)
                    logger.info("[reporter] Pipeline done: %d fields", len(traits_data))
                    await write_species_cache("traits", scientific_name, traits_data)
                except Exception as exc:
                    logger.error("[reporter] run_pipeline failed for %r: %s", scientific_name, exc)

        # ── Severity mapping ─────────────────────────────────────────────────
        text_severity = (text_data.get("severity") or "low") if isinstance(text_data, dict) else "low"
        incident_severity = _map_severity(text_severity)

        # ── Save report ──────────────────────────────────────────────────────
        report_id = str(uuid.uuid4())
        report_doc = {
            "id":              report_id,
            "user_id":         request.user_id,
            "session_id":      request.session_id,
            "query_type":      QueryType.REPORT.value,
            "scientific_name": scientific_name,
            "common_name":     _first(traits_data.get("common_names")),
            "risk_level":      traits_data.get("human_risk_level"),
            "status":          ReportStatus.SUBMITTED.value,
            "severity":        incident_severity.value,
            "location": {
                "latitude":       request.latitude,
                "longitude":      request.longitude,
                "district":       loc_data.get("district"),
                "state":          loc_data.get("state"),
                "protected_area": loc_data.get("protected_area"),
                "formatted":      loc_data.get("formatted"),
            },
            "animal_traits": {
                "size":                 _trait(text_data, "size"),
                "colour":               _trait(text_data, "colour"),
                "behaviour":            _trait(text_data, "behaviour"),
                "distinctive_features": _trait(text_data, "distinctive_features") or [],
            },
        }
        try:
            await upsert_report(report_doc)
            logger.info("[reporter] Report saved: id=%s species=%s", report_id, scientific_name)
        except Exception as exc:
            logger.error("[reporter] Failed to save report: %s", exc)

        return {**traits_data, "report_id": report_id} if traits_data else {
            "scientific_name": scientific_name or "Unknown",
            "report_id": report_id,
        }


def _first(lst: Any) -> Optional[str]:
    return lst[0] if isinstance(lst, list) and lst else None


def _trait(text_data: Any, key: str) -> Any:
    """Read a field directly from text_data (flat schema after simplification)."""
    if isinstance(text_data, dict):
        return text_data.get(key)
    return None


def _map_severity(text_severity: str) -> IncidentSeverity:
    return {
        "critical":      IncidentSeverity.CRITICAL,
        "high":          IncidentSeverity.HIGH,
        "medium":        IncidentSeverity.MEDIUM,
    }.get(text_severity.lower(), IncidentSeverity.LOW)

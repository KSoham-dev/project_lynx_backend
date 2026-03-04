"""
agents/layer2/explorer.py
==========================
Layer 2 — Explorer Agent (EXPLORE query type)

Calls run_pipeline() for species traits and iucn_fetcher for raw IUCN data,
merges them into a combined response for the frontend.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from agents.layer0.receiver import AgentRequest

logger = logging.getLogger(__name__)


class ExplorerAgent:
    async def run(
        self,
        request: AgentRequest,
        tool_results: dict[str, Any],
    ) -> dict[str, Any]:
        """
        1. Extract scientific_name from tool_results.
        2. Call run_pipeline() → species traits.
        3. Call iucn_fetcher.get_iucn_raw() → raw IUCN data.
        4. Merge and return combined response.
        """
        text_data:  dict = tool_results.get("analyse_text") or {}
        image_data: dict = tool_results.get("analyse_image") or {}
        loc_data:   dict = tool_results.get("location_context") or {}

        scientific_name: Optional[str] = (
            image_data.get("scientific_name")
            or (text_data.get("scientific_name") if isinstance(text_data, dict) else None)
        )

        traits_data: dict[str, Any] = {}
        iucn_data:   dict[str, Any] = {}

        if scientific_name and scientific_name.lower() not in ("unknown", "unidentified"):
            # Run pipeline and raw IUCN fetch concurrently
            try:
                from pipeline.species_traits import run_pipeline
                from agents.layer2.iucn_fetcher import get_iucn_raw
                logger.info("[explorer] Fetching traits + IUCN for %r", scientific_name)
                traits_data, iucn_data = await asyncio.gather(
                    asyncio.to_thread(run_pipeline, scientific_name),
                    get_iucn_raw(scientific_name),
                    return_exceptions=False,
                )
                logger.info("[explorer] Done: traits=%d fields, iucn threats=%d",
                            len(traits_data), len(iucn_data.get("threats") or []))
            except Exception as exc:
                logger.error("[explorer] Enrichment failed for %r: %s", scientific_name, exc)

        return {
            **traits_data,
            "scientific_name": scientific_name or traits_data.get("scientific_name"),
            "iucn": {
                "assessment_id":        iucn_data.get("assessment_id"),
                "year_published":       iucn_data.get("year_published"),
                "red_list_category":    iucn_data.get("red_list_category"),
                "red_list_code":        iucn_data.get("red_list_code"),
                "population_trend":     iucn_data.get("population_trend"),
                "population_size":      iucn_data.get("population_size"),
                "geographic_range":     iucn_data.get("geographic_range"),
                "habitats":             iucn_data.get("habitats") or [],
                "threats":              iucn_data.get("threats") or [],
                "conservation_actions": iucn_data.get("conservation_actions") or [],
                "use_trade":            iucn_data.get("use_trade") or [],
                "references":           iucn_data.get("references") or [],
                "url":                  iucn_data.get("url"),
            },
            "location": {
                "district":       loc_data.get("district"),
                "state":          loc_data.get("state"),
                "country":        loc_data.get("country"),
                "protected_area": loc_data.get("protected_area"),
                "formatted":      loc_data.get("formatted"),
                "latitude":       request.latitude,
                "longitude":      request.longitude,
            } if loc_data else None,
        }

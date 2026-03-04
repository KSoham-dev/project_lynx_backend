"""
agents/layer2/species_info.py
==============================
Layer 2 — Species Info Agent (ENCYCLOPEDIA query type)

Calls iucn_fetcher.get_iucn_raw() directly and returns raw IUCN data.
No Groq enrichment — pure IUCN blob data for the encyclopedia view.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from agents.layer0.receiver import AgentRequest
from agents.layer2.species_cache import read_species_cache, write_species_cache

logger = logging.getLogger(__name__)


class SpeciesInfoAgent:
    async def run(
        self,
        request: AgentRequest,
        tool_results: dict[str, Any],
    ) -> dict[str, Any]:
        """
        1. Extract scientific_name from tool_results (analyse_text).
        2. Call get_iucn_raw(scientific_name) directly.
        3. Return structured IUCN data.
        """
        text_data: dict = tool_results.get("analyse_text") or {}
        image_data: dict = tool_results.get("analyse_image") or {}

        scientific_name: Optional[str] = (
            image_data.get("scientific_name")
            or (text_data.get("scientific_name") if isinstance(text_data, dict) else None)
            # Also try the message directly — for encyclopedia the message IS the name
            or ((request.message or "").strip() if " " in (request.message or "").strip() else None)
        )

        if not scientific_name or scientific_name.lower() in ("unknown", "unidentified"):
            logger.warning("[species_info] No scientific name found in tool_results")
            return {"error": "Could not identify species from the provided input."}

        try:
            from agents.layer2.iucn_fetcher import get_iucn_raw
            logger.info("[species_info] Fetching raw IUCN for %r", scientific_name)

            # ── Check Cosmos cache first ──────────────────────────────────────
            cached = await read_species_cache("iucn", scientific_name)
            if cached:
                logger.info("[species_info] Using cached IUCN for %r", scientific_name)
                return cached

            iucn_data = await get_iucn_raw(scientific_name)
            logger.info(
                "[species_info] Done: category=%r threats=%d",
                iucn_data.get("red_list_category"),
                len(iucn_data.get("threats") or []),
            )
            await write_species_cache("iucn", scientific_name, iucn_data)
            return iucn_data
        except FileNotFoundError:
            logger.warning("[species_info] No IUCN blob found for %r", scientific_name)
            return {"error": f"No IUCN data found for '{scientific_name}'."}
        except Exception as exc:
            logger.error("[species_info] Failed for %r: %s", scientific_name, exc)
            return {"error": str(exc)}

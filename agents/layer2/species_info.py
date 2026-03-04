"""
agents/layer2/species_info.py
==============================
Layer 2 — Species Info Agent

This is the FIRST agent invoked in Layer 2, regardless of query_type.
Its output (raw IUCN data) is shared with all query-specific agents
(Encyclopedia, Explorer, Reporter).

Responsibilities
----------------
1. Extract the best ``scientific_name`` from Layer 1 tool_results.
2. Check the Cosmos DB species_context_cache for a previous result.
3. If cache miss — fetch raw IUCN data from Azure Blob Storage.
4. Write result to cache and return the structured IUCN dict.
5. Return ``None`` if no species could be identified — callers must handle.

Output dict keys (from iucn_fetcher.get_iucn_raw)
--------------------------------------------------
    assessment_id, year_published, scientific_name, common_names,
    red_list_category, red_list_code, population_trend, population_size,
    geographic_range, habitats, threats, conservation_actions,
    use_trade, url, sis_taxon_id, references
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from agents.layer0.receiver import AgentRequest
from agents.layer2.iucn_fetcher import iucn_red_list_category
from agents.layer2.species_cache import read_species_cache, write_species_cache

logger = logging.getLogger(__name__)


def _extract_scientific_name(tool_results: dict[str, Any]) -> Optional[str]:
    """
    Best-effort extraction of scientific_name from Layer 1 tool_results.

    Priority: image_analysis (SpeciesNet/GPT) -> text_analysis (Groq LLM).
    Returns None if neither tool ran or both returned unknown/unidentified.
    """
    image_data: dict = tool_results.get("analyse_image") or {}
    text_data:  dict = tool_results.get("analyse_text")  or {}

    candidates = [
        image_data.get("scientific_name"),
        text_data.get("scientific_name") if isinstance(text_data, dict) else None,
    ]

    for name in candidates:
        if name and name.strip().lower() not in ("unknown", "unidentified", "", "none"):
            return name.strip()
    return None


class SpeciesInfoAgent:
    """
    Shared first-step Layer 2 agent.

    Always runs before any query-specific agent. Its return value is
    passed directly into every downstream agent's run() call as
    the iucn_data parameter.
    """

    async def run(
        self,
        request: AgentRequest,
        tool_results: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """
        Fetch raw IUCN data for the identified species.

        Returns
        -------
        dict
            Structured IUCN data dict on success.
        None
            If no species was identified or no IUCN blob exists.
        """
        scientific_name = _extract_scientific_name(tool_results)

        if not scientific_name:
            logger.warning("[species_info] No scientific name in tool_results — skipping IUCN fetch")
            return None

        logger.info("[species_info] START for %r", scientific_name)

        # Cache check
        cached = await read_species_cache("iucn", scientific_name)
        if cached:
            logger.info("[species_info] Cache HIT for %r", scientific_name)
            return cached

        # Blob fetch
        try:
            from agents.layer2.iucn_fetcher import get_iucn_raw

            iucn_data = await get_iucn_raw(scientific_name)
            logger.info(
                "[species_info] IUCN fetched: category=%r threats=%d",
                iucn_red_list_category(iucn_data),
                len(iucn_data.get("threats") or []),
            )
            await write_species_cache("iucn", scientific_name, iucn_data)
            return iucn_data

        except FileNotFoundError:
            logger.warning("[species_info] No IUCN blob found for %r", scientific_name)
            return None

        except Exception as exc:
            logger.error(
                "[species_info] IUCN fetch failed for %r: %s", scientific_name, exc, exc_info=True
            )
            return None

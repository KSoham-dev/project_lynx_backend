"""
agents/layer2/explorer.py
==========================
Layer 2 — Explorer Agent  (query_type = explore)

Invoked after SpeciesInfoAgent, which provides the shared raw IUCN data.
This agent enriches that foundation with:
  1. GPT-extracted biological traits  — via pipeline.species_traits.run_pipeline()
     (Groq llama-3.3-70b: lifespan, mass, length, description, risk, fun facts)
  2. iNaturalist representative photo  — URL + attribution credit
  3. Location context                  — district, state, protected area

Both the Groq enrichment and the iNaturalist call run CONCURRENTLY via
asyncio.gather to minimise total latency.

Input
-----
    iucn_data : dict from SpeciesInfoAgent (raw IUCN fields), or None.

Output
------
    A merged dict containing:
        - All raw IUCN fields (red_list_category, habitats, threats, ...)
        - GPT-enriched trait fields (lifespan_years, mass, short_description, ...)
        - photo_url, photo_credit
        - location sub-dict (district, state, country, protected_area, coordinates)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from fastapi import HTTPException

from agents.layer0.receiver import AgentRequest
from agents.layer2.iucn_fetcher import (
    iucn_habitat_names,
    iucn_population_trend,
    iucn_red_list_category,
    iucn_scientific_name,
    iucn_threat_titles,
    is_iucn_not_found,
)
from agents.layer2.inaturalist import get_inaturalist_photo
from agents.layer2.species_cache import read_species_cache, write_species_cache

logger = logging.getLogger(__name__)

def _iucn_summary(iucn_data: dict[str, Any]) -> str:
    """
    Distil the full IUCN assessment dict into a compact plain-text summary.

    Rationale: the raw IUCN blob can be several kilobytes of deeply nested JSON
    (full taxonomy, all habitats, all threats, narrative descriptions, references,
    supplementary data, …).  The Groq trait-extraction call only needs conservation
    headline facts to infer lifespan, mass, fun facts, etc.  Sending the full dict
    wastes > 90 % of the tokens on fields Groq never reads, inflates cost, and can
    push the useful content (scientific name, category, threats) deep into a noisy
    context where the model is more likely to ignore it.  This summary keeps exactly
    the four fields that matter.
    """
    return (
        f"Red List category: {iucn_red_list_category(iucn_data) or 'Unknown'}\n"
        f"Population trend: {iucn_population_trend(iucn_data) or 'Unknown'}\n"
        f"Top threats: {', '.join(iucn_threat_titles(iucn_data)) or 'Unknown'}\n"
        f"Habitats: {', '.join(iucn_habitat_names(iucn_data)) or 'Unknown'}"
    )


# GPT trait fields produced by run_pipeline() / Groq enrichment
_GPT_TRAIT_FIELDS = (
    "lifespan_years",
    "mass",
    "length",
    "short_description",
    "human_risk_level",
    "human_threat_level",
    "fun_fact_1",
    "fun_fact_2",
    "fun_fact_3",
)


async def _get_gpt_traits(
    scientific_name: str,
    iucn_data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Fetch GPT-enriched traits from cache, or run the Groq pipeline.

    Before calling run_pipeline:
      - A compact plain-text IUCN summary is built via _iucn_summary() instead
        of forwarding the raw dict.  This keeps the Groq context to ~4 lines of
        conservation facts rather than kilobytes of nested JSON.
      - Wikipedia text is fetched concurrently here and passed in explicitly so
        Groq has the best available natural-language context for trait inference.
    Always non-fatal — returns {} on any failure.
    """
    cached = await read_species_cache("traits", scientific_name)
    if cached:
        logger.info("[explorer] Traits cache HIT for %r", scientific_name)
        return cached
    try:
        from pipeline.species_traits import run_pipeline, _wiki_extract
        logger.info("[explorer] Running Groq traits pipeline for %r", scientific_name)

        # Build the compact IUCN summary; None when no IUCN data is available
        # (run_pipeline will fall back to the Azure Blob fetch in that case).
        summary: Optional[str] = _iucn_summary(iucn_data) if iucn_data else None

        # Fetch Wikipedia text asynchronously — run_pipeline skips this on the
        # caller-supplied path, so we fetch it here and forward it explicitly.
        wiki_text: str = await asyncio.to_thread(_wiki_extract, scientific_name)
        logger.info(
            "[explorer] Wikipedia fetch done for %r: %d chars",
            scientific_name, len(wiki_text),
        )

        traits = await asyncio.to_thread(
            run_pipeline, scientific_name, None, wiki_text, summary
        )
        logger.info("[explorer] Groq traits done: %d fields", len(traits))
        await write_species_cache("traits", scientific_name, traits)
        return traits
    except Exception as exc:
        logger.error("[explorer] Groq pipeline failed for %r: %s", scientific_name, exc, exc_info=True)
        return {}


class ExplorerAgent:
    """Merge raw IUCN + GPT-enriched traits + iNaturalist photo."""

    async def run(
        self,
        request: AgentRequest,
        tool_results: dict[str, Any],
        iucn_data: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        request      : normalised AgentRequest
        tool_results : Layer 1 outputs
        iucn_data    : output of SpeciesInfoAgent — may be None

        Returns
        -------
        Merged exploration dict.
        """
        if is_iucn_not_found(iucn_data):
            scientific_name = iucn_data.get("scientific_name", "unknown")
            logger.warning(
                "[explorer] Species not in any database: %r",
                scientific_name,
            )
            raise HTTPException(
                status_code=404,
                detail=iucn_data["message"],
            )

        if not iucn_data:
            logger.warning("[explorer] No IUCN data — cannot enrich")
            raise HTTPException(
                status_code=404,
                detail="Could not identify the species from the provided input.",
            )

        scientific_name: str = iucn_scientific_name(iucn_data) or ""
        loc_data: dict = tool_results.get("location_context") or {}

        logger.info(
            "[explorer] START for %r — fetching GPT traits + iNaturalist concurrently",
            scientific_name,
        )

        # Run both enrichment sources concurrently.
        # Pass iucn_data so the traits pipeline skips blob storage when the
        # data was sourced from the live IUCN API fallback.
        gpt_traits, (photo_url, photo_credit) = await asyncio.gather(
            _get_gpt_traits(scientific_name, iucn_data),
            get_inaturalist_photo(scientific_name),
        )

        # Extract only the GPT-specific fields to overlay on IUCN data
        gpt_overlay = {k: gpt_traits.get(k) for k in _GPT_TRAIT_FIELDS if gpt_traits.get(k)}

        result = {
            # Raw IUCN fields form the base
            **iucn_data,
            # GPT traits overlay (may add/overwrite some fields)
            **gpt_overlay,
            # iNaturalist photo
            "photo_url":    photo_url    or "Not available",
            "photo_credit": photo_credit or "Not available",
            # Location context from Layer 1
            "location": {
                "latitude":       request.latitude,
                "longitude":      request.longitude,
                "district":       loc_data.get("district"),
                "state":          loc_data.get("state"),
                "country":        loc_data.get("country"),
                "protected_area": loc_data.get("protected_area"),
                "formatted":      loc_data.get("formatted"),
            },
        }

        logger.info(
            "[explorer] Done for %r: category=%r traits=%d photo=%s",
            scientific_name,
            iucn_red_list_category(iucn_data),
            len(gpt_overlay),
            "yes" if photo_url else "no",
        )
        return result

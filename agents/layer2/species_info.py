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

import aiohttp

from agents.layer0.receiver import AgentRequest
from agents.layer2.iucn_fetcher import iucn_red_list_category
from agents.layer2.species_cache import read_species_cache, write_species_cache

logger = logging.getLogger(__name__)

_IUCN_API_BASE = "https://api.iucnredlist.org/api/v4"


def _not_in_database_response(scientific_name: str) -> dict[str, Any]:
    """Return a sentinel dict signalling the species was not found in any database."""
    return {
        "_not_in_database": True,
        "scientific_name": scientific_name,
        "message": (
            f"We regret to inform you that our database does not currently contain "
            f"information regarding '{scientific_name}'. Our team has been notified "
            f"and will work on updating our records accordingly. "
            f"In the meantime, we recommend referring to open-source media sources "
            f"for information about this species. "
            f"Scientific name: {scientific_name}."
        ),
    }


async def _fetch_iucn_api(scientific_name: str) -> Optional[dict[str, Any]]:
    """
    Fallback: fetch species data from the live IUCN Red List API v4.

    Makes two sequential calls:
      1. GET /taxa/scientific_name?genus_name=X&species_name=Y  → resolve assessment_id
      2. GET /assessment/<assessment_id>                        → full assessment data

    Returns the raw assessment dict on success, or None if the species is not
    found or any error occurs.
    """
    from agents.config import get_shared_settings

    api_key = get_shared_settings().iucn_api_key
    if not api_key:
        logger.warning("[species_info] IUCN_API_KEY not configured — skipping live API fallback")
        return None

    parts = scientific_name.strip().split()
    if len(parts) < 2:
        logger.warning("[species_info] Cannot parse genus/species from %r", scientific_name)
        return None

    genus_name, species_name = parts[0], parts[1]
    headers = {"accept": "application/json", "Authorization": api_key}

    try:
        async with aiohttp.ClientSession() as session:
            # ── Step 1: resolve taxon → get assessment_id ─────────────────────
            params = {"genus_name": genus_name, "species_name": species_name}
            logger.info(
                "[species_info] IUCN API taxa lookup: genus=%r species=%r",
                genus_name, species_name,
            )
            async with session.get(
                f"{_IUCN_API_BASE}/taxa/scientific_name",
                params=params,
                headers=headers,
            ) as resp:
                if resp.status == 404:
                    logger.info(
                        "[species_info] IUCN API: taxon not found for %r", scientific_name
                    )
                    return None
                resp.raise_for_status()
                taxa_data: dict[str, Any] = await resp.json()

            assessments: list = taxa_data.get("assessments") or []
            if not assessments:
                logger.info(
                    "[species_info] IUCN API: no assessments listed for %r", scientific_name
                )
                return None

            assessment_id = assessments[0].get("assessment_id")
            if not assessment_id:
                logger.warning(
                    "[species_info] IUCN API: assessment_id missing in response for %r",
                    scientific_name,
                )
                return None

            # ── Step 2: fetch full assessment ─────────────────────────────────
            logger.info(
                "[species_info] IUCN API assessment fetch: id=%s for %r",
                assessment_id, scientific_name,
            )
            async with session.get(
                f"{_IUCN_API_BASE}/assessment/{assessment_id}",
                headers=headers,
            ) as resp:
                if resp.status == 404:
                    logger.warning(
                        "[species_info] IUCN API: assessment %s not found", assessment_id
                    )
                    return None
                resp.raise_for_status()
                assessment: dict[str, Any] = await resp.json()

        logger.info(
            "[species_info] IUCN API: successfully fetched assessment %s for %r",
            assessment_id, scientific_name,
        )
        return assessment

    except Exception as exc:
        logger.error(
            "[species_info] IUCN API fetch failed for %r: %s",
            scientific_name, exc, exc_info=True,
        )
        return None


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
            logger.warning(
                "[species_info] No IUCN blob found for %r — trying live IUCN API",
                scientific_name,
            )
            api_data = await _fetch_iucn_api(scientific_name)
            if api_data:
                logger.info(
                    "[species_info] IUCN API fallback SUCCESS for %r — category=%r",
                    scientific_name, iucn_red_list_category(api_data),
                )
                await write_species_cache("iucn", scientific_name, api_data)
                return api_data

            # Species not found in blob storage or live IUCN API — log for DB update
            logger.warning(
                "[species_info] SPECIES_NOT_IN_DATABASE species=%r — queued for database update",
                scientific_name,
            )
            return _not_in_database_response(scientific_name)

        except Exception as exc:
            logger.error(
                "[species_info] IUCN fetch failed for %r: %s", scientific_name, exc, exc_info=True
            )
            return None

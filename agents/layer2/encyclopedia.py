"""
agents/layer2/encyclopedia.py
==============================
Layer 2 — Encyclopedia Agent  (query_type = encyclopedia)

Invoked after SpeciesInfoAgent has already fetched and shared the raw
IUCN data.  This agent's only job is to enrich that data with an
iNaturalist photo (URL + credits) and return the combined response.

No additional LLM calls are made — the IUCN blob already contains rich
conservation and taxonomy information sufficient for an encyclopedia view.

Input
-----
    iucn_data   : dict returned by SpeciesInfoAgent (raw IUCN fields)
                  or None if no species was identified.

Output
------
    {
        # all raw IUCN fields (assessment_id, scientific_name, common_names,
        #   red_list_category, habitats, threats, conservation_actions, ...),
        "photo_url"   : str | "Not available",
        "photo_credit": str | "Not available",
    }
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import HTTPException

from agents.layer0.receiver import AgentRequest
from agents.layer2.iucn_fetcher import iucn_red_list_category, iucn_scientific_name, is_iucn_not_found
from agents.layer2.inaturalist import get_inaturalist_photo

logger = logging.getLogger(__name__)


class EncyclopediaAgent:
    """Enrich raw IUCN data with iNaturalist photo for the encyclopedia view."""

    async def run(
        self,
        request: AgentRequest,
        tool_results: dict[str, Any],
        iucn_data: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        request      : normalised AgentRequest from the receiver
        tool_results : Layer 1 outputs (analyse_text, analyse_image, location_context)
        iucn_data    : output of SpeciesInfoAgent.run() — may be None

        Returns
        -------
        dict with IUCN fields + photo_url + photo_credit
        """
        if is_iucn_not_found(iucn_data):
            scientific_name = iucn_data.get("scientific_name", "unknown")
            logger.warning(
                "[encyclopedia] Species not in any database: %r",
                scientific_name,
            )
            raise HTTPException(
                status_code=404,
                detail=iucn_data["message"],
            )

        if not iucn_data:
            # Species Info Agent could not identify or fetch the species.
            scientific_name = _best_name(tool_results, request)
            logger.warning("[encyclopedia] No IUCN data available for %r", scientific_name)
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No IUCN data found for '{scientific_name}'."
                    if scientific_name
                    else "Could not identify the species from the provided input."
                ),
            )

        scientific_name: str = iucn_scientific_name(iucn_data) or ""
        logger.info("[encyclopedia] Fetching iNaturalist photo for %r", scientific_name)

        photo_url, photo_credit = await get_inaturalist_photo(scientific_name)

        result = {
            **iucn_data,
            "photo_url":    photo_url    or "Not available",
            "photo_credit": photo_credit or "Not available",
        }
        logger.info(
            "[encyclopedia] Done for %r: category=%r photo=%s",
            scientific_name,
            iucn_red_list_category(iucn_data),
            "yes" if photo_url else "no",
        )
        return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _best_name(
    tool_results: dict[str, Any],
    request: AgentRequest,
) -> Optional[str]:
    """Return the best available species name from any source."""
    image_data: dict = tool_results.get("analyse_image") or {}
    text_data:  dict = tool_results.get("analyse_text")  or {}
    return (
        image_data.get("scientific_name")
        or (text_data.get("scientific_name") if isinstance(text_data, dict) else None)
        or (request.message or "").strip() or None
    )

"""
agents/layer0/router.py
========================
FastAPI router — POST /agent/chat

Flow:
  1. Receiver   → load/create Cosmos session
  2. Orchestrator → LLM tool loop (analyse_image, analyse_text, location_context)
                  → returns (response_text, tool_results)
  3. Layer 2    → enrichment + persistence based on query_type
  4. Return ChatResponse {response, data}
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from agents.enums import QueryType
from agents.layer0.orchestrator import run_orchestrator
from agents.layer0.receiver import ReceiverInput, run_receiver

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])


# ── Response schema ───────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    session_id: str
    user_id: str
    query_type: str
    response: str                          # LLM conversational text
    data: Optional[dict[str, Any]] = None  # structured species / IUCN data


# ── Layer 2 dispatch ──────────────────────────────────────────────────────────

async def _dispatch_layer2(
    query_type: QueryType,
    request,
    tool_results: dict[str, Any],
) -> Optional[dict[str, Any]]:
    try:
        if query_type == QueryType.REPORT:
            from agents.layer2.reporter import ReporterAgent
            return await ReporterAgent().run(request, tool_results)

        elif query_type == QueryType.EXPLORE:
            from agents.layer2.explorer import ExplorerAgent
            return await ExplorerAgent().run(request, tool_results)

        elif query_type == QueryType.ENCYCLOPEDIA:
            from agents.layer2.species_info import SpeciesInfoAgent
            return await SpeciesInfoAgent().run(request, tool_results)

    except Exception as exc:
        logger.error("Layer 2 dispatch failed for %s: %s", query_type, exc, exc_info=True)

    return None


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ReceiverInput, request: Request) -> ChatResponse:
    """
    Primary entry point for the Prahari agent system.

    - **user_id**: unique user/device identifier
    - **session_id**: resume an existing conversation (omit to start fresh)
    - **query_type**: ``report`` | ``explore`` | ``encyclopedia``
    - **message**: user's text message
    - **image_url**: optional image URL
    - **latitude** / **longitude**: optional GPS coordinates
    """
    try:
        agent_request = await run_receiver(payload, request)
        response_text, tool_results = await run_orchestrator(agent_request)

        structured_data = await _dispatch_layer2(
            payload.query_type, agent_request, tool_results
        )

    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Orchestrator error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /agent/chat: %s", exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    return ChatResponse(
        session_id=agent_request.session_id,
        user_id=payload.user_id,
        query_type=payload.query_type.value,
        response=response_text,
        data=structured_data,
    )

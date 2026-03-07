"""
agents/layer0/router.py
========================
FastAPI router — POST /agent/chat

This is a thin HTTP adapter only. All agent logic lives in the orchestrator.

Flow:
  1. Receiver    → validate request, load/create Cosmos session → AgentRequest
  2. Orchestrator → full pipeline (Layer 1 tools + Layer 2 agent) → structured_data
  3. Return ChatResponse {session_id, user_id, query_type, response, data}
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from agents.layer0.orchestrator import run_orchestrator
from agents.layer0.receiver import ReceiverInput, run_receiver

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])


# ── Response schema ───────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    session_id: str
    user_id: str
    query_type: str
    response: str
    data: Optional[dict[str, Any]] = None


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ReceiverInput, request: Request) -> ChatResponse:
    """
    Primary entry point for the Prahari agent system.

    - **user_id**: unique user/device identifier
    - **session_id**: resume an existing conversation (omit to start fresh)
    - **query_type**: ``report`` | ``explore`` | ``encyclopedia``
    - **message**: user's text message (optional)
    - **image_url**: optional image URL
    - **latitude** / **longitude**: optional GPS coordinates
    """
    try:
        agent_request = await run_receiver(payload, request)
        response_text, structured_data = await run_orchestrator(agent_request)

    except HTTPException:
        raise  # pass through HTTPException raised by agents (e.g. unidentified species)
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
        data=structured_data or None,
    )


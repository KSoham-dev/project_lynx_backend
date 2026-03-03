"""
agents/layer0/router.py
========================
FastAPI router exposing the Layer 0 Orchestrator via HTTP.

Endpoints
---------
POST /agent/chat
    Accept a frontend payload, run it through the Receiver then Orchestrator,
    and return the AI response together with the session_id so the client can
    resume the conversation on subsequent calls.

Device headers (set by React frontend)
---------------------------------------
    X-Device-ID   – stable UUID stored in client localStorage
    X-FCM-Token   – Firebase Cloud Messaging token for web-push notifications
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from agents.layer0.orchestrator import run_orchestrator
from agents.layer0.receiver import ReceiverInput, run_receiver

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])


# ── Response schema ───────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    """Response returned by POST /agent/chat."""

    session_id: str
    user_id: str
    query_type: str
    response: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "user_id": "ranger-42",
                "query_type": "report",
                "response": (
                    "The species you spotted is likely a Bengal Tiger "
                    "(Panthera tigris tigris). It is classified as Endangered "
                    "by the IUCN. Rangers should keep a safe distance of at "
                    "least 100 metres and avoid direct eye contact."
                ),
            }
        }
    }


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to the Prahari orchestrator",
    description=(
        "Accepts a user message (with optional image URL and GPS coordinates), "
        "runs it through the Receiver Agent to initialise / resume a session, "
        "then invokes the Orchestrator Agent.  "
        "Returns the AI response and the ``session_id`` for conversation continuity.\n\n"
        "**Device headers** (for push notifications — React frontend):\n"
        "- `X-Device-ID`: stable UUID from localStorage\n"
        "- `X-FCM-Token`: Firebase Cloud Messaging web-push token"
    ),
)
async def chat(payload: ReceiverInput, request: Request) -> ChatResponse:
    """
    Primary entry point for the Prahari agent system.

    The FastAPI ``Request`` object is injected automatically by FastAPI
    so that the Receiver can extract device metadata from headers.

    - **user_id**: Unique identifier for the calling user / device.
    - **session_id**: Resume an existing conversation. Omit to start fresh.
    - **query_type**: ``report`` | ``explore`` | ``encyclopedia``
    - **message**: The text message from the user.
    - **image_url**: Optional public URL of an image to identify.
    - **latitude** / **longitude**: Optional GPS coordinates.
    """
    try:
        agent_request = await run_receiver(payload, request)
        response_text = await run_orchestrator(agent_request)

    except ValueError as exc:
        logger.warning("Receiver/Orchestrator validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Orchestrator runtime error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /agent/chat: %s", exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    return ChatResponse(
        session_id=agent_request.session_id,
        user_id=payload.user_id,
        query_type=payload.query_type.value,
        response=response_text,
    )

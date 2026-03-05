"""
agents/layer0/receiver.py
==========================
Receiver Agent — the front door of the Prahari agent system.

Responsibilities
----------------
1. Accept raw frontend payloads (validated via ``ReceiverInput``).
2. Extract device metadata from request headers (device_id, FCM token, user-agent)
   for the future targeted push-notification pipeline.
3. Generate a ``session_id`` UUID if the caller did not provide one.
4. Load an existing Cosmos DB session or create a fresh one.
5. Return a normalised ``AgentRequest`` ready for the Orchestrator.

Device headers (set by React frontend)
---------------------------------------
    X-Device-ID   – client-generated UUID (localStorage)
    X-FCM-Token   – Firebase Cloud Messaging token (web-push)
    User-Agent    – standard HTTP header

Typical flow
-----------
    receiver_input  = ReceiverInput(**json_body)
    agent_request   = await run_receiver(receiver_input, request)
    response        = await run_orchestrator(agent_request)
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from fastapi import Request
from pydantic import BaseModel, Field

from agents.enums import QueryType
from agents.state.data_schemas import DeviceInfo
from agents.state.schemas import SessionDocument
from agents.state.session_store import create_session, get_session

logger = logging.getLogger(__name__)


# ── Input schema (mirrors what the frontend sends) ────────────────────────────

class ReceiverInput(BaseModel):
    """
    Schema for the raw payload sent by the frontend / mobile client.

    ``session_id`` is optional — when omitted the Receiver creates a new
    session and returns the generated UUID so the client can reuse it.
    ``query_type`` is **required** and drives which Layer 2 agent handles
    the final output.
    """

    user_id: str = Field(..., description="Unique identifier for the calling user.")
    session_id: Optional[str] = Field(
        default=None,
        description="Existing session UUID to resume. Omit to start a new session.",
    )
    query_type: QueryType = Field(
        ...,
        description=(
            "Type of query: 'report' (incident report), "
            "'explore' (area exploration), 'encyclopedia' (species facts)."
        ),
    )
    message: Optional[str] = Field(default=None, description="The user's text message / query.")
    image_url: Optional[str] = Field(
        default=None, description="Public URL of an image attached to this turn."
    )
    latitude: Optional[float] = Field(default=None, description="GPS latitude.")
    longitude: Optional[float] = Field(default=None, description="GPS longitude.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "ranger-42",
                "query_type": "report",
                "image_url": "https://inaturalist-open-data.s3.amazonaws.com/photos/605060662/large.jpg",
                "latitude": 26.01,
                "longitude": 76.50,
            }
        }
    }


# ── Device metadata (extracted from HTTP headers, not from the body) ──────────

class DeviceContext(BaseModel):
    """Device metadata extracted from HTTP request headers."""
    device_id: Optional[str] = None     # X-Device-ID header
    fcm_token: Optional[str] = None     # X-FCM-Token header (React web push)
    user_agent: Optional[str] = None    # standard User-Agent header
    platform: str = "web"               # always "web" for React frontend


def extract_device_context(request: Optional[Request]) -> DeviceContext:
    """
    Pull device metadata from FastAPI request headers.

    The React frontend should set:
      - ``X-Device-ID``  — a stable UUID stored in localStorage
      - ``X-FCM-Token``  — Firebase web-push token (refreshed as needed)

    Returns a :class:`DeviceContext` (all fields optional — graceful if absent).
    """
    if request is None:
        return DeviceContext()

    headers = request.headers
    return DeviceContext(
        device_id=headers.get("x-device-id"),
        fcm_token=headers.get("x-fcm-token"),
        user_agent=headers.get("user-agent"),
        platform="web",
    )


# ── Normalised internal request (passed to Orchestrator) ─────────────────────

class AgentRequest(BaseModel):
    """
    Enriched, normalised request object produced by the Receiver.

    Guaranteed non-null fields after receiver runs:
      - session_id (UUID)
      - session    (loaded/created SessionDocument)
      - query_type
    """

    user_id: str
    session_id: str
    query_type: QueryType
    message: Optional[str] = None
    image_url: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    session: SessionDocument
    device_context: DeviceContext = Field(default_factory=DeviceContext)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def geo_context(self) -> Optional[dict[str, Any]]:
        """Returns lat/lon dict or None."""
        if self.latitude is not None and self.longitude is not None:
            return {"latitude": self.latitude, "longitude": self.longitude}
        return None

    @property
    def device_info(self) -> Optional[DeviceInfo]:
        """
        Convert device context to a :class:`DeviceInfo` for persistence.
        Returns ``None`` if no device_id was provided.
        """
        if not self.device_context.device_id:
            return None
        return DeviceInfo(
            device_id=self.device_context.device_id,
            fcm_token=self.device_context.fcm_token,
            user_agent=self.device_context.user_agent,
            platform=self.device_context.platform,
        )


# ── Public entry point ────────────────────────────────────────────────────────

async def run_receiver(
    payload: ReceiverInput,
    request: Optional[Request] = None,
) -> AgentRequest:
    """
    Validate the incoming frontend payload, extract device metadata,
    and initialise the Cosmos DB session.

    Steps
    -----
    1. Extract device context from HTTP headers.
    2. Resolve ``session_id`` — use provided value or generate UUID4.
    3. Look up session in *prahari-state/sessions*.
    4. If not found, create a fresh session document.
    5. Return a fully populated :class:`AgentRequest`.

    Parameters
    ----------
    payload:
        Validated request body from the FastAPI endpoint.
    request:
        FastAPI ``Request`` object for header access (device metadata).

    Returns
    -------
    AgentRequest
        Enriched request with guaranteed session, query_type, and device context.
    """
    # Step 1 — extract device metadata from headers
    device_ctx = extract_device_context(request)

    # Step 2 — resolve session ID
    session_id: str = payload.session_id or str(uuid.uuid4())

    # Step 3 — attempt to load existing session
    session: Optional[SessionDocument] = await get_session(session_id)

    # Step 4 — create if not found
    if session is None:
        logger.info(
            "Creating new session: id=%s user=%s query_type=%s",
            session_id,
            payload.user_id,
            payload.query_type,
        )
        session = await create_session(
            user_id=payload.user_id,
            query_type=payload.query_type,
            session_id=session_id,
            metadata={
                "initial_image_url": payload.image_url,
                "initial_latitude":  payload.latitude,
                "initial_longitude": payload.longitude,
                "device_id":         device_ctx.device_id,
            },
        )
    else:
        logger.info(
            "Resumed session: id=%s user=%s messages=%d",
            session_id,
            payload.user_id,
            len(session.messages),
        )

    # Step 5 — return normalised request
    return AgentRequest(
        user_id=payload.user_id,
        session_id=session_id,
        query_type=payload.query_type,
        message=payload.message,
        image_url=payload.image_url,
        latitude=payload.latitude,
        longitude=payload.longitude,
        session=session,
        device_context=device_ctx,
    )

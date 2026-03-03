"""
agents/state/session_store.py
==============================
Async CRUD helpers for the ``sessions`` container in *prahari-state*.

Shared across all agent layers — import from agents.state.session_store,
never from agents.layer0.state.session_store.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from azure.cosmos.exceptions import CosmosResourceNotFoundError

from agents.enums import AgentLayer, MessageRole, QueryType
from agents.state.containers import StateContainers
from agents.state.cosmos_client import get_state_container
from agents.state.schemas import MessageRecord, SessionDocument

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


async def create_session(
    user_id: str,
    query_type: QueryType,
    session_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> SessionDocument:
    """Create a new blank session document in *prahari-state/sessions*."""
    sid = session_id or str(uuid.uuid4())
    doc = SessionDocument(
        id=sid,
        session_id=sid,
        user_id=user_id,
        query_type=query_type,
        metadata=metadata or {},
    )
    container = await get_state_container(StateContainers.SESSIONS)
    await container.create_item(body=doc.to_cosmos())
    logger.info("Session created: id=%s user=%s query_type=%s", sid, user_id, query_type)
    return doc


async def get_session(session_id: str) -> Optional[SessionDocument]:
    """Fetch a session by ``session_id``. Returns ``None`` if not found."""
    container = await get_state_container(StateContainers.SESSIONS)
    try:
        doc = await container.read_item(item=session_id, partition_key=session_id)
        return SessionDocument.from_cosmos(doc)
    except CosmosResourceNotFoundError:
        logger.debug("Session not found: id=%s", session_id)
        return None


async def upsert_session(doc: SessionDocument) -> None:
    """Insert or replace an entire session document."""
    doc.updated_at = _utcnow()
    container = await get_state_container(StateContainers.SESSIONS)
    await container.upsert_item(body=doc.to_cosmos())
    logger.debug("Session upserted: id=%s", doc.id)


async def append_message(
    session_id: str,
    role: MessageRole,
    content: str,
    tool_name: Optional[str] = None,
    layer: Optional[AgentLayer] = None,
) -> None:
    """
    Append a :class:`MessageRecord` to an existing session and persist it.

    Raises
    ------
    ValueError
        If no session with ``session_id`` exists.
    """
    doc = await get_session(session_id)
    if doc is None:
        raise ValueError(
            f"Session '{session_id}' not found. "
            "Create it first via create_session()."
        )

    doc.messages.append(
        MessageRecord(
            role=role,
            content=content,
            tool_name=tool_name,
            layer=layer,
        )
    )
    doc.updated_at = _utcnow()

    container = await get_state_container(StateContainers.SESSIONS)
    await container.upsert_item(body=doc.to_cosmos())
    logger.debug("Message appended: session=%s role=%s layer=%s", session_id, role, layer)

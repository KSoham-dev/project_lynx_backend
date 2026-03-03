"""
tests/test_layer0/test_session_store.py
========================================
Unit tests for agents/layer0/state/session_store.py
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.enums import AgentLayer, MessageRole, QueryType
from agents.state.schemas import SessionDocument
from agents.state.session_store import (
    append_message,
    create_session,
    get_session,
    upsert_session,
)


@pytest.fixture
def mock_container():
    return AsyncMock()


@pytest.fixture(autouse=True)
def patch_state_container(mock_container):
    with patch(
        "agents.state.session_store.get_state_container",
        new=AsyncMock(return_value=mock_container),
    ):
        yield mock_container


@pytest.mark.asyncio
async def test_create_session_generates_uuid(patch_state_container):
    patch_state_container.create_item = AsyncMock()
    doc = await create_session(user_id="u1", query_type=QueryType.REPORT)
    assert doc.user_id == "u1"
    assert doc.query_type == QueryType.REPORT
    assert len(doc.id) == 36  # UUID4
    patch_state_container.create_item.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_session_uses_provided_id(patch_state_container):
    patch_state_container.create_item = AsyncMock()
    doc = await create_session(user_id="u2", query_type=QueryType.EXPLORE, session_id="fixed-id")
    assert doc.id == "fixed-id"
    assert doc.session_id == "fixed-id"


@pytest.mark.asyncio
async def test_get_session_returns_none_when_not_found(patch_state_container):
    from azure.cosmos.exceptions import CosmosResourceNotFoundError
    patch_state_container.read_item = AsyncMock(
        side_effect=CosmosResourceNotFoundError(message="not found", response=MagicMock())
    )
    result = await get_session("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_get_session_returns_document(patch_state_container):
    existing = SessionDocument(id="s1", session_id="s1", user_id="u3", query_type=QueryType.ENCYCLOPEDIA)
    patch_state_container.read_item = AsyncMock(return_value=existing.to_cosmos())
    result = await get_session("s1")
    assert result is not None
    assert result.id == "s1"
    assert result.query_type == QueryType.ENCYCLOPEDIA


@pytest.mark.asyncio
async def test_append_message_uses_enum_role(patch_state_container):
    existing = SessionDocument(id="s2", session_id="s2", user_id="u4", query_type=QueryType.REPORT)
    patch_state_container.read_item = AsyncMock(return_value=existing.to_cosmos())
    patch_state_container.upsert_item = AsyncMock()

    await append_message("s2", role=MessageRole.USER, content="hello", layer=AgentLayer.LAYER0)
    patch_state_container.upsert_item.assert_awaited_once()
    upserted = patch_state_container.upsert_item.call_args.kwargs.get(
        "body"
    ) or patch_state_container.upsert_item.call_args.args[0]
    assert upserted["messages"][0]["role"] == "user"
    assert upserted["messages"][0]["layer"] == "layer0"


@pytest.mark.asyncio
async def test_append_message_raises_when_session_not_found(patch_state_container):
    from azure.cosmos.exceptions import CosmosResourceNotFoundError
    patch_state_container.read_item = AsyncMock(
        side_effect=CosmosResourceNotFoundError(message="not found", response=MagicMock())
    )
    with pytest.raises(ValueError, match="not found"):
        await append_message("bad-id", role=MessageRole.USER, content="hi")

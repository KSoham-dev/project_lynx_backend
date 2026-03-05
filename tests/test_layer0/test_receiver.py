"""
tests/test_layer0/test_receiver.py
====================================
Unit tests for agents/layer0/receiver.py
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.enums import QueryType
from agents.layer0.receiver import (
    DeviceContext,
    ReceiverInput,
    extract_device_context,
    run_receiver,
)
from agents.state.schemas import SessionDocument


def _make_session(session_id: str = "s-1", user_id: str = "u-1") -> SessionDocument:
    return SessionDocument(
        id=session_id,
        session_id=session_id,
        user_id=user_id,
        query_type=QueryType.REPORT,
    )


@pytest.fixture(autouse=True)
def patch_store():
    with (
        patch(
            "agents.layer0.receiver.get_session",
            new=AsyncMock(return_value=None),
        ) as mock_get,
        patch(
            "agents.layer0.receiver.create_session",
            new=AsyncMock(
                side_effect=lambda user_id, query_type, session_id, **kw:
                    _make_session(session_id, user_id)
            ),
        ) as mock_create,
    ):
        yield mock_get, mock_create


@pytest.mark.asyncio
async def test_new_session_generated_when_none_provided(patch_store):
    _, mock_create = patch_store
    payload = ReceiverInput(user_id="ranger-1", query_type=QueryType.REPORT, message="Hello!")
    req = await run_receiver(payload)
    assert len(req.session_id) == 36   # UUID4
    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_existing_session_resumed(patch_store):
    mock_get, mock_create = patch_store
    existing = _make_session("existing-id", "ranger-2")
    mock_get.return_value = existing
    payload = ReceiverInput(
        user_id="ranger-2",
        session_id="existing-id",
        query_type=QueryType.EXPLORE,
        message="What animals live here?",
    )
    req = await run_receiver(payload)
    assert req.session_id == "existing-id"
    mock_create.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_type_propagated(patch_store):
    payload = ReceiverInput(
        user_id="r-3",
        query_type=QueryType.ENCYCLOPEDIA,
        message="Tell me about tigers.",
    )
    req = await run_receiver(payload)
    assert req.query_type == QueryType.ENCYCLOPEDIA


@pytest.mark.asyncio
async def test_geo_context_populated(patch_store):
    payload = ReceiverInput(
        user_id="r-4",
        query_type=QueryType.REPORT,
        message="Leopard sighting.",
        latitude=26.0,
        longitude=76.5,
    )
    req = await run_receiver(payload)
    assert req.geo_context == {"latitude": 26.0, "longitude": 76.5}


@pytest.mark.asyncio
async def test_geo_context_none_when_missing(patch_store):
    payload = ReceiverInput(user_id="r-5", query_type=QueryType.EXPLORE, message="Hello")
    req = await run_receiver(payload)
    assert req.geo_context is None


@pytest.mark.asyncio
async def test_device_context_populated_from_mock_request(patch_store):
    """Simulate a FastAPI Request with device headers."""
    mock_request = MagicMock()
    mock_request.headers = {
        "x-device-id": "device-uuid-1",
        "x-fcm-token": "fcm-tok-abc",
        "user-agent": "Mozilla/5.0 (React App)",
    }
    payload = ReceiverInput(user_id="r-6", query_type=QueryType.REPORT, message="Test")
    req = await run_receiver(payload, request=mock_request)
    assert req.device_context.device_id == "device-uuid-1"
    assert req.device_context.fcm_token == "fcm-tok-abc"


@pytest.mark.asyncio
async def test_device_info_property_with_device_id(patch_store):
    mock_request = MagicMock()
    mock_request.headers = {"x-device-id": "d-1", "user-agent": "ReactApp/1.0"}
    payload = ReceiverInput(user_id="r-7", query_type=QueryType.REPORT, message="Test")
    req = await run_receiver(payload, request=mock_request)
    info = req.device_info
    assert info is not None
    assert info.device_id == "d-1"


@pytest.mark.asyncio
async def test_device_info_none_without_device_id(patch_store):
    payload = ReceiverInput(user_id="r-8", query_type=QueryType.REPORT, message="Test")
    req = await run_receiver(payload)
    assert req.device_info is None


class TestExtractDeviceContext:
    def test_returns_empty_context_when_request_is_none(self):
        ctx = extract_device_context(None)
        assert ctx.device_id is None
        assert ctx.fcm_token is None

    def test_extracts_all_headers(self):
        mock_req = MagicMock()
        mock_req.headers = {
            "x-device-id": "dev-xyz",
            "x-fcm-token": "tok-xyz",
            "user-agent": "Mozilla",
        }
        ctx = extract_device_context(mock_req)
        assert ctx.device_id == "dev-xyz"
        assert ctx.fcm_token == "tok-xyz"
        assert ctx.user_agent == "Mozilla"
        assert ctx.platform == "web"

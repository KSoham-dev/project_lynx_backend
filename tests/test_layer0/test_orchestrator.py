"""
tests/test_layer0/test_orchestrator.py
=======================================
Unit tests for agents/layer0/orchestrator.py
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.enums import AgentLayer, MessageRole, QueryType
from agents.layer0.orchestrator import _session_to_lc_messages, run_orchestrator
from agents.layer0.receiver import AgentRequest
from agents.state.schemas import MessageRecord, SessionDocument
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_request(
    session_id: str = "sess-1",
    user_id: str = "u-1",
    message: str = "Is this leopard dangerous?",
    messages: list | None = None,
) -> AgentRequest:
    session = SessionDocument(
        id=session_id,
        session_id=session_id,
        user_id=user_id,
        query_type=QueryType.REPORT,
    )
    if messages:
        session.messages.extend(messages)
    return AgentRequest(
        user_id=user_id,
        session_id=session_id,
        query_type=QueryType.REPORT,
        message=message,
        session=session,
    )


# ── _session_to_lc_messages ───────────────────────────────────────────────────

class TestSessionToLcMessages:
    def test_no_history_returns_only_system(self):
        req = _make_request()
        msgs = _session_to_lc_messages(req)
        assert len(msgs) == 1
        assert isinstance(msgs[0], SystemMessage)

    def test_history_converted_correctly(self):
        prior = [
            MessageRecord(role=MessageRole.USER, content="I saw a big cat."),
            MessageRecord(role=MessageRole.ASSISTANT, content="Can you describe it?"),
        ]
        req = _make_request(messages=prior)
        msgs = _session_to_lc_messages(req)
        assert len(msgs) == 3
        assert isinstance(msgs[1], HumanMessage)
        assert isinstance(msgs[2], AIMessage)
        assert msgs[1].content == "I saw a big cat."


# ── run_orchestrator ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_orchestrator_returns_response():
    mock_response = AIMessage(content="Bengal Tiger. Keep distance.")
    mock_response.tool_calls = []

    mock_llm = MagicMock()
    mock_llm_with_tools = MagicMock()
    mock_llm_with_tools.ainvoke = AsyncMock(return_value=mock_response)
    mock_llm.bind_tools = MagicMock(return_value=mock_llm_with_tools)

    with (
        patch("agents.layer0.orchestrator._build_llm", return_value=mock_llm),
        patch("agents.layer0.orchestrator.append_message", new=AsyncMock()) as mock_append,
    ):
        req = _make_request()
        result = await run_orchestrator(req)

    assert result == "Bengal Tiger. Keep distance."
    assert mock_append.await_count == 2


@pytest.mark.asyncio
async def test_run_orchestrator_persists_with_enum_roles():
    """Verify append_message is called with MessageRole enum values."""
    mock_response = AIMessage(content="It's a lion.")
    mock_response.tool_calls = []

    mock_llm = MagicMock()
    mock_llm.bind_tools = MagicMock(return_value=MagicMock(
        ainvoke=AsyncMock(return_value=mock_response),
    ))

    calls = []

    async def capture_append(session_id, role, content, **kwargs):
        calls.append((role, content))

    with (
        patch("agents.layer0.orchestrator._build_llm", return_value=mock_llm),
        patch("agents.layer0.orchestrator.append_message", side_effect=capture_append),
    ):
        await run_orchestrator(_make_request(message="What animal is this?"))

    assert calls[0][0] == MessageRole.USER
    assert calls[1][0] == MessageRole.ASSISTANT


@pytest.mark.asyncio
async def test_run_orchestrator_includes_history():
    prior = [
        MessageRecord(role=MessageRole.USER, content="Earlier message"),
        MessageRecord(role=MessageRole.ASSISTANT, content="Earlier response"),
    ]

    captured_msgs = []
    mock_response = AIMessage(content="Leopard.")
    mock_response.tool_calls = []

    async def capture_ainvoke(messages, **kwargs):
        captured_msgs.extend(messages)
        return mock_response

    mock_llm = MagicMock()
    mock_llm.bind_tools = MagicMock(return_value=MagicMock(ainvoke=capture_ainvoke))

    with (
        patch("agents.layer0.orchestrator._build_llm", return_value=mock_llm),
        patch("agents.layer0.orchestrator.append_message", new=AsyncMock()),
    ):
        await run_orchestrator(_make_request(messages=prior))

    # system + 2 history + 1 current user = 4 messages
    assert len(captured_msgs) == 4
    assert isinstance(captured_msgs[0], SystemMessage)
    assert isinstance(captured_msgs[3], HumanMessage)

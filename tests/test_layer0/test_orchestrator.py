"""
Tests for agents/layer0/orchestrator.py

Tests the new single-LLM-call architecture:
  - One LLM call for tool selection (parallel_tool_calls=True)
  - asyncio.gather for concurrent tool execution
  - No final LLM text call — returns ("", tool_results)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.enums import MessageRole, QueryType
from agents.layer0.orchestrator import run_orchestrator
from agents.state.schemas import MessageRecord, SessionDocument


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_session(messages: list[MessageRecord] | None = None) -> SessionDocument:
    return SessionDocument(
        id="sess-1",
        session_id="sess-1",
        user_id="user-1",
        query_type=QueryType.REPORT,
        messages=messages or [],
    )


def _make_request(
    message: str = "I saw a tiger",
    messages: list[MessageRecord] | None = None,
    image_url: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
):
    from agents.layer0.receiver import AgentRequest

    return AgentRequest(
        user_id="user-1",
        session_id="sess-1",
        query_type=QueryType.REPORT,
        message=message,
        image_url=image_url,
        latitude=latitude,
        longitude=longitude,
        session=_make_session(messages),
    )


# ── run_orchestrator ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_orchestrator_no_tool_calls_returns_empty():
    """When LLM returns no tool_calls, orchestrator returns ('', {})."""
    mock_response = AIMessage(content="")
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
        response_text, tool_results = await run_orchestrator(req)

    assert response_text == ""
    assert tool_results == {}
    # Only user message is persisted — no assistant msg (no LLM text)
    assert mock_append.await_count == 1


@pytest.mark.asyncio
async def test_run_orchestrator_persists_user_message_only():
    """Only user message is persisted — no assistant message written."""
    mock_response = AIMessage(content="")
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

    assert len(calls) == 1
    assert calls[0][0] == MessageRole.USER
    assert calls[0][1] == "What animal is this?"


@pytest.mark.asyncio
async def test_run_orchestrator_includes_history():
    prior = [
        MessageRecord(role=MessageRole.USER, content="Earlier message"),
        MessageRecord(role=MessageRole.ASSISTANT, content="Earlier response"),
    ]

    captured_msgs = []
    mock_response = AIMessage(content="")
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


@pytest.mark.asyncio
async def test_context_block_injected_when_image_and_gps_present():
    """Image URL and GPS coordinates appear in the HumanMessage content block."""
    captured_msgs = []
    mock_response = AIMessage(content="")
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
        await run_orchestrator(_make_request(
            image_url="https://example.com/img.jpg",
            latitude=26.01,
            longitude=76.50,
        ))

    # System prompt is first, last message is the HumanMessage with context
    human_msg = next(m for m in captured_msgs if isinstance(m, HumanMessage))
    assert "Image URL: https://example.com/img.jpg" in human_msg.content
    # Python serialises 76.50 as 76.5 — check prefix only
    assert "GPS coordinates: latitude=26.01" in human_msg.content
    assert "longitude=76.5" in human_msg.content


@pytest.mark.asyncio
async def test_tool_results_parsed_from_json():
    """Tool results are JSON-parsed into the returned dict."""
    import json

    tool_output = json.dumps({"scientific_name": "Panthera tigris", "severity": "high"})

    tool_call = {"name": "analyse_text", "args": {"message": "tiger"}, "id": "tc-1"}
    mock_response = AIMessage(content="")
    mock_response.tool_calls = [tool_call]

    mock_llm = MagicMock()
    mock_llm.bind_tools = MagicMock(return_value=MagicMock(
        ainvoke=AsyncMock(return_value=mock_response),
    ))

    with (
        patch("agents.layer0.orchestrator._build_llm", return_value=mock_llm),
        patch("agents.layer0.orchestrator._run_tool_async", new=AsyncMock(
            return_value=MagicMock(content=tool_output)
        )),
        patch("agents.layer0.orchestrator.append_message", new=AsyncMock()),
    ):
        _, tool_results = await run_orchestrator(_make_request())

    assert tool_results["analyse_text"]["scientific_name"] == "Panthera tigris"
    assert tool_results["analyse_text"]["severity"] == "high"

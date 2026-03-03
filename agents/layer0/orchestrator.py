"""
agents/layer0/orchestrator.py
==============================
Orchestrator Agent — Layer 0 brain of the Prahari multi-agent system.

Architecture
------------
* LLM  : Azure OpenAI ``gpt-4o-mini`` via LangChain ``AzureChatOpenAI``
* Tools: Stub tools representing Layer 1 integrations (bound via ``bind_tools``)
* Loop : Tool-calling loop (LCEL pattern) — LLM decides which tools to call,
         we dispatch, then feed results back until the LLM issues a final answer.
* Memory: Prior session messages are prepended to every invocation from Cosmos DB.

Stub tools (wired to Layer 1 in a later sprint)
------------------------------------------------
``relevancy_check``   – determine if the user's message is wildlife-related
``location_context``  – extract / validate GPS / place context
``image_identifier``  – delegate to the SpeciesNet /predict endpoint
``context_text``      – retrieve IUCN / Wikipedia enrichment for a species

Public API
----------
    agent_request = await run_receiver(payload)
    response_text = await run_orchestrator(agent_request)
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI

from agents.enums import AgentLayer, MessageRole
from agents.layer0.config import get_settings
from agents.layer0.receiver import AgentRequest
from agents.state.session_store import append_message, get_session

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are Prahari, an intelligent wildlife assistant for general public, forest
officials, and conservation officers in India.

Your goals:
1. Help users identify wildlife from images, GPS coordinates, and text descriptions.
2. Provide species information (IUCN status, danger level, behaviour, habitat).
3. Assist with incident and sighting reports.
4. Always prioritise user safety.

Tool usage rules:
- Use ``analyse_image`` whenever the user provides an image URL.
  Always pass latitude and longitude if available — they improve accuracy.
- Use ``context_text`` after identification to get IUCN / trait details.
- Use ``location_context`` when the user asks about a specific area or GPS location.
- If a query is unrelated to wildlife, conservation, or safety, politely decline.
"""

# ── Layer 1 tool — real image analysis (replaces relevancy_check + image_identifier stubs)

@tool
async def analyse_image(
    image_url: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> str:
    """
    Detect and identify wildlife in an image.

    Runs the full Layer 1 pipeline:
      1. Detection-based relevancy check (is there an animal?).
      2. Two-stage species identification:
           - SpeciesNet label parsing if confidence >= 80% and not priority genus.
           - GPT-4o-mini vision fallback otherwise.
      3. Returns JSON with: is_relevant, scientific_name, confidence,
         identification_source (model|gpt), family, genus.

    Always pass latitude and longitude when available — SpeciesNet uses them
    to improve geo-constrained classification.
    """
    from agents.layer1.image_agent import get_image_agent
    agent = get_image_agent()
    result = await agent.run(image_url, latitude, longitude)
    return result.model_dump_json()


# ── Remaining Layer 1 stubs (context_text + location_context) ─────────────────

@tool
def location_context(latitude: float, longitude: float) -> str:
    """
    Given GPS coordinates, return the forest zone, district, and protected
    area name (if any).
    [STUB — Layer 1 Location agent will replace this implementation]
    """
    logger.debug("[STUB] location_context called: lat=%s lon=%s", latitude, longitude)
    return f"Location context for ({latitude}, {longitude}): [stub — to be implemented in Layer 1]"



@tool
def context_text(species_name: str) -> str:
    """
    Retrieve enriched species information: IUCN status, habitat, danger level,
    and Wikipedia summary.
    [STUB — Layer 1 Context/Text agent will replace this implementation]
    """
    logger.debug("[STUB] context_text called: %s", species_name)
    return f"Species context for '{species_name}': [stub — to be implemented in Layer 1]"


@tool
async def analyse_text(message: str) -> str:
    """
    Analyse a user's text message about a wildlife sighting or encounter.

    Returns a JSON object with three sections:
      - traits: extracted animal characteristics (size, colour, behaviour,
                distinctive features, count, movement)
      - severity: semantic/intent severity of the message
                  (critical | high | medium | low | informational)
      - species identification from text: scientific name, common name,
                  confidence — 'null' scientific_name means UNIDENTIFIED.

    Use this tool whenever the user sends a text description of an animal
    (with or without an image). Always call this before context_text so
    you have the species name to look up.
    """
    from agents.layer1.text_agent import get_text_agent
    agent = get_text_agent()
    result = await agent.run(message)
    return result.model_dump_json()


_TOOLS = [analyse_image, analyse_text, location_context, context_text]
_TOOLS_BY_NAME: dict[str, Any] = {t.name: t for t in _TOOLS}
_ASYNC_TOOLS: frozenset[str] = frozenset({"analyse_image", "analyse_text"})

# ── LLM factory ───────────────────────────────────────────────────────────────

def _build_llm() -> AzureChatOpenAI:
    """Construct the Azure OpenAI LLM client from settings."""
    s = get_settings()
    return AzureChatOpenAI(
        azure_endpoint=s.azure_openai_endpoint,
        api_key=s.azure_openai_api_key,          # type: ignore[arg-type]
        azure_deployment=s.azure_openai_deployment,
        api_version=s.azure_openai_api_version,
        max_tokens=1024
    )


# ── Cosmos DB–backed chat history (in-memory snapshot) ────────────────────────

def _session_to_lc_messages(request: AgentRequest) -> List[BaseMessage]:
    """
    Convert persisted :class:`MessageRecord` objects to LangChain message types.
    Prepended to every new invocation as conversation context.
    """
    lc_messages: List[BaseMessage] = [SystemMessage(content=_SYSTEM_PROMPT)]
    for record in request.session.messages:
        if record.role == "user":
            lc_messages.append(HumanMessage(content=record.content))
        elif record.role == "assistant":
            lc_messages.append(AIMessage(content=record.content))
        elif record.role == "system":
            lc_messages.append(SystemMessage(content=record.content))
    return lc_messages


# ── Tool execution helper ─────────────────────────────────────────────────────

def _run_tool(tool_call: dict) -> ToolMessage:
    """Invoke a synchronous tool call and return a ToolMessage."""
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_fn = _TOOLS_BY_NAME.get(tool_name)

    if tool_fn is None:
        result = f"Error: unknown tool '{tool_name}'"
    else:
        try:
            result = tool_fn.invoke(tool_args)
        except Exception as exc:  # noqa: BLE001
            result = f"Tool error: {exc}"

    return ToolMessage(
        content=str(result),
        tool_call_id=tool_call.get("id", tool_name),
    )


async def _run_tool_async(tool_call: dict) -> ToolMessage:
    """Invoke an async tool call and return a ToolMessage."""
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_fn = _TOOLS_BY_NAME.get(tool_name)

    if tool_fn is None:
        result = f"Error: unknown tool '{tool_name}'"
    else:
        try:
            result = await tool_fn.ainvoke(tool_args)
        except Exception as exc:  # noqa: BLE001
            result = f"Tool error: {exc}"

    return ToolMessage(
        content=str(result),
        tool_call_id=tool_call.get("id", tool_name),
    )


# ── Tool-calling loop ──────────────────────────────────────────────────────────

async def _run_tool_loop(
    llm_with_tools: AzureChatOpenAI,
    messages: List[BaseMessage],
    max_iterations: int = 6,
) -> str:
    """
    Execute a multi-turn tool-calling loop:
    1. Call LLM (with tools bound).
    2. If the response contains tool_calls, execute them and add ToolMessages.
    3. Repeat until the LLM returns a final text answer (no tool_calls).

    Returns the final text content.
    """
    for iteration in range(max_iterations):
        response: AIMessage = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        # If no tool calls in response, we have the final answer
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return response.content or ""

        logger.debug("Iteration %d: executing %d tool call(s)", iteration + 1, len(tool_calls))

        # Execute all tool calls — use async runner for async tools, sync for sync
        for tc in tool_calls:
            if tc["name"] in _ASYNC_TOOLS:
                tool_msg = await _run_tool_async(tc)
            else:
                tool_msg = _run_tool(tc)
            messages.append(tool_msg)

    # Safety fallback: return last content if loop exhausted
    last = messages[-1]
    return getattr(last, "content", "") or ""


# ── Public entry point ────────────────────────────────────────────────────────

async def run_orchestrator(request: AgentRequest) -> str:
    """
    Invoke the LangChain tool-calling orchestrator for one turn.

    Steps
    -----
    1. Build message list from system prompt + stored session history.
    2. Append the current user message.
    3. Run tool-calling loop (LLM → tools → LLM → … → final answer).
    4. Persist both the user message and AI response to Cosmos DB.
    5. Return the final response string.

    Parameters
    ----------
    request:
        Normalised agent request from :func:`agents.layer0.receiver.run_receiver`.

    Returns
    -------
    str
        The orchestrator's final text response to the user.
    """
    logger.info(
        "Orchestrator invoked: session=%s user=%s message_len=%d",
        request.session_id,
        request.user_id,
        len(request.message),
    )

    # 1 — build message history from session
    messages: List[BaseMessage] = _session_to_lc_messages(request)

    # 2 — append current user turn
    messages.append(HumanMessage(content=request.message))

    # 3 — build LLM with tools bound and run loop
    llm = _build_llm()
    llm_with_tools = llm.bind_tools(_TOOLS)  # type: ignore[arg-type]
    response_text = await _run_tool_loop(llm_with_tools, messages)

    # 4 — persist both turns to Cosmos DB
    await append_message(
        request.session_id,
        role=MessageRole.USER,
        content=request.message,
        layer=AgentLayer.LAYER0,
    )
    await append_message(
        request.session_id,
        role=MessageRole.ASSISTANT,
        content=response_text,
        layer=AgentLayer.LAYER0,
    )

    logger.info(
        "Orchestrator response: session=%s chars=%d",
        request.session_id,
        len(response_text),
    )
    return response_text

"""
agents/layer0/orchestrator.py
==============================
Orchestrator Agent — coordinates the full Prahari multi-agent pipeline.

Standard pipeline (EXPLORE / REPORT / ENCYCLOPEDIA)
------------------------------------------------------
1. Build messages (system prompt + session history + user context block).
2. FIRST LLM call — Layer 1 tool selection.
   Reads the user message + context, returns tool_calls specifying
   which Layer 1 tools to invoke (analyse_image, analyse_text,
   location_context) and with what arguments.

3. Layer 1 tools run CONCURRENTLY via asyncio.gather().
   analyse_image    → SpeciesNet + GPT vision
   analyse_text     → Groq trait/severity extraction
   location_context → Nominatim reverse geocode

4. SECOND LLM call — Layer 2 routing (fully traced in LangSmith).
   Consolidated Layer 1 results + query_type are sent to the LLM.
   Three no-arg tools are offered as closures over request + tool_results:
     generate_exploration_profile (EXPLORE)
     generate_incident_report     (REPORT)
     generate_encyclopedia_entry  (ENCYCLOPEDIA)
   tool_choice is forced to the tool matching the request's query_type,
   making routing deterministic while keeping full LangSmith trace coverage.

5. Layer 2 tool runs inside LangChain's tracing context:
   SpeciesInfoAgent (shared IUCN fetch) →
     ExplorerAgent | ReporterAgent | EncyclopediaAgent

Returns: ("", structured_data: dict)
  structured_data — final Layer 2 response dict, ready for the frontend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
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
from fastapi import HTTPException

from agents.enums import AgentLayer, MessageRole
from agents.layer0.config import get_settings
from agents.layer0.receiver import AgentRequest
from agents.state.session_store import append_message

logger = logging.getLogger(__name__)

# ── System prompt factory ─────────────────────────────────────────────────────

_TOOL_SELECTION_PROMPT = """\
You are a tool-selection router for Prahari, a wildlife assistant system.

Your ONLY job is to choose which tools to call based on the incoming request.
You MUST select ALL applicable tools simultaneously in your first (and ONLY) response.
Do NOT write any text — output ONLY tool calls.

SELECTION RULES — only call a tool when its required input is present:

  Call analyse_text(message) ONLY when [Context] shows:
    Text message: (something other than [not provided])
    — NEVER call it when Text message is [not provided].

  Call analyse_image(image_url, latitude, longitude) ONLY when [Context] shows:
    Image URL: (something other than [not provided])
    — Pass latitude/longitude from Context if they are not [not provided].

  Call location_context(latitude, longitude) ONLY when [Context] shows:
    GPS coordinates: (something other than [not provided])
    — Pass the exact values given.

If a field shows "[not provided]" in [Context], do NOT call the tool that requires it.
For ENCYCLOPEDIA queries: call analyse_text only (no image or location needed).
"""


def _build_system_prompt(query_type: str) -> str:
    """Return the tool-selection system prompt (same for all query types)."""
    return _TOOL_SELECTION_PROMPT


# ── Tool definitions ──────────────────────────────────────────────────────────

@tool
async def analyse_image(
    image_url: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> str:
    """
    Detect and identify wildlife in an image.

    Runs SpeciesNet → relevancy check → two-stage species identification
    (SpeciesNet model, then GPT vision fallback for priority genera).

    Returns JSON: is_relevant, scientific_name, confidence,
    identification_source (model|gpt), family, genus.

    Call this whenever an Image URL is present in the request context.
    """
    from agents.layer1.image_agent import get_image_agent
    logger.info("[analyse_image] START url=%.60s lat=%s lon=%s", image_url, latitude, longitude)
    t0 = time.monotonic()
    agent = get_image_agent()
    result = await agent.run(image_url, latitude, longitude)
    logger.info("[analyse_image] DONE in %.2fs: relevant=%s species=%s",
                time.monotonic() - t0, result.is_relevant, result.scientific_name)
    return result.model_dump_json()


@tool
async def analyse_text(message: str) -> str:
    """
    Analyse a user's text message about a wildlife sighting or encounter.

    Returns JSON with:
      - traits: size, colour, behaviour, movement, distinctive_features, count
      - severity: critical | high | medium | low | informational
      - scientific_name, common_name, identification_confidence
        (scientific_name is null if species cannot be identified from text alone)

    Always call this for any wildlife-related user message.
    """
    from agents.layer1.text_agent import get_text_agent
    logger.info("[analyse_text] START: message_len=%d", len(message))
    t0 = time.monotonic()
    agent = get_text_agent()
    result = await agent.run(message)
    logger.info("[analyse_text] DONE in %.2fs: severity=%s species=%s confidence=%s",
                time.monotonic() - t0, result.severity.value,
                result.scientific_name or "UNIDENTIFIED", result.identification_confidence)
    return result.model_dump_json()


@tool
async def location_context(latitude: float, longitude: float) -> str:
    """
    Reverse-geocode GPS coordinates into district, state, protected area.

    Returns JSON: district, state, country, protected_area, geohash, formatted.
    Call this whenever GPS coordinates are present in the request context.
    """
    from agents.layer1.location import get_location_context
    logger.info("[location_context] START: lat=%s lon=%s", latitude, longitude)
    t0 = time.monotonic()
    result = await get_location_context(latitude, longitude)
    logger.info("[location_context] DONE in %.2fs: district=%r state=%r",
                time.monotonic() - t0, result.get("district"), result.get("state"))
    return json.dumps(result)


# ── Tool registry ─────────────────────────────────────────────────────────────

_TOOLS = [analyse_image, analyse_text, location_context]
_TOOLS_BY_NAME: dict[str, Any] = {t.name: t for t in _TOOLS}


def _available_tools(request: AgentRequest) -> list:
    """
    Return only the tools applicable to the current request payload.

    This prevents the LLM from even seeing — and therefore accidentally
    calling — tools whose required inputs are absent.  The mapping is:

        analyse_text       ← message is non-empty
        analyse_image      ← image_url is present
        location_context   ← both latitude AND longitude are present
    """
    tools = []
    if request.message:
        tools.append(analyse_text)
    if request.image_url:
        tools.append(analyse_image)
    if request.latitude is not None and request.longitude is not None:
        tools.append(location_context)
    if not tools:
        logger.warning(
            "[orchestrator] No applicable tools — payload has no message, image, or GPS."
        )
    return tools


# ── LLM factory ───────────────────────────────────────────────────────────────

def _build_llm() -> AzureChatOpenAI:
    """
    Construct the Azure LLM client.

    max_tokens=1024 — enough room for 3 tool call JSONs with headroom to spare.
    """
    s = get_settings()
    return AzureChatOpenAI(
        azure_endpoint=s.azure_openai_endpoint,
        api_key=s.azure_openai_api_key,          # type: ignore[arg-type]
        azure_deployment=s.azure_openai_deployment,
        api_version=s.azure_openai_api_version,
        max_tokens=1024,
    )


# ── Session history → LangChain messages ──────────────────────────────────────

def _session_to_lc_messages(request: AgentRequest) -> List[BaseMessage]:
    """
    Build the message list: system prompt + session history.
    Uses the same tool-selection prompt regardless of query type.
    """
    lc_messages: List[BaseMessage] = [
        SystemMessage(content=_build_system_prompt(
            request.query_type.value if request.query_type else "report"
        ))
    ]
    for record in request.session.messages:
        if record.role == "user":
            lc_messages.append(HumanMessage(content=record.content))
        elif record.role == "assistant":
            lc_messages.append(AIMessage(content=record.content))
    return lc_messages


# ── Concurrent tool runner ────────────────────────────────────────────────────

async def _run_tool_async(tool_call: dict) -> ToolMessage:
    """Invoke an async tool and return a ToolMessage."""
    tool_name = tool_call["name"]
    tool_fn = _TOOLS_BY_NAME.get(tool_name)

    if tool_fn is None:
        result = f"Error: unknown tool '{tool_name}'"
    else:
        try:
            result = await tool_fn.ainvoke(tool_call["args"])
        except Exception as exc:
            logger.error("[orchestrator] Tool %r failed: %s", tool_name, exc, exc_info=True)
            result = f"Tool error: {exc}"

    return ToolMessage(
        content=str(result),
        tool_call_id=tool_call.get("id", tool_name),
    )


# ── Public entry point ────────────────────────────────────────────────────────

async def run_orchestrator(request: AgentRequest) -> tuple[str, dict[str, Any]]:
    """
    Full pipeline orchestrator: L1 tool selection → L1 tools → L2 routing → L2 agent.

    Steps
    -----
    1. Build messages (system prompt + session history + user context block).
    2. FIRST LLM call — selects Layer 1 tools (tool_choice="required").
    3. Layer 1 tools run concurrently via asyncio.gather().
    4. Persist user message to Cosmos.
    5. SECOND LLM call — routes to the appropriate Layer 2 tool (LangSmith-traced).
       Layer 2 tool (EXPLORE / REPORT / ENCYCLOPEDIA) runs as a LangChain StructuredTool closure.

    Returns
    -------
    ("", structured_data)
        structured_data — final Layer 2 response dict, ready for the frontend.
    """
    t_start = time.monotonic()
    logger.info(
        "[orchestrator] START session=%s user=%s query_type=%s image=%s gps=%s",
        request.session_id, request.user_id, request.query_type,
        bool(request.image_url),
        bool(request.latitude is not None),
    )

    # ── 1. Build message list ────────────────────────────────────────────────
    messages = _session_to_lc_messages(request)

    # Build an explicit [Context] block so absent fields are clearly labelled "[not provided]".
    # The LLM reads these labels to decide which tools to call; _available_tools() enforces
    # the same rules programmatically so absent-input tools are never even offered.
    context_lines: list[str] = [
        f"Text message: {request.message!r}" if request.message else "Text message: [not provided]",
        f"Image URL: {request.image_url}" if request.image_url else "Image URL: [not provided]",
        (
            f"GPS coordinates: latitude={request.latitude}, longitude={request.longitude}"
            if request.latitude is not None and request.longitude is not None
            else "GPS coordinates: [not provided]"
        ),
    ]
    user_content = (request.message or "[no message]") + "\n\n[Context]\n" + "\n".join(context_lines)
    messages.append(HumanMessage(content=user_content))

    # ── 2. Single LLM call — tool selection only ─────────────────────────────
    applicable_tools = _available_tools(request)
    if not applicable_tools:
        logger.error("[orchestrator] No inputs present — cannot route request.")
        return "", {}

    llm = _build_llm()
    # tool_choice="required" forces the API to guarantee at least one tool call
    # is returned, eliminating silent hallucination-as-text responses.
    # parallel_tool_calls=True allows multiple tools to be selected in one response.
    llm_with_tools = llm.bind_tools(applicable_tools, tool_choice="required", parallel_tool_calls=True)

    logger.info("[orchestrator] LLM call (tool selection)")
    t_llm = time.monotonic()
    response: AIMessage = await llm_with_tools.ainvoke(messages)
    logger.info("[orchestrator] LLM responded in %.2fs", time.monotonic() - t_llm)

    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        # ── Diagnostics: log what the model actually returned ────────────────
        finish_reason = (
            (response.response_metadata or {}).get("finish_reason")
            or (response.response_metadata or {}).get("stop_reason")
            or "unknown"
        )
        content_preview = (response.content or "")[:300].replace("\n", " ")
        logger.warning(
            "[orchestrator] LLM returned NO tool calls on first attempt — retrying. "
            "finish_reason=%r content=%r",
            finish_reason, content_preview,
        )

        # ── Single retry: inject a hard correction turn ───────────────────────
        retry_messages = messages + [
            AIMessage(content=response.content or ""),
            HumanMessage(
                content=(
                    "IMPORTANT: You must call the appropriate tools. "
                    "Do NOT write any text response. "
                    "Call the tools now based on the context provided above."
                )
            ),
        ]
        logger.info("[orchestrator] LLM retry call (tool selection)")
        t_retry = time.monotonic()
        response = await llm_with_tools.ainvoke(retry_messages)
        logger.info("[orchestrator] LLM retry responded in %.2fs", time.monotonic() - t_retry)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            finish_reason_retry = (
                (response.response_metadata or {}).get("finish_reason")
                or (response.response_metadata or {}).get("stop_reason")
                or "unknown"
            )
            content_preview_retry = (response.content or "")[:300].replace("\n", " ")
            logger.error(
                "[orchestrator] LLM returned NO tool calls after retry — aborting. "
                "finish_reason=%r content=%r "
                "Check LangSmith trace for tool definitions in request.",
                finish_reason_retry, content_preview_retry,
            )
            await append_message(
                request.session_id, role=MessageRole.USER,
                content=request.message or "", layer=AgentLayer.LAYER0,
            )
            return "", {}

    logger.info(
        "[orchestrator] Tools selected: %s",
        [tc["name"] for tc in tool_calls],
    )
    for tc in tool_calls:
        logger.info(
            "[orchestrator] ▶ %r args=%s",
            tc["name"],
            {k: str(v)[:80] for k, v in tc.get("args", {}).items()},
        )

    # ── 3. Run ALL tools concurrently ────────────────────────────────────────
    t_tools = time.monotonic()
    tool_msgs: list[ToolMessage] = await asyncio.gather(
        *[_run_tool_async(tc) for tc in tool_calls]
    )
    logger.info("[orchestrator] All tools done in %.2fs", time.monotonic() - t_tools)

    # Parse each tool result
    tool_results: dict[str, Any] = {}
    for tc, msg in zip(tool_calls, tool_msgs):
        preview = str(msg.content)[:200].replace("\n", " ")
        logger.info("[orchestrator] ◀ %r: %s…", tc["name"], preview)
        try:
            tool_results[tc["name"]] = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            tool_results[tc["name"]] = msg.content

    # ── 4. Persist user message ───────────────────────────────────────────────
    await append_message(
        request.session_id, role=MessageRole.USER,
        content=request.message or "", layer=AgentLayer.LAYER0,
    )

    logger.info(
        "[orchestrator] Layer 1 done in %.2fs total | tools=%s",
        time.monotonic() - t_start,
        list(tool_results.keys()),
    )

    # ── is_relevant gate ──────────────────────────────────────────────────────
    # If the image analysis ran and flagged the image as non-relevant, AND
    # the user provided no text message to fall back on, skip Layer 2 entirely
    # and return a friendly retry prompt.
    image_result: dict = tool_results.get("analyse_image") or {}
    if isinstance(image_result, dict) and image_result.get("is_relevant") is False and not request.message:
        logger.info(
            "[orchestrator] Image not relevant and no message — skipping Layer 2. "
            "session=%s", request.session_id,
        )
        # Delete the irrelevant image from blob storage so it doesn't accumulate
        if request.image_url:
            asyncio.create_task(_delete_irrelevant_image(request.image_url))
        return "", {
            "message": (
                "We couldn't detect any animal in your image. "
                "Please retake the photo with the animal clearly visible and try again."
            )
        }

    # ── 5. Layer 2 — direct dispatch based on query_type (no LLM call needed) ─
    structured_data = await _direct_dispatch_layer2(request, tool_results)

    logger.info(
        "[orchestrator] Pipeline complete in %.2fs",
        time.monotonic() - t_start,
    )
    return "", structured_data


async def _delete_irrelevant_image(image_url: str) -> None:
    """
    Delete a blob by URL when the image is declared irrelevant.

    Parses the container and blob name from the URL, then issues a
    synchronous delete_blob() in a thread.  Failures are logged and
    swallowed — they must never surface to the caller.
    """
    try:
        from urllib.parse import urlparse
        parsed    = urlparse(image_url)
        # Path is /<container>/<blob_name_possibly_with_slashes>
        path_parts = parsed.path.lstrip("/").split("/", 1)
        if len(path_parts) != 2:
            logger.warning(
                "[orchestrator] Cannot parse blob URL for deletion: %s", image_url
            )
            return
        container_name, blob_name = path_parts

        def _delete_sync() -> None:
            from pipeline.species_traits import _build_blob_service_client
            service = _build_blob_service_client()
            blob_client = service.get_blob_client(container=container_name, blob=blob_name)
            blob_client.delete_blob(delete_snapshots="include")
            logger.info(
                "[orchestrator] Irrelevant image deleted: container=%s blob=%s",
                container_name, blob_name,
            )

        await asyncio.to_thread(_delete_sync)
    except Exception as exc:
        logger.warning(
            "[orchestrator] Failed to delete irrelevant image %s: %s", image_url, exc
        )


async def _direct_dispatch_layer2(
    request: AgentRequest,
    tool_results: dict[str, Any],
) -> dict[str, Any]:
    """
    Fallback: dispatch Layer 2 without a second LLM call.
    Used when the routing LLM fails or returns no tool calls.
    """
    logger.info("[orchestrator] Direct Layer 2 dispatch query_type=%s", request.query_type)
    try:
        from agents.layer2.species_info import SpeciesInfoAgent
        iucn_data = await SpeciesInfoAgent().run(request, tool_results)

        from agents.enums import QueryType as QT
        if request.query_type == QT.ENCYCLOPEDIA:
            from agents.layer2.encyclopedia import EncyclopediaAgent
            return await EncyclopediaAgent().run(request, tool_results, iucn_data) or {}

        if request.query_type == QT.EXPLORE:
            from agents.layer2.explorer import ExplorerAgent
            return await ExplorerAgent().run(request, tool_results, iucn_data) or {}

        from agents.layer2.reporter import ReporterAgent
        return await ReporterAgent().run(request, tool_results, iucn_data) or {}

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[orchestrator] Direct Layer 2 dispatch failed: %s", exc, exc_info=True)
        return {}



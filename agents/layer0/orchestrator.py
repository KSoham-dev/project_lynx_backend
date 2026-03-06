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


_LAYER2_ROUTING_PROMPT = """\
You are a Layer 2 routing agent for Prahari, a wildlife management system.

The Layer 1 analysis is complete. You will be given the consolidated results and
the requested query type. You MUST select and call EXACTLY ONE Layer 2 tool.

Tool selection rules (strictly follow query_type):
  EXPLORE      →  call generate_exploration_profile
  REPORT       →  call generate_incident_report
  ENCYCLOPEDIA →  call generate_encyclopedia_entry

Do NOT write any text — output ONLY the tool call.
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
        return "", {
            "message": (
                "We couldn't detect any animal in your image. "
                "Please retake the photo with the animal clearly visible and try again."
            )
        }

    # ── 5. Layer 2 — LLM-routed via second call (traced in LangSmith) ────────
    structured_data = await _route_layer2_via_llm(llm, request, tool_results)

    logger.info(
        "[orchestrator] Pipeline complete in %.2fs",
        time.monotonic() - t_start,
    )
    return "", structured_data


# ── Layer 2: LLM routing (second call) ───────────────────────────────────────

async def _route_layer2_via_llm(
    llm: AzureChatOpenAI,
    request: AgentRequest,
    tool_results: dict[str, Any],
) -> dict[str, Any]:
    """
    Second LLM call — routes to the appropriate Layer 2 agent
    (EXPLORE, REPORT, or ENCYCLOPEDIA).

    Builds three no-argument tool closures that close over the current
    ``request`` and ``tool_results``, so LangChain/LangSmith traces both
    the routing LLM call and the Layer 2 tool execution.

    Falls back to direct dispatch if the LLM fails to return a tool call.
    """
    from langchain_core.tools import StructuredTool

    from agents.enums import QueryType as QT
    from agents.layer2.encyclopedia import EncyclopediaAgent
    from agents.layer2.explorer import ExplorerAgent
    from agents.layer2.reporter import ReporterAgent
    from agents.layer2.species_info import SpeciesInfoAgent

    # ── Tool closures (capture request + tool_results) ────────────────────
    async def _explore() -> str:
        iucn = await SpeciesInfoAgent().run(request, tool_results)
        result = await ExplorerAgent().run(request, tool_results, iucn)
        return json.dumps(result, default=str)

    async def _report() -> str:
        iucn = await SpeciesInfoAgent().run(request, tool_results)
        result = await ReporterAgent().run(request, tool_results, iucn)
        return json.dumps(result, default=str)

    async def _encyclopedia() -> str:
        iucn = await SpeciesInfoAgent().run(request, tool_results)
        result = await EncyclopediaAgent().run(request, tool_results, iucn)
        return json.dumps(result, default=str)

    exp_tool = StructuredTool.from_function(
        coroutine=_explore,
        name="generate_exploration_profile",
        description=(
            "Generate an in-depth species exploration profile with traits, "
            "habitat information, and photographs."
        ),
    )
    rep_tool = StructuredTool.from_function(
        coroutine=_report,
        name="generate_incident_report",
        description=(
            "Generate an official wildlife incident report PDF and save "
            "it to Cosmos DB."
        ),
    )
    enc_tool = StructuredTool.from_function(
        coroutine=_encyclopedia,
        name="generate_encyclopedia_entry",
        description=(
            "Return the raw IUCN data and iNaturalist photo for a species "
            "identified in the Layer 1 analysis — no additional LLM calls."
        ),
    )

    _TOOL_MAP: dict = {
        QT.EXPLORE:      ("generate_exploration_profile", exp_tool),
        QT.REPORT:       ("generate_incident_report",     rep_tool),
        QT.ENCYCLOPEDIA: ("generate_encyclopedia_entry",  enc_tool),
    }
    tool_name, l2_tool = _TOOL_MAP.get(
        request.query_type, ("generate_incident_report", rep_tool)
    )
    all_l2_tools = [exp_tool, rep_tool, enc_tool]

    # ── Build Layer 1 summary for LLM context ────────────────────────────
    l1_summary = json.dumps(tool_results, indent=2, default=str)
    layer2_msgs: List[BaseMessage] = [
        SystemMessage(content=_LAYER2_ROUTING_PROMPT),
        HumanMessage(content=(
            f"Query type: "
            f"{request.query_type.value if request.query_type else 'report'}\n\n"
            f"Consolidated Layer 1 analysis results:\n{l1_summary}\n\n"
            "Call the appropriate Layer 2 tool now."
        )),
    ]

    # ── Second LLM call ───────────────────────────────────────────────────
    llm_l2 = llm.bind_tools(all_l2_tools, tool_choice=tool_name, parallel_tool_calls=False)
    logger.info("[orchestrator] Layer 2 LLM call — routing to tool %r", tool_name)
    t_l2 = time.monotonic()
    try:
        l2_response: AIMessage = await llm_l2.ainvoke(layer2_msgs)
    except Exception as exc:
        logger.error("[orchestrator] Layer 2 LLM call failed: %s", exc, exc_info=True)
        return await _direct_dispatch_layer2(request, tool_results)

    logger.info("[orchestrator] Layer 2 LLM responded in %.2fs", time.monotonic() - t_l2)

    l2_tool_calls = getattr(l2_response, "tool_calls", [])
    if not l2_tool_calls:
        logger.warning(
            "[orchestrator] Layer 2 LLM returned no tool calls — falling back to direct dispatch"
        )
        return await _direct_dispatch_layer2(request, tool_results)

    l2_tc = l2_tool_calls[0]
    logger.info("[orchestrator] Layer 2 tool selected: %r", l2_tc["name"])

    # ── Run Layer 2 tool (traced by LangChain) ────────────────────────────
    t_l2_tool = time.monotonic()
    try:
        l2_result_raw = await l2_tool.ainvoke(l2_tc.get("args") or {})
    except Exception as exc:
        logger.error("[orchestrator] Layer 2 tool %r failed: %s", l2_tc["name"], exc, exc_info=True)
        # Still close the LangSmith loop so the span isn't left open
        _tool_err_msg = ToolMessage(
            content=f"Tool failed: {exc}",
            tool_call_id=l2_tc.get("id", ""),
            name=l2_tc["name"],
        )
        try:
            await llm.ainvoke(layer2_msgs + [l2_response, _tool_err_msg])
        except Exception:
            pass
        return {}

    logger.info("[orchestrator] Layer 2 tool done in %.2fs", time.monotonic() - t_l2_tool)

    # ── Close the agent loop so LangSmith marks the tool call completed ───
    # Append the AIMessage (which contains the tool_call) and the ToolMessage
    # (which carries the result) back to the LLM.  This is the standard
    # LangChain agent pattern — without it LangSmith never receives the "end"
    # event for the tool span, so the trace shows as still running.
    tool_close_msg = ToolMessage(
        content=(l2_result_raw or "")[:500],   # truncated — we only need to close the span
        tool_call_id=l2_tc.get("id", ""),
        name=l2_tc["name"],
    )
    try:
        await llm.ainvoke(layer2_msgs + [l2_response, tool_close_msg])
    except Exception as close_exc:
        logger.warning("[orchestrator] Loop-close LLM call failed (non-fatal): %s", close_exc)

    try:
        return json.loads(l2_result_raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("[orchestrator] Layer 2 result non-JSON: %.200s", str(l2_result_raw)[:200])
        return {}


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

    except Exception as exc:
        logger.error("[orchestrator] Direct Layer 2 dispatch failed: %s", exc, exc_info=True)
        return {}



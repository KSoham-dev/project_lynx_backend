"""
agents/layer1/text_agent.py
============================
Layer 1 — TextAnalysisAgent

Analyses the user's natural-language message about a wildlife encounter
and returns three structured outputs in a single GPT call:

  a. Animal trait extraction
       – size, colour, behaviour, distinctive features
  b. Semantic / intent severity
       – CRITICAL → HIGH → MEDIUM → LOW → INFORMATIONAL
  c. Text-based species identification
       – best-guess scientific name, or "UNIDENTIFIED"

Uses GPT-4o-mini (same model as the rest of Layer 0/1) with a strict
JSON response format.

Public API
----------
    from agents.layer1.text_agent import get_text_agent
    result: TextAnalysisResult = await get_text_agent().run("I saw a big cat...")
"""
from __future__ import annotations

import json
import logging
from typing import Optional, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from agents.enums import SeverityLevel
from agents.layer0.config import get_settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Schema ───────────────────────────────────────────────────────────────────

from pydantic import BaseModel


class TextAnalysisResult(BaseModel):
    """Simplified output of the TextAnalysisAgent — 5 fields only."""
    size:                    Optional[str] = None   # e.g. "large", "cub-sized", null
    severity:                SeverityLevel          # critical | high | medium | low | informational
    scientific_name:         Optional[str] = None   # e.g. "Panthera tigris" or null
    common_name:             Optional[str] = None   # e.g. "Bengal Tiger" or null
    identification_confidence: int = 0              # 0–100 percent


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a wildlife analyst for Prahari (India). Analyse the user's wildlife message.
Return ONLY this JSON object — no extra text:

{
  "size": "<animal size in one word: tiny|small|medium|large|massive — or null>",
  "severity": "<critical|high|medium|low|informational>",
  "scientific_name": "<Genus species — or null if unsure>",
  "common_name": "<English name — or null>",
  "identification_confidence": <integer 0-100>
}

Severity: critical=immediate danger, high=active threat, medium=notable sighting,
          low=distant/calm sighting, informational=no wildlife interaction.
Species: use text clues only. Null if description is too vague. Never hallucinate.
Confidence: 90-100 only if certain, 50-89 if reasonable guess, 0-49 if vague.\
"""


# ── Agent ─────────────────────────────────────────────────────────────────────

class TextAnalysisAgent:
    """
    Analyses wildlife-related user text with a single GPT-4o-mini call.

    Returns traits, severity, and text-based species identification.
    """

    def __init__(self, llm: AzureChatOpenAI) -> None:
        self._llm = llm

    async def run(self, message: str) -> TextAnalysisResult:
        """
        Analyse a user message.

        Parameters
        ----------
        message:
            The user's natural-language description of a wildlife sighting
            or encounter.

        Returns
        -------
        TextAnalysisResult
        """
        logger.info("[analyse_text] START: message_len=%d", len(message))

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Analyse this message:\n\n{message}"),
        ]

        try:
            response = await self._llm.ainvoke(messages)
            raw = (response.content or "").strip()

            finish_reason = (
                (response.response_metadata or {}).get("finish_reason")
                or (response.response_metadata or {}).get("stop_reason")
                or "unknown"
            )
            logger.info(
                "[analyse_text] LLM finish_reason=%r raw_len=%d preview=%.200s",
                finish_reason, len(raw), raw or "<EMPTY>",
            )

            if not raw:
                raise ValueError(
                    f"LLM returned empty response — finish_reason={finish_reason!r}. "
                    "Check Azure content filter or deployment config."
                )

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            data: dict = json.loads(raw)

            severity_raw = (data.get("severity") or "low").lower()
            try:
                severity = SeverityLevel(severity_raw)
            except ValueError:
                logger.warning("[analyse_text] Unknown severity %r — defaulting to LOW", severity_raw)
                severity = SeverityLevel.LOW

            # identification_confidence: accept int or string ("high"→85, "medium"→50, "low"→20)
            raw_conf = data.get("identification_confidence", 0)
            if isinstance(raw_conf, int):
                confidence_int = max(0, min(100, raw_conf))
            elif isinstance(raw_conf, float):
                confidence_int = max(0, min(100, int(raw_conf)))
            elif isinstance(raw_conf, str):
                confidence_int = {"high": 85, "medium": 50, "low": 20}.get(raw_conf.lower(), 0)
            else:
                confidence_int = 0

            result = TextAnalysisResult(
                size=data.get("size") or None,
                severity=severity,
                scientific_name=data.get("scientific_name") or None,
                common_name=data.get("common_name") or None,
                identification_confidence=confidence_int,
            )

            logger.info(
                "[analyse_text] DONE: severity=%s species=%s confidence=%d%%",
                result.severity.value,
                result.scientific_name or "UNIDENTIFIED",
                result.identification_confidence,
            )
            return result

        except Exception as exc:
            logger.error("[analyse_text] FAILED: %s | raw=%r", exc, locals().get("raw", "<not set>"))
            return TextAnalysisResult(
                size=None,
                severity=SeverityLevel.LOW,
                scientific_name=None,
                identification_confidence=0,
            )


# ── Singleton factory ─────────────────────────────────────────────────────────

_text_agent_instance: Optional[TextAnalysisAgent] = None


def get_text_agent() -> TextAnalysisAgent:
    """Return a singleton TextAnalysisAgent, constructing it on first call."""
    global _text_agent_instance  # noqa: PLW0603
    if _text_agent_instance is None:
        s = get_settings()
        llm = AzureChatOpenAI(
            azure_endpoint=s.azure_openai_endpoint,
            api_key=s.azure_openai_api_key,         # type: ignore[arg-type]
            azure_deployment=s.azure_openai_deployment,
            api_version=s.azure_openai_api_version,
            max_tokens=1024,
            # temperature=0 not supported by GPT-5-mini (only default=1 allowed)
            model_kwargs={
                "response_format": {"type": "json_object"},
            },
        )
        _text_agent_instance = TextAnalysisAgent(llm=llm)
    return _text_agent_instance

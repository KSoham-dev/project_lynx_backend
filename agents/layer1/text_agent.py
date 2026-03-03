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

# ── Schema ────────────────────────────────────────────────────────────────────

from pydantic import BaseModel, Field


class AnimalTraits(BaseModel):
    """Traits of the animal extracted from the user's text."""
    size: Optional[str] = None          # "large", "small", "cub-sized"
    colour: Optional[str] = None         # "tawny with black spots"
    behaviour: Optional[str] = None      # "aggressive", "feeding", "resting"
    movement: Optional[str] = None       # "charging", "stationary", "fleeing"
    distinctive_features: list[str] = Field(default_factory=list)  # ["mane", "stripes", "trunk"]
    count: Optional[str] = None         # "single animal", "pair", "herd of ~20"
    additional: Optional[str] = None    # any other relevant details


class TextAnalysisResult(BaseModel):
    """Complete output of the TextAnalysisAgent."""
    # (a) Traits
    traits: AnimalTraits

    # (b) Severity
    severity: SeverityLevel
    severity_reason: str   # one-sentence explanation of the severity classification

    # (c) Species identification from text
    scientific_name: Optional[str] = None     # e.g. "Panthera tigris" — None if unidentified
    common_name: Optional[str] = None         # e.g. "Bengal Tiger"
    identification_confidence: str = "low"   # "high" | "medium" | "low"
    identification_note: str = ""            # brief reason for the identification or failure


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert wildlife analyst for Prahari — a wildlife sighting and safety system in India.

A user has sent a text message about a wildlife encounter or sighting. Your task is to analyse
the message and return a single JSON object with three sections.

━━━━━━━━━━━━━━━━━━━━━━━ OUTPUT FORMAT (strict JSON) ━━━━━━━━━━━━━━━━━━━━━━━
{
  "traits": {
    "size": "<string or null — e.g. 'large', 'cub-sized', 'enormous'>",
    "colour": "<string or null — e.g. 'tawny with black rosettes'>",
    "behaviour": "<string or null — e.g. 'aggressive', 'feeding on prey', 'resting'>",
    "movement": "<string or null — e.g. 'charging', 'stationary', 'swimming'>",
    "distinctive_features": ["<feature1>", "<feature2>"],
    "count": "<string or null — e.g. 'single animal', 'pair', 'herd of ~15'>",
    "additional": "<string or null — any other relevant details>"
  },
  "severity": "<exactly one of: critical | high | medium | low | informational>",
  "severity_reason": "<one sentence explaining why this severity was chosen>",
  "scientific_name": "<Genus species, e.g. 'Panthera tigris', or null if cannot identify>",
  "common_name": "<English common name, or null>",
  "identification_confidence": "<exactly one of: high | medium | low>",
  "identification_note": "<one sentence — reason for identification or why it was not possible>"
}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEVERITY GUIDE:
  critical     – User or others in immediate danger right now
  high         – Active threat nearby, could escalate quickly
  medium       – Notable sighting, no immediate danger
  low          – Indirect observation, distant sighting, low concern
  informational – General question or no wildlife interaction

SPECIES IDENTIFICATION RULES:
  - Use only information from the text — do NOT assume species from location alone.
  - If the user describes enough traits (e.g. "orange with black stripes"), give your best guess.
  - If the description is too vague (e.g. "I saw a bird"), set scientific_name to null.
  - NEVER hallucinate a species. Prefer null over an incorrect identification.
  - identification_confidence: "high" only if you are very certain.

Return ONLY the JSON object. No preamble, no explanation, no markdown fences.\
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
        logger.info("TextAnalysisAgent.run: message_len=%d", len(message))

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Analyse this message:\n\n{message}"),
        ]

        try:
            response = await self._llm.ainvoke(messages)
            raw = response.content.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            data: dict = json.loads(raw)

            traits = AnimalTraits(
                size=data.get("traits", {}).get("size"),
                colour=data.get("traits", {}).get("colour"),
                behaviour=data.get("traits", {}).get("behaviour"),
                movement=data.get("traits", {}).get("movement"),
                distinctive_features=data.get("traits", {}).get("distinctive_features") or [],
                count=data.get("traits", {}).get("count"),
                additional=data.get("traits", {}).get("additional"),
            )

            severity_raw = (data.get("severity") or "low").lower()
            try:
                severity = SeverityLevel(severity_raw)
            except ValueError:
                severity = SeverityLevel.LOW

            # scientific_name: null in JSON → None → "UNIDENTIFIED" string for LLM readability
            sci_name = data.get("scientific_name") or None

            result = TextAnalysisResult(
                traits=traits,
                severity=severity,
                severity_reason=data.get("severity_reason", ""),
                scientific_name=sci_name,
                common_name=data.get("common_name") or None,
                identification_confidence=data.get("identification_confidence", "low"),
                identification_note=data.get("identification_note", ""),
            )

            logger.info(
                "Text analysis: severity=%s species=%s confidence=%s",
                result.severity,
                result.scientific_name or "UNIDENTIFIED",
                result.identification_confidence,
            )
            return result

        except Exception as exc:
            logger.error("TextAnalysisAgent failed: %s", exc)
            # Return a safe fallback result
            return TextAnalysisResult(
                traits=AnimalTraits(),
                severity=SeverityLevel.LOW,
                severity_reason="Analysis failed — defaulting to LOW severity.",
                scientific_name=None,
                identification_note=f"Analysis error: {exc}",
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
            max_tokens=512,
        )
        _text_agent_instance = TextAnalysisAgent(llm=llm)
    return _text_agent_instance

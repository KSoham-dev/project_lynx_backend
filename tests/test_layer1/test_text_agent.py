"""
tests/test_layer1/test_text_agent.py
======================================
Unit tests for agents/layer1/text_agent.py

Covers:
  - AnimalTraits construction
  - TextAnalysisResult structure
  - SeverityLevel enum values
  - TextAnalysisAgent.run() — success path with mocked LLM
  - TextAnalysisAgent.run() — error / malformed JSON fallback
  - Species identified vs UNIDENTIFIED paths
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.enums import SeverityLevel
from agents.layer1.text_agent import (
    AnimalTraits,
    TextAnalysisAgent,
    TextAnalysisResult,
)


# ── Schema tests ──────────────────────────────────────────────────────────────

def test_animal_traits_defaults():
    t = AnimalTraits()
    assert t.size is None
    assert t.distinctive_features == []


def test_text_analysis_result_requires_traits_and_severity():
    result = TextAnalysisResult(
        traits=AnimalTraits(size="large", colour="orange with black stripes"),
        severity=SeverityLevel.HIGH,
        severity_reason="Animal was nearby.",
    )
    assert result.severity == SeverityLevel.HIGH
    assert result.scientific_name is None
    assert result.identification_confidence == "low"


def test_severity_level_values():
    assert SeverityLevel.CRITICAL == "critical"
    assert SeverityLevel.INFORMATIONAL == "informational"
    assert len(list(SeverityLevel)) == 5


# ── TextAnalysisAgent — success path ─────────────────────────────────────────

def _make_agent(llm_response: str) -> TextAnalysisAgent:
    """Build a TextAnalysisAgent with a mocked LLM that returns llm_response."""
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = llm_response
    mock_llm.ainvoke.return_value = mock_response
    return TextAnalysisAgent(llm=mock_llm)


@pytest.fixture
def tiger_gpt_response():
    return json.dumps({
        "traits": {
            "size": "large",
            "colour": "orange with black stripes",
            "behaviour": "stalking",
            "movement": "slow and cautious",
            "distinctive_features": ["stripes", "long tail"],
            "count": "single animal",
            "additional": "animal was very close to the user",
        },
        "severity": "high",
        "severity_reason": "Large predatory cat observed at close range.",
        "scientific_name": "Panthera tigris",
        "common_name": "Bengal Tiger",
        "identification_confidence": "high",
        "identification_note": "Orange and black striped large cat — consistent with Bengal Tiger.",
    })


@pytest.mark.asyncio
async def test_run_returns_correct_severity(tiger_gpt_response):
    agent = _make_agent(tiger_gpt_response)
    result = await agent.run("I saw a huge orange cat with black stripes nearby")
    assert result.severity == SeverityLevel.HIGH
    assert "predatory" in result.severity_reason.lower()


@pytest.mark.asyncio
async def test_run_extracts_traits(tiger_gpt_response):
    agent = _make_agent(tiger_gpt_response)
    result = await agent.run("I saw a huge orange cat with black stripes nearby")
    assert result.traits.size == "large"
    assert result.traits.colour == "orange with black stripes"
    assert "stripes" in result.traits.distinctive_features


@pytest.mark.asyncio
async def test_run_identifies_species(tiger_gpt_response):
    agent = _make_agent(tiger_gpt_response)
    result = await agent.run("I saw a huge orange cat with black stripes nearby")
    assert result.scientific_name == "Panthera tigris"
    assert result.common_name == "Bengal Tiger"
    assert result.identification_confidence == "high"


@pytest.mark.asyncio
async def test_run_unidentified_when_null():
    response = json.dumps({
        "traits": {"size": "medium", "colour": "brown", "behaviour": "running",
                   "movement": None, "distinctive_features": [], "count": None, "additional": None},
        "severity": "low",
        "severity_reason": "Distant sighting, no immediate threat.",
        "scientific_name": None,
        "common_name": None,
        "identification_confidence": "low",
        "identification_note": "Description too vague to identify the species.",
    })
    agent = _make_agent(response)
    result = await agent.run("I saw a brown animal running")
    assert result.scientific_name is None
    assert result.identification_confidence == "low"


@pytest.mark.asyncio
async def test_run_critical_severity():
    response = json.dumps({
        "traits": {"size": "very large", "colour": "tawny", "behaviour": "aggressive",
                   "movement": "charging", "distinctive_features": ["mane"], "count": None, "additional": None},
        "severity": "critical",
        "severity_reason": "Lion charging toward user — immediate danger.",
        "scientific_name": "Panthera leo",
        "common_name": "Lion",
        "identification_confidence": "high",
        "identification_note": "Large maned cat charging — Panthera leo.",
    })
    agent = _make_agent(response)
    result = await agent.run("A lion is running towards me right now!")
    assert result.severity == SeverityLevel.CRITICAL


# ── TextAnalysisAgent — error paths ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_bad_json_returns_fallback():
    """Malformed JSON from LLM should return safe fallback, not raise."""
    agent = _make_agent("This is not JSON at all!!!")
    result = await agent.run("some message")
    assert result.severity == SeverityLevel.LOW
    assert "error" in result.identification_note.lower()


@pytest.mark.asyncio
async def test_run_unknown_severity_defaults_to_low():
    """Unknown severity string should fall back to LOW without crashing."""
    response = json.dumps({
        "traits": {"size": None, "colour": None, "behaviour": None,
                   "movement": None, "distinctive_features": [], "count": None, "additional": None},
        "severity": "extreme_danger",  # not a valid SeverityLevel
        "severity_reason": "Unknown severity value.",
        "scientific_name": None,
        "common_name": None,
        "identification_confidence": "low",
        "identification_note": "",
    })
    agent = _make_agent(response)
    result = await agent.run("some message")
    assert result.severity == SeverityLevel.LOW


@pytest.mark.asyncio
async def test_run_strips_markdown_fences():
    """GPT sometimes wraps JSON in ```json fences — should still parse."""
    inner = json.dumps({
        "traits": {"size": "small", "colour": "green", "behaviour": "stationary",
                   "movement": None, "distinctive_features": ["scales"], "count": None, "additional": None},
        "severity": "low",
        "severity_reason": "Small reptile, no threat.",
        "scientific_name": "Calotes versicolor",
        "common_name": "Oriental Garden Lizard",
        "identification_confidence": "medium",
        "identification_note": "Green lizard with scales — likely garden lizard.",
    })
    wrapped = f"```json\n{inner}\n```"
    agent = _make_agent(wrapped)
    result = await agent.run("There's a green lizard on the wall")
    assert result.scientific_name == "Calotes versicolor"

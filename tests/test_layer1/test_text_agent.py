"""
tests/test_layer1/test_text_agent.py
======================================
Unit tests for agents/layer1/text_agent.py (simplified 5-field schema)

Covers:
  - TextAnalysisResult structure (flat, no AnimalTraits)
  - SeverityLevel enum values
  - TextAnalysisAgent.run() — success path with mocked LLM
  - identification_confidence as int 0-100
  - Error / malformed JSON fallback
  - Unknown severity defaults to LOW
  - Markdown fence stripping
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from agents.enums import SeverityLevel
from agents.layer1.text_agent import (
    TextAnalysisAgent,
    TextAnalysisResult,
)


# ── Schema tests ──────────────────────────────────────────────────────────────

def test_text_analysis_result_defaults():
    result = TextAnalysisResult(severity=SeverityLevel.LOW)
    assert result.size is None
    assert result.scientific_name is None
    assert result.common_name is None
    assert result.identification_confidence == 0


def test_text_analysis_result_fields():
    result = TextAnalysisResult(
        size="large",
        severity=SeverityLevel.HIGH,
        scientific_name="Panthera tigris",
        common_name="Bengal Tiger",
        identification_confidence=90,
    )
    assert result.severity == SeverityLevel.HIGH
    assert result.size == "large"
    assert result.identification_confidence == 90


def test_severity_level_values():
    assert SeverityLevel.CRITICAL == "critical"
    assert SeverityLevel.INFORMATIONAL == "informational"
    assert len(list(SeverityLevel)) == 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_agent(llm_response: str) -> TextAnalysisAgent:
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = llm_response
    mock_response.response_metadata = {"finish_reason": "stop"}
    mock_llm.ainvoke.return_value = mock_response
    return TextAnalysisAgent(llm=mock_llm)


@pytest.fixture
def tiger_response():
    return json.dumps({
        "size": "large",
        "severity": "high",
        "scientific_name": "Panthera tigris",
        "common_name": "Bengal Tiger",
        "identification_confidence": 92,
    })


# ── Success paths ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_returns_correct_severity(tiger_response):
    agent = _make_agent(tiger_response)
    result = await agent.run("I saw a huge orange cat with black stripes nearby")
    assert result.severity == SeverityLevel.HIGH


@pytest.mark.asyncio
async def test_run_extracts_size(tiger_response):
    agent = _make_agent(tiger_response)
    result = await agent.run("I saw a huge orange cat with black stripes nearby")
    assert result.size == "large"


@pytest.mark.asyncio
async def test_run_identifies_species(tiger_response):
    agent = _make_agent(tiger_response)
    result = await agent.run("I saw a huge orange cat with black stripes nearby")
    assert result.scientific_name == "Panthera tigris"
    assert result.common_name == "Bengal Tiger"
    assert result.identification_confidence == 92


@pytest.mark.asyncio
async def test_run_confidence_clamped_to_100():
    response = json.dumps({
        "size": None,
        "severity": "low",
        "scientific_name": None,
        "common_name": None,
        "identification_confidence": 150,  # out of range
    })
    result = await _make_agent(response).run("some message")
    assert result.identification_confidence == 100


@pytest.mark.asyncio
async def test_run_confidence_string_mapped():
    """Old string confidence values mapped gracefully."""
    response = json.dumps({
        "size": None,
        "severity": "low",
        "scientific_name": None,
        "common_name": None,
        "identification_confidence": "medium",  # string fallback
    })
    result = await _make_agent(response).run("some message")
    assert result.identification_confidence == 50


@pytest.mark.asyncio
async def test_run_unidentified_when_null():
    response = json.dumps({
        "size": "medium",
        "severity": "low",
        "scientific_name": None,
        "common_name": None,
        "identification_confidence": 10,
    })
    result = await _make_agent(response).run("I saw a brown animal running")
    assert result.scientific_name is None
    assert result.identification_confidence == 10


@pytest.mark.asyncio
async def test_run_critical_severity():
    response = json.dumps({
        "size": "very large",
        "severity": "critical",
        "scientific_name": "Panthera leo",
        "common_name": "Lion",
        "identification_confidence": 95,
    })
    result = await _make_agent(response).run("A lion is running towards me right now!")
    assert result.severity == SeverityLevel.CRITICAL


# ── Error paths ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_bad_json_returns_fallback():
    """Malformed JSON from LLM should return safe fallback, not raise."""
    result = await _make_agent("This is not JSON at all!!!").run("some message")
    assert result.severity == SeverityLevel.LOW
    assert result.identification_confidence == 0


@pytest.mark.asyncio
async def test_run_unknown_severity_defaults_to_low():
    response = json.dumps({
        "size": None, "severity": "extreme_danger",
        "scientific_name": None, "common_name": None,
        "identification_confidence": 0,
    })
    result = await _make_agent(response).run("some message")
    assert result.severity == SeverityLevel.LOW


@pytest.mark.asyncio
async def test_run_strips_markdown_fences():
    inner = json.dumps({
        "size": "small",
        "severity": "low",
        "scientific_name": "Calotes versicolor",
        "common_name": "Oriental Garden Lizard",
        "identification_confidence": 55,
    })
    wrapped = f"```json\n{inner}\n```"
    result = await _make_agent(wrapped).run("There's a green lizard on the wall")
    assert result.scientific_name == "Calotes versicolor"

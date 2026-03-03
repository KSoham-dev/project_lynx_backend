"""
agents/layer1/schemas.py
=========================
Pydantic models for Layer 1 Image Analysis Agent.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class RelevancyResult(BaseModel):
    """Output of the relevancy check step."""
    is_relevant: bool
    reason: str
    top_detection_label: Optional[str] = None
    top_detection_score: Optional[float] = None


class IdentificationResult(BaseModel):
    """Output of the species identification step."""
    scientific_name: str          # "Panthera pardus" or "Unknown"
    confidence: float             # 0.0–1.0  (0.0 when source="gpt")
    source: str                   # "model" | "gpt" | "gpt_error"
    top_label_raw: Optional[str] = None   # raw SpeciesNet taxonomy label
    family: Optional[str] = None          # e.g. "FELIDAE"
    genus: Optional[str] = None           # e.g. "Panthera"


class ImageAnalysisResult(BaseModel):
    """
    Final result returned by ImageAnalysisAgent.run() to the orchestrator.
    Serialised as JSON string inside the LangChain tool response.
    """
    is_relevant: bool
    relevancy_reason: str
    scientific_name: Optional[str] = None
    confidence: Optional[float] = None
    identification_source: Optional[str] = None   # "model" | "gpt" | "gpt_error"
    family: Optional[str] = None
    genus: Optional[str] = None
    image_url: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    cached: bool = False

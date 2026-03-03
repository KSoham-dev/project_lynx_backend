"""
agents/layer0/state/schemas.py
================================
Pydantic models for the *prahari-state* Cosmos DB database.

Containers covered
------------------
sessions              → SessionDocument
agent_traces          → AgentTraceRecord
image_analysis_cache  → ImageAnalysisCacheDoc
species_context_cache → SpeciesContextCacheDoc
location_cache        → LocationCacheDoc

All documents follow the Cosmos DB convention:
  - ``id``  is the unique document key (string)
  - The partition key field matches ``id`` in every container
    (i.e., each container uses /id as its partition key path).

TTL fields (``_ttl``) are set as integers (seconds) — you must enable
the DefaultTimeToLive policy on the Cosmos container for TTL to take effect.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from agents.enums import AgentLayer, HumanRiskLevel, MessageRole, QueryType


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ── sessions ──────────────────────────────────────────────────────────────────

class MessageRecord(BaseModel):
    """A single conversation turn inside a session document."""

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=_utcnow)
    tool_name: Optional[str] = None        # populated when role == TOOL
    layer: Optional[AgentLayer] = None     # which agent layer produced this

    model_config = {"populate_by_name": True}


class SessionDocument(BaseModel):
    """
    Cosmos DB document for the ``sessions`` container.

    Tracks the full conversation history and metadata for one user session.
    One session = one continuous conversation across the frontend.

    Partition key: /id  (= session_id)
    """

    id: str                                       # session UUID  — Cosmos PK
    session_id: str                               # alias for readability
    user_id: str
    query_type: QueryType
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    messages: list[MessageRecord] = Field(default_factory=list)
    active_layer: AgentLayer = AgentLayer.LAYER0  # advances as processing deepens
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_cosmos(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_cosmos(cls, doc: dict[str, Any]) -> "SessionDocument":
        return cls.model_validate(doc)


# ── agent_traces ──────────────────────────────────────────────────────────────

class AgentTraceRecord(BaseModel):
    """
    Cosmos DB document for the ``agent_traces`` container.

    One record per agent invocation (tool call, sub-agent, LLM call).
    Enables full auditability and latency analysis across all layers.

    Partition key: /session_id
    """

    id: str                                       # trace UUID
    session_id: str                               # links to SessionDocument
    layer: AgentLayer
    agent_name: str                               # e.g. "relevancy_check", "reporter"
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    latency_ms: Optional[int] = None
    timestamp: datetime = Field(default_factory=_utcnow)
    success: bool = True
    error: Optional[str] = None

    def to_cosmos(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_cosmos(cls, doc: dict[str, Any]) -> "AgentTraceRecord":
        return cls.model_validate(doc)


# ── image_analysis_cache ──────────────────────────────────────────────────────

class ImageAnalysisCacheDoc(BaseModel):
    """
    Cosmos DB document for the ``image_analysis_cache`` container.

    Caches SpeciesNet predictions keyed by a hash of the image URL.
    TTL: 7 days (604 800 s) — requires DefaultTimeToLive set on container.

    Partition key: /id  (= SHA-256 hex of image_url, truncated to 64 chars)
    """

    id: str                                    # hash of image_url
    image_url: str
    top_species: str
    confidence: float
    classifications: list[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    ttl: int = Field(default=604_800, alias="_ttl")    # 7 days

    model_config = {"populate_by_name": True}

    def to_cosmos(self) -> dict[str, Any]:
        d = self.model_dump(mode="json", by_alias=True)
        return d

    @classmethod
    def from_cosmos(cls, doc: dict[str, Any]) -> "ImageAnalysisCacheDoc":
        return cls.model_validate(doc)


# ── species_context_cache ─────────────────────────────────────────────────────

class SpeciesContextCacheDoc(BaseModel):
    """
    Cosmos DB document for the ``species_context_cache`` container.

    Caches IUCN + Wikipedia + iNaturalist results for one species.
    TTL: 24 hours (86 400 s).

    Partition key: /id  (= normalised scientific name, lower-cased, spaces→underscores)
    """

    id: str                                    # normalised species name
    species_name: str                          # canonical scientific name
    iucn_data: dict[str, Any] = Field(default_factory=dict)
    wiki_extract: str = ""
    photo_url: Optional[str] = None
    photo_credit: Optional[str] = None
    traits: dict[str, Any] = Field(default_factory=dict)   # LLM-extracted trait dict
    iucn_category: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)
    ttl: int = Field(default=86_400, alias="_ttl")          # 24 hours

    model_config = {"populate_by_name": True}

    def to_cosmos(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)

    @classmethod
    def from_cosmos(cls, doc: dict[str, Any]) -> "SpeciesContextCacheDoc":
        return cls.model_validate(doc)


# ── location_cache ────────────────────────────────────────────────────────────

class LocationCacheDoc(BaseModel):
    """
    Cosmos DB document for the ``location_cache`` container.

    Caches reverse-geocoded place data for a geohash cell.
    TTL: 30 days (2 592 000 s).

    Partition key: /id  (= geohash string, precision 6 ≈ 1 km²)
    """

    id: str                                    # geohash (precision 6)
    latitude: float
    longitude: float
    district: Optional[str] = None
    state: Optional[str] = None
    country: str = "India"
    forest_zone: Optional[str] = None
    protected_area: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)
    ttl: int = Field(default=2_592_000, alias="_ttl")       # 30 days

    model_config = {"populate_by_name": True}

    def to_cosmos(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)

    @classmethod
    def from_cosmos(cls, doc: dict[str, Any]) -> "LocationCacheDoc":
        return cls.model_validate(doc)

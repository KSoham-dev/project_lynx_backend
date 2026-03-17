"""
agents/layer0/state/data_schemas.py
=====================================
Pydantic models for the *prahari-data* Cosmos DB database.

Containers covered
------------------
users        → UserDocument
reports      → ReportDocument
explorations → ExplorationDocument
encyclopedia → EncyclopediaDocument
incidents    → IncidentDocument
sos_events   → SOSEventDocument

All partition keys follow the pattern /id (document-level partitioning)
unless the document is clearly subordinate to a single user, in which case
the partition key is /user_id to collocate all user data.

Device metadata tracked here (in UserDocument.devices) enables the
targeted push-notification pipeline in a later phase. The React frontend
should send X-Device-ID and X-FCM-Token request headers; the Receiver
extracts them and upserts the DeviceInfo record on every session.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from agents.enums import (
    HumanRiskLevel,
    IncidentSeverity,
    IncidentStatus,
    IncidentType,
    QueryType,
    ReportStatus,
    SOSStatus,
    UserRole,
)


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ── Device metadata (embedded in UserDocument) ────────────────────────────────

class DeviceInfo(BaseModel):
    """
    Metadata for a single registered device belonging to a user.

    The ``device_id`` is a client-generated UUID persisted in the React
    app's localStorage (survives page refreshes but not clear-storage).
    ``fcm_token`` is the Firebase Cloud Messaging token for web-push —
    refreshed by the client and sent on every session start via the
    ``X-FCM-Token`` header.
    """

    device_id: str                                  # client-generated UUID
    fcm_token: Optional[str] = None                 # FCM web-push token
    user_agent: Optional[str] = None                # HTTP User-Agent header
    platform: str = "web"                           # "web" | "android" | "ios"
    last_seen: datetime = Field(default_factory=_utcnow)
    registered_at: datetime = Field(default_factory=_utcnow)


class ActiveSession(BaseModel):
    """
    Tracking a stateful JWT session to allow strict invalidation (logouts).
    """
    jti: str
    created_at: datetime = Field(default_factory=_utcnow)
    expires_at: datetime


# ── users ─────────────────────────────────────────────────────────────────────

class UserDocument(BaseModel):
    """
    Cosmos DB document for the ``users`` container.

    Partition key: /id  (= user_id)

    The ``devices`` list stores all registered devices for targeted
    push notifications.  On each session start the Receiver upserts the
    current device info (keyed by device_id) so tokens stay fresh.
    """

    id: str                                               # user_id — Cosmos PK
    user_id: str
    name: str = ""
    mobile_number: Optional[str] = None
    email: Optional[str] = None
    hashed_password: Optional[str] = None
    device_fsm_token: Optional[str] = None
    role: UserRole = UserRole.PUBLIC
    forest_zone: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    devices: list[DeviceInfo] = Field(default_factory=list)
    active_sessions: list[ActiveSession] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    last_active: datetime = Field(default_factory=_utcnow)
    preferences: dict[str, Any] = Field(default_factory=dict)

    def cleanup_sessions(self) -> None:
        """Remove expired active sessions."""
        now = _utcnow()
        self.active_sessions = [s for s in self.active_sessions if s.expires_at > now]

    def add_session(self, jti: str, expires_at: datetime) -> None:
        """Add a new session and cleanup expired ones."""
        self.cleanup_sessions()
        self.active_sessions.append(ActiveSession(jti=jti, expires_at=expires_at))

    def remove_session(self, jti: str) -> None:
        """Remove a session by jti."""
        self.active_sessions = [s for s in self.active_sessions if s.jti != jti]

    def upsert_device(self, device: DeviceInfo) -> None:
        """Replace existing device entry (by device_id) or append a new one."""
        self.devices = [d for d in self.devices if d.device_id != device.device_id]
        self.devices.append(device)
        self.last_active = _utcnow()

    def to_cosmos(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_cosmos(cls, doc: dict[str, Any]) -> "UserDocument":
        return cls.model_validate(doc)


# ── reports ───────────────────────────────────────────────────────────────────

class ReportDocument(BaseModel):
    """
    Cosmos DB document for the ``reports`` container.

    Created by the Layer 2 Reporter agent for query_type=REPORT sessions.
    Partition key: /user_id  (collocates a user's reports together)
    """

    id: str                                               # report UUID
    user_id: str                                          # partition key
    session_id: str
    species_name: Optional[str] = None
    image_url: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_context: Optional[dict[str, Any]] = None    # from location_cache
    species_context: Optional[dict[str, Any]] = None     # from species_context_cache
    risk_level: Optional[HumanRiskLevel] = None
    report_text: str = ""                                 # AI-generated report body
    status: ReportStatus = ReportStatus.DRAFT
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def to_cosmos(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_cosmos(cls, doc: dict[str, Any]) -> "ReportDocument":
        return cls.model_validate(doc)


# ── explorations ──────────────────────────────────────────────────────────────

class ObservationRecord(BaseModel):
    """One wildlife observation within an exploration session."""
    species_name: str
    image_url: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    notes: str = ""
    timestamp: datetime = Field(default_factory=_utcnow)


class ExplorationDocument(BaseModel):
    """
    Cosmos DB document for the ``explorations`` container.

    Created by the Layer 2 Explorer agent for query_type=EXPLORE sessions.
    Partition key: /user_id
    """

    id: str                                               # exploration UUID
    user_id: str                                          # partition key
    session_id: str
    area_name: Optional[str] = None
    observations: list[ObservationRecord] = Field(default_factory=list)
    species_found: list[str] = Field(default_factory=list)
    summary: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def to_cosmos(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_cosmos(cls, doc: dict[str, Any]) -> "ExplorationDocument":
        return cls.model_validate(doc)


# ── encyclopedia ──────────────────────────────────────────────────────────────

class EncyclopediaDocument(BaseModel):
    """
    Cosmos DB document for the ``encyclopedia`` container.

    Persistent species knowledge base populated by the Layer 2 Species Info
    agent for query_type=ENCYCLOPEDIA sessions.

    Partition key: /id  (= normalised species name, same as species_context_cache)

    Unlike the cache, encyclopedia entries are permanent and curated.
    """

    id: str                                               # normalised species name
    species_name: str
    common_names: list[str] = Field(default_factory=list)
    iucn_category: Optional[str] = None
    iucn_url: Optional[str] = None
    habitat: str = ""
    risk_level: Optional[HumanRiskLevel] = None
    traits: dict[str, Any] = Field(default_factory=dict)
    photo_url: Optional[str] = None
    photo_credit: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def to_cosmos(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_cosmos(cls, doc: dict[str, Any]) -> "EncyclopediaDocument":
        return cls.model_validate(doc)


# ── incidents ─────────────────────────────────────────────────────────────────

class IncidentDocument(BaseModel):
    """
    Cosmos DB document for the ``incidents`` container.

    Formal incident records created by the Layer 2 Incident agent.
    Can be linked to a ReportDocument and/or SOSEventDocument.

    Partition key: /id  (= incident UUID)
    """

    id: str                                               # incident UUID
    session_id: str
    user_id: str
    type: IncidentType = IncidentType.OTHER
    severity: IncidentSeverity = IncidentSeverity.MEDIUM
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_context: Optional[dict[str, Any]] = None
    description: str = ""
    animal_name: Optional[str] = None           # common name, or scientific name if absent
    linked_report_id: Optional[str] = None
    linked_sos_id: Optional[str] = None
    log_blob_url: Optional[str] = None          # URL of the append-blob incident log
    status: IncidentStatus = IncidentStatus.OPEN
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    resolved_at: Optional[datetime] = None

    def to_cosmos(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_cosmos(cls, doc: dict[str, Any]) -> "IncidentDocument":
        return cls.model_validate(doc)


# ── sos_events ────────────────────────────────────────────────────────────────

class SOSEventDocument(BaseModel):
    """
    Cosmos DB document for the ``sos_events`` container.

    SOS alerts triggered by the Layer 3 SOS agent.
    The ``device_id`` field ties the SOS back to a specific ranger device
    so targeted push notifications can be sent to the right devices.

    Partition key: /user_id
    """

    id: str                                               # SOS UUID
    user_id: str                                          # partition key
    session_id: str
    device_id: Optional[str] = None                       # device that triggered SOS
    latitude: float
    longitude: float
    location_context: Optional[dict[str, Any]] = None
    message: str = ""
    status: SOSStatus = SOSStatus.ACTIVE
    triggered_at: datetime = Field(default_factory=_utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    linked_incident_id: Optional[str] = None
    notified_devices: list[str] = Field(default_factory=list)  # device_ids notified

    def to_cosmos(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_cosmos(cls, doc: dict[str, Any]) -> "SOSEventDocument":
        return cls.model_validate(doc)

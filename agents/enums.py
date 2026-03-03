"""
agents/enums.py
================
Centralised enumerations used across all Prahari agent layers.

Import from here — never define enums inline in other modules.
"""

from __future__ import annotations

from enum import Enum


class QueryType(str, Enum):
    """
    The type of query the user is making.

    Drives which Layer 2 agent handles the final output:
        REPORT      → Reporter agent   (wildlife incident report)
        EXPLORE     → Explorer agent   (area / species exploration)
        ENCYCLOPEDIA → Species Info agent (species facts & IUCN data)
    """
    REPORT       = "report"
    EXPLORE      = "explore"
    ENCYCLOPEDIA = "encyclopedia"


class AgentLayer(str, Enum):
    """Identifies which processing layer produced a message or trace."""
    LAYER0 = "layer0"   # Orchestrator + Context/State
    LAYER1 = "layer1"   # Relevancy, Location, Image Identifier, Context/Text
    LAYER2 = "layer2"   # Reporter, Explorer, Species Info, Incident
    LAYER3 = "layer3"   # SOS, Message, Data


class MessageRole(str, Enum):
    """Roles in a conversation message record."""
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"
    TOOL      = "tool"


class HumanRiskLevel(str, Enum):
    """
    Standardised human-risk classification for a wildlife species.

    Mirrors the values expected by the Groq LLM extraction prompt.
    """
    VERY_HIGH = "Very High"
    HIGH      = "High"
    CAUTION   = "Caution"
    LOW       = "Low"


class NotificationChannel(str, Enum):
    """
    Targeted notification delivery channel.

    Used when pushing alerts / SOS acknowledgements to rangers.
    """
    WEB_PUSH = "web_push"   # FCM for React PWA
    EMAIL    = "email"
    SMS      = "sms"


class SOSStatus(str, Enum):
    """Lifecycle status of an SOS event."""
    ACTIVE       = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED     = "resolved"


class IncidentStatus(str, Enum):
    """Lifecycle status of an incident record."""
    OPEN         = "open"
    INVESTIGATING = "investigating"
    RESOLVED     = "resolved"


class IncidentType(str, Enum):
    """Category of a wildlife incident."""
    POACHING              = "poaching"
    HUMAN_WILDLIFE_CONFLICT = "human_wildlife_conflict"
    INJURED_ANIMAL        = "injured_animal"
    ILLEGAL_ENTRY         = "illegal_entry"
    OTHER                 = "other"


class IncidentSeverity(str, Enum):
    """Severity rating of a wildlife incident."""
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class ReportStatus(str, Enum):
    """Publication status of a generated report."""
    DRAFT     = "draft"
    SUBMITTED = "submitted"
    ARCHIVED  = "archived"


class UserRole(str, Enum):
    """Role / rank of a Prahari user."""
    RANGER  = "ranger"
    OFFICER = "officer"
    ADMIN   = "admin"
    PUBLIC  = "public"   # general public user


class SeverityLevel(str, Enum):
    """
    Semantic / intent severity of a user's wildlife message.

    Used by the Layer 1 TextAnalysisAgent to classify urgency.

    CRITICAL     – immediate danger (e.g. "tiger charging at me")
    HIGH         – active threat nearby (e.g. "saw a leopard 50m away")
    MEDIUM       – noteworthy sighting, no immediate danger
    LOW          – general curiosity / educational question
    INFORMATIONAL – purely informational, no wildlife interaction
    """
    CRITICAL      = "critical"
    HIGH          = "high"
    MEDIUM        = "medium"
    LOW           = "low"
    INFORMATIONAL = "informational"

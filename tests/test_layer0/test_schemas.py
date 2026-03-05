"""
tests/test_layer0/test_schemas.py
=================================
Unit tests for agents/layer0/state/schemas.py and agents/layer0/state/data_schemas.py
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agents.enums import AgentLayer, HumanRiskLevel, MessageRole, QueryType, SOSStatus
from agents.state.schemas import (
    AgentTraceRecord,
    ImageAnalysisCacheDoc,
    LocationCacheDoc,
    MessageRecord,
    SessionDocument,
    SpeciesContextCacheDoc,
)
from agents.state.data_schemas import (
    DeviceInfo,
    EncyclopediaDocument,
    ExplorationDocument,
    IncidentDocument,
    ReportDocument,
    SOSEventDocument,
    UserDocument,
)


# ── MessageRecord ─────────────────────────────────────────────────────────────

class TestMessageRecord:
    def test_defaults(self):
        msg = MessageRecord(role=MessageRole.USER, content="hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "hello"
        assert isinstance(msg.timestamp, datetime)
        assert msg.tool_name is None
        assert msg.layer is None

    def test_tool_role_with_tool_name(self):
        msg = MessageRecord(role=MessageRole.TOOL, content="result", tool_name="image_identifier")
        assert msg.tool_name == "image_identifier"

    def test_round_trip_json(self):
        msg = MessageRecord(role=MessageRole.ASSISTANT, content="Hi!", layer=AgentLayer.LAYER0)
        data = msg.model_dump(mode="json")
        restored = MessageRecord.model_validate(data)
        assert restored.role == msg.role
        assert restored.layer == msg.layer


# ── SessionDocument ───────────────────────────────────────────────────────────

class TestSessionDocument:
    def test_to_cosmos_returns_dict(self):
        doc = SessionDocument(id="s1", session_id="s1", user_id="u1", query_type=QueryType.REPORT)
        cosmos = doc.to_cosmos()
        assert isinstance(cosmos, dict)
        assert cosmos["id"] == "s1"
        assert cosmos["query_type"] == "report"
        assert cosmos["active_layer"] == "layer0"

    def test_from_cosmos_round_trip(self):
        doc = SessionDocument(id="s2", session_id="s2", user_id="u2", query_type=QueryType.EXPLORE)
        doc.messages.append(MessageRecord(role=MessageRole.USER, content="test"))
        cosmos = doc.to_cosmos()
        restored = SessionDocument.from_cosmos(cosmos)
        assert restored.query_type == QueryType.EXPLORE
        assert len(restored.messages) == 1
        assert restored.messages[0].role == MessageRole.USER

    def test_all_query_types(self):
        for qt in QueryType:
            doc = SessionDocument(id="s", session_id="s", user_id="u", query_type=qt)
            assert doc.query_type == qt

    def test_timestamps_are_utc(self):
        doc = SessionDocument(id="s3", session_id="s3", user_id="u3", query_type=QueryType.ENCYCLOPEDIA)
        assert doc.created_at.tzinfo is not None


# ── AgentTraceRecord ──────────────────────────────────────────────────────────

class TestAgentTraceRecord:
    def test_basic_construction(self):
        trace = AgentTraceRecord(
            id="t1",
            session_id="s1",
            layer=AgentLayer.LAYER1,
            agent_name="relevancy_check",
            latency_ms=45,
        )
        assert trace.layer == AgentLayer.LAYER1
        assert trace.success is True
        assert trace.error is None

    def test_to_cosmos(self):
        trace = AgentTraceRecord(id="t2", session_id="s1", layer=AgentLayer.LAYER2, agent_name="reporter")
        cosmos = trace.to_cosmos()
        assert cosmos["layer"] == "layer2"


# ── Cache documents ───────────────────────────────────────────────────────────

class TestCacheDocs:
    def test_image_cache_ttl(self):
        doc = ImageAnalysisCacheDoc(id="h1", image_url="https://example.com/img.jpg", top_species="Leo", confidence=0.92)
        cosmos = doc.to_cosmos()
        assert cosmos["_ttl"] == 604_800

    def test_species_cache_ttl(self):
        doc = SpeciesContextCacheDoc(id="panthera_leo", species_name="Panthera leo")
        cosmos = doc.to_cosmos()
        assert cosmos["_ttl"] == 86_400

    def test_location_cache_ttl(self):
        doc = LocationCacheDoc(id="geohash1", latitude=26.0, longitude=76.5)
        cosmos = doc.to_cosmos()
        assert cosmos["_ttl"] == 2_592_000


# ── Data schemas ──────────────────────────────────────────────────────────────

class TestDeviceInfo:
    def test_defaults(self):
        dev = DeviceInfo(device_id="dev-uuid-1")
        assert dev.platform == "web"
        assert dev.fcm_token is None

    def test_with_fcm_token(self):
        dev = DeviceInfo(device_id="dev-1", fcm_token="fcm-token-abc", user_agent="Mozilla/5.0")
        assert dev.fcm_token == "fcm-token-abc"


class TestUserDocument:
    def test_upsert_device_adds_new(self):
        user = UserDocument(id="u1", user_id="u1")
        user.upsert_device(DeviceInfo(device_id="d1", fcm_token="tok1"))
        assert len(user.devices) == 1

    def test_upsert_device_replaces_existing(self):
        user = UserDocument(id="u1", user_id="u1")
        user.upsert_device(DeviceInfo(device_id="d1", fcm_token="old"))
        user.upsert_device(DeviceInfo(device_id="d1", fcm_token="new"))
        assert len(user.devices) == 1
        assert user.devices[0].fcm_token == "new"

    def test_upsert_device_multiple(self):
        user = UserDocument(id="u1", user_id="u1")
        user.upsert_device(DeviceInfo(device_id="d1"))
        user.upsert_device(DeviceInfo(device_id="d2"))
        assert len(user.devices) == 2


class TestSOSEventDocument:
    def test_defaults(self):
        sos = SOSEventDocument(id="sos1", user_id="u1", session_id="s1", latitude=26.0, longitude=76.5)
        assert sos.status == SOSStatus.ACTIVE
        assert sos.notified_devices == []
        assert sos.device_id is None

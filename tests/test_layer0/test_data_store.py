"""
tests/test_layer0/test_data_store.py
=====================================
Unit tests for agents/layer0/state/data_store.py

All Cosmos DB containers are mocked — no real Azure calls are made.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.enums import IncidentSeverity, IncidentStatus, IncidentType, ReportStatus, SOSStatus
from agents.state.data_schemas import (
    DeviceInfo,
    EncyclopediaDocument,
    ExplorationDocument,
    IncidentDocument,
    ObservationRecord,
    ReportDocument,
    SOSEventDocument,
    UserDocument,
)
from agents.state.data_store import (
    acknowledge_sos,
    add_notified_device,
    append_observation,
    create_exploration,
    create_incident,
    create_report,
    create_sos_event,
    get_encyclopedia_entry,
    get_incident,
    get_report,
    get_sos_event,
    get_user,
    register_device,
    resolve_sos,
    update_incident_status,
    update_report_status,
    upsert_encyclopedia_entry,
    upsert_user,
)


# ── Shared mock container fixture ─────────────────────────────────────────────

@pytest.fixture
def mock_container():
    c = AsyncMock()
    c.create_item = AsyncMock()
    c.upsert_item = AsyncMock()
    c.read_item = AsyncMock()
    return c


@pytest.fixture(autouse=True)
def patch_data_container(mock_container):
    with patch(
        "agents.state.data_store.get_data_container",
        new=AsyncMock(return_value=mock_container),
    ):
        yield mock_container


# ═══════════════════════════════════════════════════════════════════════════════
# users
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_get_user_returns_none_when_not_found(patch_data_container):
    from azure.cosmos.exceptions import CosmosResourceNotFoundError
    patch_data_container.read_item = AsyncMock(
        side_effect=CosmosResourceNotFoundError(message="not found", response=MagicMock())
    )
    result = await get_user("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_get_user_returns_document(patch_data_container):
    user = UserDocument(id="u1", user_id="u1")
    patch_data_container.read_item = AsyncMock(return_value=user.to_cosmos())
    result = await get_user("u1")
    assert result is not None
    assert result.user_id == "u1"


@pytest.mark.asyncio
async def test_upsert_user_calls_upsert(patch_data_container):
    user = UserDocument(id="u2", user_id="u2", name="Ranger Priya")
    await upsert_user(user)
    patch_data_container.upsert_item.assert_awaited_once()


@pytest.mark.asyncio
async def test_register_device_creates_user_if_not_exists(patch_data_container):
    from azure.cosmos.exceptions import CosmosResourceNotFoundError
    patch_data_container.read_item = AsyncMock(
        side_effect=CosmosResourceNotFoundError(message="not found", response=MagicMock())
    )
    patch_data_container.upsert_item = AsyncMock()
    device = DeviceInfo(device_id="dev-1", fcm_token="tok-abc")
    user = await register_device("new-user", device)
    assert user.user_id == "new-user"
    assert len(user.devices) == 1
    assert user.devices[0].device_id == "dev-1"
    patch_data_container.upsert_item.assert_awaited_once()


@pytest.mark.asyncio
async def test_register_device_updates_existing_token(patch_data_container):
    user = UserDocument(id="u3", user_id="u3")
    user.devices.append(DeviceInfo(device_id="dev-1", fcm_token="old-token"))
    patch_data_container.read_item = AsyncMock(return_value=user.to_cosmos())
    patch_data_container.upsert_item = AsyncMock()
    await register_device("u3", DeviceInfo(device_id="dev-1", fcm_token="new-token"))
    upsert_call_body = patch_data_container.upsert_item.call_args[1].get("body") or \
                        patch_data_container.upsert_item.call_args[0][0]
    assert upsert_call_body["devices"][0]["fcm_token"] == "new-token"


# ═══════════════════════════════════════════════════════════════════════════════
# reports
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_create_report_generates_uuid(patch_data_container):
    doc = await create_report(
        user_id="u1",
        session_id="s1",
        report_text="Spotted a leopard.",
        species_name="Panthera pardus",
    )
    assert len(doc.id) == 36  # UUID4
    assert doc.status == ReportStatus.DRAFT
    patch_data_container.create_item.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_report_status(patch_data_container):
    report = ReportDocument(id="r1", user_id="u1", session_id="s1", report_text="text")
    patch_data_container.read_item = AsyncMock(return_value=report.to_cosmos())
    await update_report_status("r1", "u1", ReportStatus.SUBMITTED)
    upserted = patch_data_container.upsert_item.call_args[1].get("body") or \
                patch_data_container.upsert_item.call_args[0][0]
    assert upserted["status"] == "submitted"


# ═══════════════════════════════════════════════════════════════════════════════
# explorations
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_create_exploration(patch_data_container):
    doc = await create_exploration(user_id="u1", session_id="s1", area_name="Ranthambore Zone 3")
    assert doc.area_name == "Ranthambore Zone 3"
    assert len(doc.id) == 36
    patch_data_container.create_item.assert_awaited_once()


@pytest.mark.asyncio
async def test_append_observation_updates_species_list(patch_data_container):
    exp = ExplorationDocument(id="e1", user_id="u1", session_id="s1")
    patch_data_container.read_item = AsyncMock(return_value=exp.to_cosmos())
    obs = ObservationRecord(species_name="Panthera tigris", notes="Near the waterhole")
    await append_observation("e1", "u1", obs)
    upserted = patch_data_container.upsert_item.call_args[1].get("body") or \
                patch_data_container.upsert_item.call_args[0][0]
    assert "Panthera tigris" in upserted["species_found"]
    assert len(upserted["observations"]) == 1


@pytest.mark.asyncio
async def test_append_observation_deduplicates_species(patch_data_container):
    exp = ExplorationDocument(id="e2", user_id="u1", session_id="s1", species_found=["Panthera tigris"])
    patch_data_container.read_item = AsyncMock(return_value=exp.to_cosmos())
    obs = ObservationRecord(species_name="Panthera tigris")
    await append_observation("e2", "u1", obs)
    upserted = patch_data_container.upsert_item.call_args[1].get("body") or \
                patch_data_container.upsert_item.call_args[0][0]
    assert upserted["species_found"].count("Panthera tigris") == 1


# ═══════════════════════════════════════════════════════════════════════════════
# encyclopedia
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_get_encyclopedia_entry_returns_none_when_missing(patch_data_container):
    from azure.cosmos.exceptions import CosmosResourceNotFoundError
    patch_data_container.read_item = AsyncMock(
        side_effect=CosmosResourceNotFoundError(message="not found", response=MagicMock())
    )
    result = await get_encyclopedia_entry("Unknown species")
    assert result is None


@pytest.mark.asyncio
async def test_upsert_encyclopedia_entry(patch_data_container):
    doc = EncyclopediaDocument(id="panthera_leo", species_name="Panthera leo")
    await upsert_encyclopedia_entry(doc)
    patch_data_container.upsert_item.assert_awaited_once()


# ═══════════════════════════════════════════════════════════════════════════════
# incidents
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_create_incident(patch_data_container):
    doc = await create_incident(
        session_id="s1",
        user_id="u1",
        description="Poaching attempt near Zone 4",
        type=IncidentType.POACHING,
        severity=IncidentSeverity.CRITICAL,
    )
    assert len(doc.id) == 36
    assert doc.type == IncidentType.POACHING
    assert doc.severity == IncidentSeverity.CRITICAL
    patch_data_container.create_item.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_incident_status_to_resolved(patch_data_container):
    incident = IncidentDocument(id="i1", session_id="s1", user_id="u1", description="Poaching")
    patch_data_container.read_item = AsyncMock(return_value=incident.to_cosmos())
    await update_incident_status("i1", IncidentStatus.RESOLVED, resolved=True)
    upserted = patch_data_container.upsert_item.call_args[1].get("body") or \
                patch_data_container.upsert_item.call_args[0][0]
    assert upserted["status"] == "resolved"
    assert upserted["resolved_at"] is not None


# ═══════════════════════════════════════════════════════════════════════════════
# sos_events
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_create_sos_event(patch_data_container):
    doc = await create_sos_event(
        user_id="u1",
        session_id="s1",
        latitude=26.0,
        longitude=76.5,
        message="Help! Poacher spotted.",
        device_id="dev-1",
    )
    assert doc.status == SOSStatus.ACTIVE
    assert doc.device_id == "dev-1"
    assert len(doc.id) == 36
    patch_data_container.create_item.assert_awaited_once()


@pytest.mark.asyncio
async def test_acknowledge_sos(patch_data_container):
    sos = SOSEventDocument(id="sos1", user_id="u1", session_id="s1", latitude=26.0, longitude=76.5)
    patch_data_container.read_item = AsyncMock(return_value=sos.to_cosmos())
    await acknowledge_sos("sos1", "u1")
    upserted = patch_data_container.upsert_item.call_args[1].get("body") or \
                patch_data_container.upsert_item.call_args[0][0]
    assert upserted["status"] == "acknowledged"
    assert upserted["acknowledged_at"] is not None


@pytest.mark.asyncio
async def test_resolve_sos_with_linked_incident(patch_data_container):
    sos = SOSEventDocument(id="sos2", user_id="u1", session_id="s1", latitude=26.0, longitude=76.5)
    patch_data_container.read_item = AsyncMock(return_value=sos.to_cosmos())
    await resolve_sos("sos2", "u1", linked_incident_id="incident-xyz")
    upserted = patch_data_container.upsert_item.call_args[1].get("body") or \
                patch_data_container.upsert_item.call_args[0][0]
    assert upserted["status"] == "resolved"
    assert upserted["linked_incident_id"] == "incident-xyz"
    assert upserted["resolved_at"] is not None


@pytest.mark.asyncio
async def test_add_notified_device_deduplicates(patch_data_container):
    sos = SOSEventDocument(
        id="sos3", user_id="u1", session_id="s1",
        latitude=26.0, longitude=76.5,
        notified_devices=["dev-1"],
    )
    patch_data_container.read_item = AsyncMock(return_value=sos.to_cosmos())
    await add_notified_device("sos3", "u1", "dev-1")   # already in list
    # upsert should NOT be called since device already notified
    patch_data_container.upsert_item.assert_not_awaited()


@pytest.mark.asyncio
async def test_add_notified_device_adds_new(patch_data_container):
    sos = SOSEventDocument(
        id="sos4", user_id="u1", session_id="s1",
        latitude=26.0, longitude=76.5,
    )
    patch_data_container.read_item = AsyncMock(return_value=sos.to_cosmos())
    await add_notified_device("sos4", "u1", "dev-2")
    upserted = patch_data_container.upsert_item.call_args[1].get("body") or \
                patch_data_container.upsert_item.call_args[0][0]
    assert "dev-2" in upserted["notified_devices"]

"""
agents/state/data_store.py
===========================
Async CRUD helpers for the *prahari-data* Cosmos DB database.

Shared across all agent layers — import from agents.state.data_store,
never from agents.layer0.state.data_store.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from azure.cosmos.exceptions import CosmosResourceNotFoundError

from agents.enums import IncidentStatus, ReportStatus, SOSStatus
from agents.state.containers import DataContainers
from agents.state.cosmos_client import get_data_container
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

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ── users ─────────────────────────────────────────────────────────────────────

async def get_user(user_id: str) -> Optional[UserDocument]:
    container = await get_data_container(DataContainers.USERS)
    try:
        doc = await container.read_item(item=user_id, partition_key=user_id)
        return UserDocument.from_cosmos(doc)
    except CosmosResourceNotFoundError:
        return None


async def upsert_user(doc: UserDocument) -> None:
    doc.last_active = _utcnow()
    container = await get_data_container(DataContainers.USERS)
    await container.upsert_item(body=doc.to_cosmos())
    logger.debug("User upserted: id=%s", doc.id)


async def register_device(user_id: str, device: DeviceInfo) -> UserDocument:
    """Upsert a device entry on UserDocument, creating the user if absent."""
    user = await get_user(user_id)
    if user is None:
        user = UserDocument(id=user_id, user_id=user_id)
        logger.info("Auto-creating user document: id=%s", user_id)
    user.upsert_device(device)
    await upsert_user(user)
    logger.info("Device registered: user=%s device=%s", user_id, device.device_id)
    return user


async def get_user_by_email(email: str) -> Optional[UserDocument]:
    """Retrieve a user by their email using a cross-partition query."""
    container = await get_data_container(DataContainers.USERS)
    query = "SELECT * FROM c WHERE c.email = @email"
    parameters = [{"name": "@email", "value": email}]
    try:
        items = [item async for item in container.query_items(
            query=query,
            parameters=parameters
        )]
        if not items:
            return None
        return UserDocument.from_cosmos(items[0])
    except Exception as e:
        logger.error("Error querying user by email: %s", e)
        return None

async def get_all_users() -> list[dict]:
    """Retrieve all users across partitions for dev bench."""
    container = await get_data_container(DataContainers.USERS)
    query = "SELECT c.id, c.user_id, c.name, c.email, c.role, c.last_active, c.created_at, c.devices FROM c"
    try:
        items = [item async for item in container.query_items(
            query=query
        )]
        return items
    except Exception as e:
        logger.error("Error querying all users: %s", e)
        return []

# ── reports ───────────────────────────────────────────────────────────────────

async def create_report(
    user_id: str,
    session_id: str,
    report_text: str,
    **kwargs,
) -> ReportDocument:
    report_id = str(uuid.uuid4())
    doc = ReportDocument(id=report_id, user_id=user_id, session_id=session_id, report_text=report_text, **kwargs)
    container = await get_data_container(DataContainers.REPORTS)
    await container.create_item(body=doc.to_cosmos())
    logger.info("Report created: id=%s user=%s", report_id, user_id)
    return doc


async def get_report(report_id: str, user_id: str) -> Optional[ReportDocument]:
    container = await get_data_container(DataContainers.REPORTS)
    try:
        doc = await container.read_item(item=report_id, partition_key=user_id)
        return ReportDocument.from_cosmos(doc)
    except CosmosResourceNotFoundError:
        return None


async def update_report_status(report_id: str, user_id: str, status: ReportStatus) -> None:
    doc = await get_report(report_id, user_id)
    if doc is None:
        raise ValueError(f"Report '{report_id}' not found for user '{user_id}'.")
    doc.status = status
    doc.updated_at = _utcnow()
    container = await get_data_container(DataContainers.REPORTS)
    await container.upsert_item(body=doc.to_cosmos())


async def upsert_report(report_dict: dict) -> None:
    """
    Upsert a raw report dict into Cosmos prahari-data/reports.
    Used by Layer 2 ReporterAgent.
    Stamps created_at / updated_at and default status if not already present.
    """
    now_iso = _utcnow().isoformat()
    report_dict.setdefault("created_at", now_iso)
    report_dict["updated_at"] = now_iso
    report_dict.setdefault("status", ReportStatus.SUBMITTED.value)

    container = await get_data_container(DataContainers.REPORTS)
    await container.upsert_item(body=report_dict)
    logger.info(
        "Report upserted: id=%s user=%s species=%s",
        report_dict.get("id"),
        report_dict.get("user_id"),
        report_dict.get("scientific_name"),
    )


# ── explorations ──────────────────────────────────────────────────────────────

async def create_exploration(
    user_id: str,
    session_id: str,
    area_name: Optional[str] = None,
) -> ExplorationDocument:
    doc = ExplorationDocument(id=str(uuid.uuid4()), user_id=user_id, session_id=session_id, area_name=area_name)
    container = await get_data_container(DataContainers.EXPLORATIONS)
    await container.create_item(body=doc.to_cosmos())
    logger.info("Exploration created: id=%s user=%s", doc.id, user_id)
    return doc


async def get_exploration(exploration_id: str, user_id: str) -> Optional[ExplorationDocument]:
    container = await get_data_container(DataContainers.EXPLORATIONS)
    try:
        doc = await container.read_item(item=exploration_id, partition_key=user_id)
        return ExplorationDocument.from_cosmos(doc)
    except CosmosResourceNotFoundError:
        return None


async def append_observation(
    exploration_id: str,
    user_id: str,
    observation: ObservationRecord,
) -> None:
    doc = await get_exploration(exploration_id, user_id)
    if doc is None:
        raise ValueError(f"Exploration '{exploration_id}' not found for user '{user_id}'.")
    doc.observations.append(observation)
    if observation.species_name not in doc.species_found:
        doc.species_found.append(observation.species_name)
    doc.updated_at = _utcnow()
    container = await get_data_container(DataContainers.EXPLORATIONS)
    await container.upsert_item(body=doc.to_cosmos())


# ── encyclopedia ──────────────────────────────────────────────────────────────

def _normalise_species_id(species_name: str) -> str:
    return species_name.strip().lower().replace(" ", "_")


async def get_encyclopedia_entry(species_name: str) -> Optional[EncyclopediaDocument]:
    doc_id = _normalise_species_id(species_name)
    container = await get_data_container(DataContainers.ENCYCLOPEDIA)
    try:
        doc = await container.read_item(item=doc_id, partition_key=doc_id)
        return EncyclopediaDocument.from_cosmos(doc)
    except CosmosResourceNotFoundError:
        return None


async def upsert_encyclopedia_entry(doc: EncyclopediaDocument) -> None:
    doc.updated_at = _utcnow()
    container = await get_data_container(DataContainers.ENCYCLOPEDIA)
    await container.upsert_item(body=doc.to_cosmos())
    logger.info("Encyclopedia entry upserted: id=%s", doc.id)


# ── incidents ─────────────────────────────────────────────────────────────────

async def create_incident(
    session_id: str,
    user_id: str,
    description: str,
    **kwargs,
) -> IncidentDocument:
    incident_id = str(uuid.uuid4())
    doc = IncidentDocument(id=incident_id, session_id=session_id, user_id=user_id, description=description, **kwargs)
    container = await get_data_container(DataContainers.INCIDENTS)
    await container.create_item(body=doc.to_cosmos())
    logger.info("Incident created: id=%s user=%s", incident_id, user_id)
    return doc


async def get_incident(incident_id: str) -> Optional[IncidentDocument]:
    container = await get_data_container(DataContainers.INCIDENTS)
    try:
        doc = await container.read_item(item=incident_id, partition_key=incident_id)
        return IncidentDocument.from_cosmos(doc)
    except CosmosResourceNotFoundError:
        return None


async def update_incident_status(
    incident_id: str,
    status: IncidentStatus,
    resolved: bool = False,
) -> None:
    doc = await get_incident(incident_id)
    if doc is None:
        raise ValueError(f"Incident '{incident_id}' not found.")
    doc.status = status
    doc.updated_at = _utcnow()
    if resolved:
        doc.resolved_at = _utcnow()
    container = await get_data_container(DataContainers.INCIDENTS)
    await container.upsert_item(body=doc.to_cosmos())


# ── sos_events ────────────────────────────────────────────────────────────────

async def create_sos_event(
    user_id: str,
    session_id: str,
    latitude: float,
    longitude: float,
    message: str = "",
    device_id: Optional[str] = None,
) -> SOSEventDocument:
    sos_id = str(uuid.uuid4())
    doc = SOSEventDocument(
        id=sos_id, user_id=user_id, session_id=session_id,
        latitude=latitude, longitude=longitude,
        message=message, device_id=device_id,
    )
    container = await get_data_container(DataContainers.SOS_EVENTS)
    await container.create_item(body=doc.to_cosmos())
    logger.info("SOS created: id=%s user=%s device=%s", sos_id, user_id, device_id)
    return doc


async def get_sos_event(sos_id: str, user_id: str) -> Optional[SOSEventDocument]:
    container = await get_data_container(DataContainers.SOS_EVENTS)
    try:
        doc = await container.read_item(item=sos_id, partition_key=user_id)
        return SOSEventDocument.from_cosmos(doc)
    except CosmosResourceNotFoundError:
        return None


async def acknowledge_sos(sos_id: str, user_id: str) -> None:
    doc = await get_sos_event(sos_id, user_id)
    if doc is None:
        raise ValueError(f"SOS event '{sos_id}' not found for user '{user_id}'.")
    doc.status = SOSStatus.ACKNOWLEDGED
    doc.acknowledged_at = _utcnow()
    container = await get_data_container(DataContainers.SOS_EVENTS)
    await container.upsert_item(body=doc.to_cosmos())


async def resolve_sos(
    sos_id: str,
    user_id: str,
    linked_incident_id: Optional[str] = None,
) -> None:
    doc = await get_sos_event(sos_id, user_id)
    if doc is None:
        raise ValueError(f"SOS event '{sos_id}' not found for user '{user_id}'.")
    doc.status = SOSStatus.RESOLVED
    doc.resolved_at = _utcnow()
    if linked_incident_id:
        doc.linked_incident_id = linked_incident_id
    container = await get_data_container(DataContainers.SOS_EVENTS)
    await container.upsert_item(body=doc.to_cosmos())


async def add_notified_device(sos_id: str, user_id: str, device_id: str) -> None:
    """Record that a push notification was sent to ``device_id`` for this SOS."""
    doc = await get_sos_event(sos_id, user_id)
    if doc is None:
        raise ValueError(f"SOS event '{sos_id}' not found for user '{user_id}'.")
    if device_id not in doc.notified_devices:
        doc.notified_devices.append(device_id)
        container = await get_data_container(DataContainers.SOS_EVENTS)
        await container.upsert_item(body=doc.to_cosmos())

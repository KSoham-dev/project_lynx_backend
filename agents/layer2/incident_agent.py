"""
agents/layer2/incident_agent.py
================================
Layer 2 — Incident Agent

Triggered by ReporterAgent immediately after a report is generated.
Performs two primary operations in parallel:

1. log_incident()
      • Appends a full structured JSON entry to an Azure Append Blob
        (container: AZURE_INCIDENTS_CONTAINER_NAME, default: incident-logs).
        Blob name: <incident_id>.log
      • Saves a concise IncidentDocument overview (with the blob URL) to
        Cosmos DB prahari-data/incidents.

2. notify_logic_app()
      • HTTP POST to Azure Logic Apps to trigger an automated email
        containing the PDF report URL to the designated receiver.

Also exposes two stubs (ready for implementation):

3. get_ranger_details(user_id)
      • [STUB] Fetches the ranger's UserDocument from Cosmos DB and
        returns key profile fields including registered FCM device tokens.

4. send_push_notification(user_id, title, body, data)
      • [STUB] Sends an FCM web-push alert to all devices registered for
        the ranger.  Requires FCM service-account credentials to implement.

Environment variables
---------------------
AZURE_INCIDENTS_CONTAINER_NAME  – blob container for incident logs
                                   (default: "incident-logs")
LOGIC_APP_URL                   – override for the Logic Apps webhook URL
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from agents.enums import IncidentSeverity, IncidentStatus, IncidentType
from agents.state.containers import DataContainers
from agents.state.cosmos_client import get_data_container
from agents.state.data_schemas import IncidentDocument

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Logic Apps webhook URL — overridable via env var for key rotation
_DEFAULT_LOGIC_APP_URL = (
    "https://prod-32.southeastasia.logic.azure.com/workflows/"
    "6e0d4a00506e488e9c711b6de6bdd226/triggers/"
    "When_an_HTTP_request_is_received/paths/invoke"
    "?api-version=2016-10-01"
    "&sp=%2Ftriggers%2FWhen_an_HTTP_request_is_received%2Frun"
    "&sv=1.0"
    "&sig=yfYY5mznDE9A2sXy_G2ScWvANRR8KH0aw2eIKs1SxN8"
)

_NOTIFICATION_EMAIL = "sohamkulkarni709@gmail.com"


def _logic_app_url() -> str:
    return os.getenv("LOGIC_APP_URL", _DEFAULT_LOGIC_APP_URL)


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ── Severity helper ───────────────────────────────────────────────────────────

_SEV_MAP: dict[str, IncidentSeverity] = {
    "critical": IncidentSeverity.CRITICAL,
    "high":     IncidentSeverity.HIGH,
    "medium":   IncidentSeverity.MEDIUM,
    "low":      IncidentSeverity.LOW,
}


def _parse_severity(value: Optional[str]) -> IncidentSeverity:
    return _SEV_MAP.get((value or "").lower(), IncidentSeverity.MEDIUM)


# ── Append Blob (sync — run via asyncio.to_thread) ────────────────────────────

def _write_append_blob_sync(incident_id: str, log_payload: dict[str, Any]) -> Optional[str]:
    """
    Append log_payload as a JSON line to an Azure Append Blob.

    Blob name : <incident_id>.log
    Container : AZURE_INCIDENTS_CONTAINER_NAME (default: incident-logs)

    Creates the container and blob on first use.
    Returns the blob URL on success, None on failure.
    Runs synchronously — always call via asyncio.to_thread.
    """
    from azure.core.exceptions import ResourceExistsError
    from pipeline.species_traits import _build_blob_service_client

    container_name = os.getenv("AZURE_INCIDENTS_CONTAINER_NAME", "incident-logs")
    blob_name = f"{incident_id}.log"
    log_line  = json.dumps(log_payload, ensure_ascii=False, default=str) + "\n"

    try:
        service          = _build_blob_service_client()
        container_client = service.get_container_client(container_name)

        # Best-effort container creation — no-op if already exists
        try:
            container_client.create_container()
            logger.info("[incident] Created incidents log container: %s", container_name)
        except Exception:
            pass  # container already exists

        blob_client = container_client.get_blob_client(blob_name)

        # Create the append blob on first write; skip if already created
        try:
            blob_client.create_append_blob()
        except ResourceExistsError:
            pass  # blob already exists — just append

        blob_client.append_block(log_line.encode("utf-8"))
        url = blob_client.url
        logger.info("[incident] Incident log written to: %s", url)
        return url

    except Exception as exc:
        logger.error(
            "[incident] Append blob write failed for incident_id=%s: %s",
            incident_id, exc, exc_info=True,
        )
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Incident Agent
# ═══════════════════════════════════════════════════════════════════════════════

class IncidentAgent:
    """
    Post-report incident logging and notification agent.

    Call IncidentAgent().run(report_response, request) immediately after
    ReporterAgent.run() returns.  Errors are caught and logged — the caller's
    response is never affected.
    """

    # ── 1. Log Incident ───────────────────────────────────────────────────────

    async def log_incident(
        self,
        incident_id: str,
        report_response: dict[str, Any],
        request: Any,  # AgentRequest — kept as Any to avoid circular import
        animal_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Write a full structured log entry to an Azure Append Blob, then save
        an IncidentDocument overview to Cosmos DB prahari-data/incidents.

        Returns the append blob URL (or None if the blob write failed).
        """
        location: dict = report_response.get("location") or {}

        # ── Full log payload (everything available) ──────────────────────────────
        log_payload: dict[str, Any] = {
            "incident_id":    incident_id,
            "logged_at":      _utcnow_iso(),
            "animal_name":    animal_name,
            # Report linkage
            "report_id":      report_response.get("report_id"),
            "pdf_url":        report_response.get("pdf_url"),
            "submitted_at":   report_response.get("submitted_at"),
            # Request context
            "user_id":        request.user_id,
            "session_id":     request.session_id,
            "message":        request.message,
            # Species identification
            "scientific_name":       report_response.get("scientific_name"),
            "common_names":          report_response.get("common_names"),
            "genus":                 report_response.get("genus"),
            "family":                report_response.get("family"),
            "confidence":            report_response.get("confidence"),
            "identification_source": report_response.get("identification_source"),
            # IUCN conservation
            "red_list_category": report_response.get("red_list_category"),
            "red_list_code":     report_response.get("red_list_code"),
            "population_trend":  report_response.get("population_trend"),
            "rationale":         report_response.get("rationale"),
            "iucn_url":          report_response.get("iucn_url"),
            # Risk & threat
            "risk_level":   report_response.get("risk_level"),
            "threat_level": report_response.get("threat_level"),
            "severity":     report_response.get("severity"),
            # Physical traits
            "length":         report_response.get("length"),
            "lifespan_years": report_response.get("lifespan_years"),
            # Safety guidance
            "user_safety_precautions": report_response.get("user_safety_precautions"),
            # GPS & location
            "latitude":           location.get("latitude"),
            "longitude":          location.get("longitude"),
            "district":           location.get("district"),
            "state":              location.get("state"),
            "country":            location.get("country"),
            "protected_area":     location.get("protected_area"),
            "formatted_location": location.get("formatted"),
            # Media
            "image_url":               report_response.get("image_url"),
            "inaturalist_photo_credit": report_response.get("inaturalist_photo_credit"),
        }

        # Write full log to append blob (synchronous call in a thread)
        log_blob_url: Optional[str] = await asyncio.to_thread(
            _write_append_blob_sync, incident_id, log_payload
        )

        # ── Cosmos DB overview (concise IncidentDocument) ────────────────────
        species_name  = report_response.get("scientific_name") or "Unknown"
        risk_level    = report_response.get("risk_level") or "Unknown"
        formatted_loc = (
            location.get("formatted")
            or (
                f"{location.get('district')}, {location.get('state')}"
                if location.get("district")
                else "Unknown location"
            )
        )
        description = (
            f"{species_name} ({risk_level} risk) sighted at {formatted_loc}. "
            f"Red list: {report_response.get('red_list_category') or 'Unknown'}. "
            f"Report ID: {report_response.get('report_id', '—')}."
        )

        try:
            doc = IncidentDocument(
                id=incident_id,
                session_id=request.session_id,
                user_id=request.user_id,
                description=description,
                animal_name=animal_name,
                severity=_parse_severity(report_response.get("severity")),
                latitude=location.get("latitude"),
                longitude=location.get("longitude"),
                location_context=location,
                linked_report_id=report_response.get("report_id"),
                log_blob_url=log_blob_url,
            )
            container = await get_data_container(DataContainers.INCIDENTS)
            await container.create_item(body=doc.to_cosmos())
            logger.info(
                "[incident] IncidentDocument saved: incident_id=%s linked_report_id=%s",
                incident_id, report_response.get("report_id"),
            )
        except Exception as exc:
            logger.error(
                "[incident] Cosmos incident save failed for incident_id=%s: %s",
                incident_id, exc, exc_info=True,
            )

        return log_blob_url

    # ── 2. Notify Logic App ───────────────────────────────────────────────────

    async def notify_logic_app(
        self,
        pdf_url: Optional[str],
        animal_name: Optional[str] = None,
    ) -> bool:
        """
        POST to Azure Logic Apps to trigger an automated email with the PDF.

        Payload: { "pdf_blob_url": <pdf_url>, "receiver_email": <email>, "animal_name": <name> }
        Returns True on HTTP 2xx, False otherwise.
        """
        if not pdf_url:
            logger.warning("[incident] notify_logic_app: pdf_url is empty — skipping")
            return False

        payload = {
            "pdf_blob_url":   pdf_url,
            "receiver_email": _NOTIFICATION_EMAIL,
            "animal_name":    animal_name,
        }

        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    _logic_app_url(),
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
            if resp.is_success:
                logger.info(
                    "[incident] Logic App notified: status=%d pdf_url=%s receiver=%s",
                    resp.status_code, pdf_url, _NOTIFICATION_EMAIL,
                )
                return True
            else:
                logger.warning(
                    "[incident] Logic App returned non-2xx: status=%d body=%.300s",
                    resp.status_code, resp.text,
                )
                return False

        except Exception as exc:
            logger.error(
                "[incident] Logic App notification failed: %s", exc, exc_info=True
            )
            return False

    # ── 3. Get Ranger Details (stub) ─────────────────────────────────────────

    async def get_ranger_details(self, user_id: str) -> dict[str, Any]:
        """
        [STUB] Fetch ranger profile from Cosmos DB prahari-data/users.

        Returns a dict with the ranger's name, role, assigned forest zone,
        district, state, and a flattened list of FCM device tokens that can
        be used by send_push_notification().

        Intended implementation:
            from agents.state.data_store import get_user
            doc = await get_user(user_id)
            return {
                "user_id":    doc.user_id,
                "name":       doc.name,
                "role":       doc.role.value,
                "forest_zone": doc.forest_zone,
                "district":   doc.district,
                "state":      doc.state,
                "fcm_tokens": [d.fcm_token for d in doc.devices if d.fcm_token],
            }
        """
        logger.info("[incident] get_ranger_details called: user_id=%s (stub)", user_id)
        return {
            "user_id":     user_id,
            "name":        None,
            "role":        None,
            "forest_zone": None,
            "district":    None,
            "state":       None,
            "fcm_tokens":  [],
        }

    # ── 4. Send Push Notification (stub) ─────────────────────────────────────

    async def send_push_notification(
        self,
        user_id: str,
        title: str,
        body: str,
        data: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        [STUB] Send an FCM web-push notification to all devices registered
        for the given ranger.

        Intended implementation:
          1. Call get_ranger_details(user_id) to retrieve FCM tokens.
          2. For each token POST to Firebase Cloud Messaging v1 API:
               POST https://fcm.googleapis.com/v1/projects/<PROJECT>/messages:send
               Authorization: Bearer <service-account-token>
               Body: { "message": { "token": <fcm_token>,
                                    "notification": { "title": title, "body": body },
                                    "data": data or {} } }
          3. Return True if at least one token delivery succeeded.

        Requires FIREBASE_PROJECT_ID and a GCP service-account key / Workload
        Identity to implement.
        """
        logger.info(
            "[incident] send_push_notification called: user_id=%s title=%r (stub)",
            user_id, title,
        )
        return False

    # ── Orchestrate ───────────────────────────────────────────────────────────

    async def run(
        self,
        report_response: dict[str, Any],
        request: Any,  # AgentRequest
    ) -> None:
        """
        Orchestrate incident logging and Logic App notification.

        Both operations run concurrently.  Called by ReporterAgent after its
        run() returns.  Failures are caught and logged — the caller is not
        affected.

        get_ranger_details() and send_push_notification() are available as
        stubs — wire them in here once FCM credentials are provisioned.
        """
        incident_id = str(uuid.uuid4())
        logger.info(
            "[incident] START incident_id=%s report_id=%s",
            incident_id, report_response.get("report_id"),
        )

        pdf_url = report_response.get("pdf_url")

        # Resolve animal_name once here so both operations share the same value
        _common_names: list = report_response.get("common_names") or []
        animal_name: Optional[str] = (
            _common_names[0] if _common_names else None
        ) or report_response.get("scientific_name") or None

        # Run log_incident and notify_logic_app concurrently
        await asyncio.gather(
            self.log_incident(incident_id, report_response, request, animal_name),
            self.notify_logic_app(pdf_url, animal_name),
            return_exceptions=True,
        )

        logger.info("[incident] DONE incident_id=%s", incident_id)

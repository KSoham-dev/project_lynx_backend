"""
agents/state/__init__.py
=========================
Top-level shared state package — Cosmos DB clients, schemas, and CRUD stores.

This package is shared across ALL agent layers (Layer 0 → Layer 3).
Never import from agents.layer0.state — that path no longer exists.

Public surface
--------------
From agents.state.session_store  (prahari-state DB):
    create_session, get_session, upsert_session, append_message

From agents.state.data_store     (prahari-data DB):
    get_user, upsert_user, register_device
    create_report, get_report, update_report_status
    create_exploration, get_exploration, append_observation
    get_encyclopedia_entry, upsert_encyclopedia_entry
    create_incident, get_incident, update_incident_status
    create_sos_event, get_sos_event, acknowledge_sos, resolve_sos, add_notified_device

From agents.state.cosmos_client:
    get_state_container, get_data_container, close_cosmos_clients

Schemas:
    agents.state.schemas      → SessionDocument, MessageRecord, AgentTraceRecord,
                                 ImageAnalysisCacheDoc, SpeciesContextCacheDoc, LocationCacheDoc
    agents.state.data_schemas → UserDocument, DeviceInfo, ReportDocument,
                                 ExplorationDocument, EncyclopediaDocument,
                                 IncidentDocument, SOSEventDocument

Constants:
    agents.state.containers   → StateContainers, DataContainers
"""

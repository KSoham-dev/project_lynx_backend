"""
agents/layer0/state/containers.py
===================================
Cosmos DB container name constants for both databases.

Never hardcode container names in application code — import from here.

Database layout
---------------
prahari-state  (agent / LangChain state — COSMOS_STATE_CONNECTION_STRING)
    sessions
    agent_traces
    image_analysis_cache
    species_context_cache
    location_cache

prahari-data   (user & application data — COSMOS_DATA_CONNECTION_STRING)
    users
    reports
    explorations
    encyclopedia
    incidents
    sos_events
"""

from __future__ import annotations


class StateContainers:
    """Container names in the *prahari-state* database."""
    SESSIONS              = "sessions"
    AGENT_TRACES          = "agent_traces"
    IMAGE_ANALYSIS_CACHE  = "image_analysis_cache"
    SPECIES_CONTEXT_CACHE = "species_context_cache"
    LOCATION_CACHE        = "location_cache"


class DataContainers:
    """Container names in the *prahari-data* database."""
    USERS        = "users"
    REPORTS      = "reports"
    EXPLORATIONS = "explorations"
    ENCYCLOPEDIA = "encyclopedia"
    INCIDENTS    = "incidents"
    SOS_EVENTS   = "sos_events"

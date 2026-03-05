"""
agents/state/cosmos_client.py
==============================
Async Azure Cosmos DB clients for both databases — singleton pattern.

Two separate CosmosClient instances are maintained:
    _state_store  → prahari-state  (agent/LangChain state)
    _data_store   → prahari-data   (user & application data)

Reads connection strings from :func:`agents.config.get_shared_settings`
(shared across all agent layers).

Usage
-----
    from agents.state.cosmos_client import get_state_container, get_data_container
    from agents.state.containers import StateContainers, DataContainers

    container = await get_state_container(StateContainers.SESSIONS)
    await container.upsert_item(...)
"""

from __future__ import annotations

import logging
from typing import Optional

from azure.cosmos.aio import CosmosClient, ContainerProxy

from agents.config import get_shared_settings

logger = logging.getLogger(__name__)

# ── Module-level singleton stores ─────────────────────────────────────────────

_state_store: dict = {}
_data_store:  dict = {}


# ── Internal initialiser ──────────────────────────────────────────────────────

async def _get_client(
    connection_string: str,
    db_name: str,
    client_store: dict,
    label: str,
) -> CosmosClient:
    """Return (or initialise) a cached CosmosClient from a connection string."""
    if client_store.get("client") is not None:
        return client_store["client"]

    if not connection_string:
        raise RuntimeError(
            f"{label} connection string is not set. "
            "Set the corresponding env var before starting the app."
        )

    logger.info("Initialising Cosmos DB client: db=%s", db_name)
    client = CosmosClient.from_connection_string(connection_string)
    client_store["client"] = client
    logger.info("Cosmos DB client ready: db=%s", db_name)
    return client


# ── Public container accessors ────────────────────────────────────────────────

async def get_state_container(container_name: str) -> ContainerProxy:
    """
    Return an async ContainerProxy for a container in *prahari-state*.

    Parameters
    ----------
    container_name:
        One of :class:`agents.state.containers.StateContainers` constants.
    """
    s = get_shared_settings()
    client = await _get_client(
        s.cosmos_state_connection_string,
        s.cosmos_state_database,
        _state_store,
        "COSMOS_STATE_CONNECTION_STRING",
    )
    db = client.get_database_client(s.cosmos_state_database)
    return db.get_container_client(container_name)


async def get_data_container(container_name: str) -> ContainerProxy:
    """
    Return an async ContainerProxy for a container in *prahari-data*.

    Parameters
    ----------
    container_name:
        One of :class:`agents.state.containers.DataContainers` constants.
    """
    s = get_shared_settings()
    client = await _get_client(
        s.cosmos_data_connection_string,
        s.cosmos_data_database,
        _data_store,
        "COSMOS_DATA_CONNECTION_STRING",
    )
    db = client.get_database_client(s.cosmos_data_database)
    return db.get_container_client(container_name)


# ── Cleanup ───────────────────────────────────────────────────────────────────

async def close_cosmos_clients() -> None:
    """
    Close both Cosmos DB HTTP connection pools.

    Call during FastAPI lifespan shutdown.
    """
    global _state_store, _data_store  # noqa: PLW0603

    for store, label in ((_state_store, "state"), (_data_store, "data")):
        client = store.get("client")
        if client is not None:
            await client.close()
            logger.info("Cosmos DB '%s' client closed.", label)

    _state_store.clear()
    _data_store.clear()

"""
scripts/provision_cosmos.py
============================
One-time script to create both Cosmos DB databases and all containers.

Run once before starting the app:
    source .venv/bin/activate
    python scripts/provision_cosmos.py

Reads COSMOS_STATE_CONNECTION_STRING and COSMOS_DATA_CONNECTION_STRING from .env.
Safe to re-run — existing databases/containers are left untouched.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from azure.cosmos.aio import CosmosClient
from azure.cosmos.exceptions import CosmosResourceExistsError

from agents.config import get_shared_settings


# ── Container specs ───────────────────────────────────────────────────────────

STATE_CONTAINERS = [
    # name                      partition_key   ttl (None = disabled, -1 = item-level only)
    ("sessions",                "/id",          None),
    ("agent_traces",            "/session_id",  None),
    ("image_analysis_cache",    "/id",          -1),   # item TTL via _ttl field (7d)
    ("species_context_cache",   "/id",          -1),   # item TTL via _ttl field (24h)
    ("location_cache",          "/id",          -1),   # item TTL via _ttl field (30d)
]

DATA_CONTAINERS = [
    ("users",        "/id",       None),
    ("reports",      "/user_id",  None),
    ("explorations", "/user_id",  None),
    ("encyclopedia", "/id",       None),
    ("incidents",    "/id",       None),
    ("sos_events",   "/user_id",  None),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

async def ensure_database(client: CosmosClient, db_name: str) -> None:
    try:
        await client.create_database(id=db_name)
        print(f"  ✓ Created database: {db_name}")
    except CosmosResourceExistsError:
        print(f"  · Database already exists: {db_name}")


async def ensure_container(
    client: CosmosClient,
    db_name: str,
    container_name: str,
    partition_key: str,
    default_ttl: int | None,
) -> None:
    db = client.get_database_client(db_name)
    kwargs: dict = {
        "id": container_name,
        "partition_key": {"paths": [partition_key], "kind": "Hash"},
    }
    if default_ttl is not None:
        kwargs["default_ttl"] = default_ttl

    try:
        await db.create_container(**kwargs)
        ttl_info = f" (TTL={'item-level' if default_ttl == -1 else default_ttl})" if default_ttl else ""
        print(f"    ✓ {container_name}  pk={partition_key}{ttl_info}")
    except CosmosResourceExistsError:
        print(f"    · {container_name}  (already exists)")


async def provision() -> None:
    settings = get_shared_settings()

    if not settings.cosmos_state_connection_string:
        print("ERROR: COSMOS_STATE_CONNECTION_STRING is not set in .env")
        sys.exit(1)
    if not settings.cosmos_data_connection_string:
        print("ERROR: COSMOS_DATA_CONNECTION_STRING is not set in .env")
        sys.exit(1)

    # ── prahari-state ─────────────────────────────────────────────────────────
    print(f"\n── {settings.cosmos_state_database} ──")
    async with CosmosClient.from_connection_string(settings.cosmos_state_connection_string) as client:
        await ensure_database(client, settings.cosmos_state_database)
        for name, pk, ttl in STATE_CONTAINERS:
            await ensure_container(client, settings.cosmos_state_database, name, pk, ttl)

    # ── prahari-data ──────────────────────────────────────────────────────────
    print(f"\n── {settings.cosmos_data_database} ──")
    async with CosmosClient.from_connection_string(settings.cosmos_data_connection_string) as client:
        await ensure_database(client, settings.cosmos_data_database)
        for name, pk, ttl in DATA_CONTAINERS:
            await ensure_container(client, settings.cosmos_data_database, name, pk, ttl)

    print("\n✅ Provisioning complete.\n")


if __name__ == "__main__":
    asyncio.run(provision())

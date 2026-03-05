"""
agents/config.py
=================
Shared infrastructure configuration used across **all agent layers**.

Currently covers Azure Cosmos DB — the two database connection strings
and their database name defaults.

Usage (from any layer)
----------------------
    from agents.config import get_shared_settings
    s = get_shared_settings()
    print(s.cosmos_state_connection_string)
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class SharedSettings(BaseSettings):
    """Shared infrastructure settings (Cosmos DB) read from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ── Azure Cosmos DB — State Management DB (prahari-state) ────────────────
    # Format: AccountEndpoint=https://...;AccountKey=...;
    cosmos_state_connection_string: str = ""
    cosmos_state_database: str = "prahari-state"

    # ── Azure Cosmos DB — Application Data DB (prahari-data) ─────────────────
    cosmos_data_connection_string: str = ""
    cosmos_data_database: str = "prahari-data"


@lru_cache(maxsize=1)
def get_shared_settings() -> SharedSettings:
    """Return a cached singleton SharedSettings instance."""
    return SharedSettings()

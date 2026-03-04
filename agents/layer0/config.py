"""
agents/layer0/config.py
========================
Layer 0–specific configuration (Azure OpenAI only).

Cosmos DB settings have moved to agents/config.py (shared across all layers).

Usage
-----
    from agents.layer0.config import get_settings
    s = get_settings()
    print(s.azure_openai_deployment)
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Layer0Settings(BaseSettings):
    """Azure OpenAI settings for the Layer 0 Orchestrator."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment: str = "gpt-5-mini"
    azure_openai_api_version: str = "2024-10-21"


@lru_cache(maxsize=1)
def get_settings() -> Layer0Settings:
    """Return a cached singleton Layer0Settings instance."""
    return Layer0Settings()

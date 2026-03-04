"""
agents/layer1/gpt_vision.py
============================
GPT-5-mini vision fallback for species identification.

Uses raw AsyncAzureOpenAI client with api_version="2025-01-01-preview"
and response_format={"type": "json_object"} to guarantee clean JSON output.
"""
from __future__ import annotations

import json
import logging

from openai import AsyncAzureOpenAI

from agents.layer0.config import get_settings
from agents.layer1.schemas import IdentificationResult

logger = logging.getLogger(__name__)

SPECIES_ID_SYSTEM_PROMPT = (
    "You are an expert Zoologist, Famous for correctly identifying any type of terrestrial species from an image specific to Indian subcontinent. You give the species name and common name in your response only in json format. Donot give any additional information in responses. The JSON should contain 'species_name' and 'common_name' fields."
)


def _normalise_scientific_name(raw: str) -> str:
    """
    Ensure the scientific name is in strict 'Genus species' (two-word) format.

    - Strips extra words beyond genus + epithet.
    - Capitalises genus, lowercases epithet.
    - Returns 'Unknown' if the result is not a valid two-part name.
    """
    parts = raw.strip().split()
    if len(parts) < 2:
        return "Unknown"
    genus = parts[0].capitalize()
    epithet = parts[1].lower()
    # Reject placeholder tokens
    if genus.lower() in {"unknown", "unidentified", "none", ""} or epithet in {"unknown", "sp.", "sp", ""}:
        return "Unknown"
    return f"{genus} {epithet}"


def _build_client() -> AsyncAzureOpenAI:
    s = get_settings()
    return AsyncAzureOpenAI(
        azure_endpoint=s.azure_openai_endpoint,
        api_key=s.azure_openai_api_key,
        api_version="2025-01-01-preview",
    )


async def gpt_identify(image_url: str) -> IdentificationResult:
    """
    Send the image URL to GPT-5-mini (vision) and parse the JSON response.

    Parameters
    ----------
    image_url:
        Publicly accessible URL of the image.

    Returns
    -------
    IdentificationResult
        scientific_name from GPT, confidence=0.0, source="gpt" | "gpt_error".
    """
    s = get_settings()
    client = _build_client()

    messages = [
        {
            "role": "system",
            "content": SPECIES_ID_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Give me the scientific name of the species in this image."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    try:
        logger.info("[gpt_vision] Calling GPT-5-mini vision: url=%.80s", image_url)
        completion = await client.chat.completions.create(
            model=s.azure_openai_deployment,
            messages=messages,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or ""
        logger.debug("[gpt_vision] raw response: %r", raw[:500])

        data: dict = json.loads(raw)
        species_name: str = _normalise_scientific_name(
            data.get("species_name") or data.get("scientific_name") or ""
        )
        common_name: str = data.get("common_name") or ""
        logger.info("[gpt_vision] identified species=%r common_name=%r", species_name, common_name)
        return IdentificationResult(
            scientific_name=species_name,
            confidence=0.0,
            source="gpt",
            common_name=common_name or None,
        )

    except Exception as exc:
        logger.error("[gpt_vision] GPT-5-mini vision call failed: %s", exc, exc_info=True)
        return IdentificationResult(
            scientific_name="Unknown",
            confidence=0.0,
            source="gpt_error",
        )

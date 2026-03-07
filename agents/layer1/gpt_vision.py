"""
agents/layer1/gpt_vision.py
============================
GPT-5-mini vision fallback for species identification.

Uses raw AsyncAzureOpenAI client with api_version="2025-01-01-preview"
and response_format={"type": "json_object"} to guarantee clean JSON output.
"""
from __future__ import annotations

import asyncio
import json
import logging

from openai import AsyncAzureOpenAI

from agents.layer0.config import get_settings
from agents.layer1.schemas import IdentificationResult

logger = logging.getLogger(__name__)

# Minimum GPT-reported confidence (0–100) required to accept an identification.
# Below this threshold the result is demoted to "Unknown".
_MIN_GPT_CONFIDENCE = 60

SPECIES_ID_SYSTEM_PROMPT = """\
You are a professional field zoologist specialising in the terrestrial fauna of the Indian subcontinent.
Your sole task is to identify the species visible in the provided photograph.

Return ONLY a single valid JSON object — no markdown, no prose, no extra keys — with exactly these fields:

  "scientific_name": two-word Latin binomial ("Genus species"), or the string "Unknown"
  "common_name":     English common name, or "" if unknown
  "confidence":      integer from 0 to 100 representing how certain you are of the identification

Guidelines for confidence:
- 90–100  You can clearly see diagnostic features and are certain of the species.
- 70–89   You are fairly confident but some features are partially obscured.
- 50–69   The image is ambiguous or the animal is partially hidden; this is your best guess.
- 0–49    You cannot reliably identify the species; set scientific_name to "Unknown".

Be honest and calibrated — do NOT inflate confidence to seem helpful.
If the image contains no animal, if the species is unrecognisable, or if you are not confident,
set scientific_name to "Unknown" and confidence to 0.

Example output:
{"scientific_name": "Panthera pardus", "common_name": "Indian leopard", "confidence": 88}
"""


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
        timeout=60.0,
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
                {
                    "type": "text",
                    "text": (
                        "Identify the species in this image. "
                        "Respond with a JSON object containing scientific_name, common_name, and confidence (0–100). "
                        "Be accurate with the confidence score — it will be used to decide whether to trust your answer."
                    ),
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    try:
        logger.info(
            "[gpt_vision] Calling GPT vision: url=%.80s model=%s api_version=%s endpoint=%.60s",
            image_url, s.azure_openai_deployment, s.azure_openai_api_version, s.azure_openai_endpoint,
        )
        completion = await asyncio.wait_for(
            client.chat.completions.create(
                model=s.azure_openai_deployment,
                messages=messages,
                response_format={"type": "json_object"},
            ),
            timeout=90.0,
        )
        raw = completion.choices[0].message.content or ""
        logger.info("[gpt_vision] raw response: %r", raw[:500])

        data: dict = json.loads(raw)
        species_name: str = _normalise_scientific_name(
            data.get("species_name") or data.get("scientific_name") or ""
        )
        common_name: str = data.get("common_name") or ""

        # GPT reports confidence as 0–100; normalise to 0.0–1.0
        raw_confidence = data.get("confidence", 0)
        try:
            raw_confidence = int(raw_confidence)
        except (TypeError, ValueError):
            raw_confidence = 0
        confidence: float = max(0.0, min(raw_confidence, 100)) / 100.0

        # Reject low-confidence identifications rather than hallucinate
        if raw_confidence < _MIN_GPT_CONFIDENCE:
            logger.info(
                "[gpt_vision] confidence %d < %d — demoting to Unknown (was %r)",
                raw_confidence, _MIN_GPT_CONFIDENCE, species_name,
            )
            species_name = "Unknown"

        logger.info(
            "[gpt_vision] identified species=%r common_name=%r confidence=%d",
            species_name, common_name, raw_confidence,
        )
        return IdentificationResult(
            scientific_name=species_name,
            confidence=confidence,
            source="gpt",
            common_name=common_name or None,
        )

    except asyncio.TimeoutError:
        logger.error(
            "[gpt_vision] TIMEOUT after 90s — Azure OpenAI did not respond: "
            "url=%.80s model=%s endpoint=%.60s",
            image_url, s.azure_openai_deployment, s.azure_openai_endpoint,
        )
        return IdentificationResult(
            scientific_name="Unknown", confidence=0.0, source="gpt_error",
            error=f"Timeout after 90s — Azure OpenAI ({s.azure_openai_deployment}) did not respond",
        )

    except json.JSONDecodeError as exc:
        logger.error(
            "[gpt_vision] JSON parse failed — model returned non-JSON: %s | raw=%.500s",
            exc, locals().get("raw", "<no response captured>"),
        )
        return IdentificationResult(
            scientific_name="Unknown", confidence=0.0, source="gpt_error",
            error=f"JSON parse error: {exc}",
        )

    except Exception as exc:
        # Capture HTTP status and Azure error code when available (openai SDK wraps them)
        status_code = getattr(exc, "status_code", None)
        error_code  = getattr(getattr(exc, "body", None), "get", lambda *_: None)("code") or getattr(exc, "code", None)
        logger.error(
            "[gpt_vision] GPT vision call failed: %s | type=%s status=%s code=%s url=%.80s",
            exc, type(exc).__name__, status_code, error_code, image_url,
            exc_info=True,
        )
        return IdentificationResult(
            scientific_name="Unknown", confidence=0.0, source="gpt_error",
            error=f"{type(exc).__name__}: {exc}" + (f" (HTTP {status_code})" if status_code else ""),
        )

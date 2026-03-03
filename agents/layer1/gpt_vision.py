"""
agents/layer1/gpt_vision.py
============================
GPT-4o-mini vision fallback for species identification.

Replaces GptPlugin.call_gpt from Identifier.txt using AzureChatOpenAI
multimodal messages (image URL embedded in HumanMessage content list).
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from agents.layer1.schemas import IdentificationResult

if TYPE_CHECKING:
    from langchain_openai import AzureChatOpenAI

logger = logging.getLogger(__name__)

SPECIES_ID_SYSTEM_PROMPT = """\
You are an expert zoologist specialising in Indian wildlife.

Your task: identify the species in the image provided.

Respond with ONLY valid JSON in this exact format:
{
  "species_name": "<Genus species — e.g. Panthera tigris>",
  "common_name": "<English common name>",
  "confidence_note": "<brief one-sentence reason for your identification>"
}

If you cannot identify the species, respond:
{"species_name": "Unknown", "common_name": "Unknown", "confidence_note": "Insufficient visual detail."}

Do NOT include any text outside the JSON object.\
"""


async def gpt_identify(image_url: str, llm: "AzureChatOpenAI") -> IdentificationResult:
    """
    Send the image URL to GPT-4o-mini (vision) and parse the JSON response.

    Parameters
    ----------
    image_url:
        Publicly accessible URL of the image.
    llm:
        Shared AzureChatOpenAI instance (gpt-4o-mini supports vision).

    Returns
    -------
    IdentificationResult
        scientific_name from GPT, confidence=0.0, source="gpt" | "gpt_error".
    """
    messages = [
        SystemMessage(content=SPECIES_ID_SYSTEM_PROMPT),
        HumanMessage(content=[
            {"type": "text", "text": "Identify the wildlife species in this image."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]),
    ]

    try:
        response = await llm.ainvoke(messages)
        raw = response.content.strip()

        # Strip markdown code fences if GPT wraps JSON in ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data: dict = json.loads(raw)
        species_name: str = (
            data.get("species_name")
            or data.get("scientific_name")
            or "Unknown"
        )
        logger.info("GPT vision identified species as %r", species_name)
        return IdentificationResult(
            scientific_name=species_name,
            confidence=0.0,
            source="gpt",
        )

    except Exception as exc:
        logger.error("GPT vision fallback failed: %s", exc)
        return IdentificationResult(
            scientific_name="Unknown",
            confidence=0.0,
            source="gpt_error",
        )

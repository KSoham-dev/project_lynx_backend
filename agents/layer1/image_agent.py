"""
agents/layer1/image_agent.py
=============================
Layer 1 — ImageAnalysisAgent

Single entry point that orchestrates:
  1. Cosmos DB cache check (TTL 7 days)
  2. SpeciesNet prediction (via injected model)
  3. Relevancy check (detection-based)
  4. Species identification (SpeciesNet label or GPT-5-mini vision)
  5. Cache write

Public API
----------
    # In main.py lifespan (after SpeciesNet loads):
    from agents.layer1.image_agent import set_speciesnet_model
    set_speciesnet_model(model)

    # In orchestrator tool:
    from agents.layer1.image_agent import get_image_agent
    result = await get_image_agent().run(image_url, latitude, longitude)
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Optional, TYPE_CHECKING

from agents.layer0.config import get_settings
from agents.layer1.identifier import identify_species
from agents.layer1.relevancy import check_relevancy
from agents.layer1.schemas import ImageAnalysisResult
from agents.state.containers import StateContainers
from agents.state.cosmos_client import get_state_container

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Module-level SpeciesNet model holder ──────────────────────────────────────

_speciesnet_model = None


def set_speciesnet_model(model) -> None:
    """Inject the loaded SpeciesNet model. Call once from main.py lifespan."""
    global _speciesnet_model  # noqa: PLW0603
    _speciesnet_model = model
    logger.info("SpeciesNet model registered with Layer 1 ImageAnalysisAgent.")


def _get_model():
    if _speciesnet_model is None:
        raise RuntimeError(
            "SpeciesNet model not loaded. "
            "Call set_speciesnet_model() in the FastAPI lifespan before using this agent."
        )
    return _speciesnet_model


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_key(image_url: str) -> str:
    """Stable, URL-safe cache document ID derived from the image URL."""
    return hashlib.md5(image_url.encode()).hexdigest()  # noqa: S324


async def _read_cache(image_url: str) -> Optional[ImageAnalysisResult]:
    """Return cached ImageAnalysisResult if it exists, else None."""
    doc_id = _cache_key(image_url)
    try:
        container = await get_state_container(StateContainers.IMAGE_ANALYSIS_CACHE)
        doc = await container.read_item(item=doc_id, partition_key=doc_id)
        result = ImageAnalysisResult.model_validate(doc.get("result", {}))
        result.cached = True
        logger.info("Cache HIT for image: key=%s", doc_id)
        return result
    except Exception as exc:
        logger.debug("Cache MISS for image: key=%s reason=%s", doc_id, exc)
        return None


async def _write_cache(image_url: str, result: ImageAnalysisResult, ttl_seconds: int = 604_800) -> None:
    """Write ImageAnalysisResult to Cosmos image_analysis_cache (TTL 7 days = 604800s)."""
    doc_id = _cache_key(image_url)
    doc = {
        "id": doc_id,
        "image_url": image_url,
        "result": result.model_dump(mode="json"),
        "ttl": ttl_seconds,
    }
    try:
        container = await get_state_container(StateContainers.IMAGE_ANALYSIS_CACHE)
        await container.upsert_item(body=doc)
        logger.info("Cache WRITE OK: key=%s ttl=%ds", doc_id, ttl_seconds)
    except Exception as exc:
        # Cache write failure is non-fatal — log and continue
        logger.warning("Cache write failed for %s: %s", doc_id, exc, exc_info=True)


# ── Agent ─────────────────────────────────────────────────────────────────────

class ImageAnalysisAgent:
    """Orchestrates the full Layer 1 image analysis pipeline."""

    async def run(
        self,
        image_url: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> ImageAnalysisResult:
        """
        Run the full image analysis pipeline.

        Returns
        -------
        ImageAnalysisResult
            Serialise with ``.model_dump_json()`` for the LangChain tool response.
        """
        logger.info("ImageAnalysisAgent.run: url=%s lat=%s lon=%s", image_url, latitude, longitude)

        # ── 1. Cache check ────────────────────────────────────────────────────
        cached = await _read_cache(image_url)
        if cached:
            return cached

        # ── 2. SpeciesNet predict (sync → thread) ─────────────────────────────
        model = _get_model()
        instance: dict = {"filepath": image_url}
        if latitude is not None:
            instance["latitude"] = latitude
        if longitude is not None:
            instance["longitude"] = longitude

        raw = await asyncio.to_thread(
            model.predict,
            instances_dict={"instances": [instance]},
        )

        predictions = (raw or {}).get("predictions") or []
        logger.info(
            "[SpeciesNet] raw output: predictions_count=%d raw_keys=%s",
            len(predictions),
            list((raw or {}).keys()),
        )
        if predictions:
            pred0 = predictions[0]
            detections = pred0.get("detections") or []
            classifications = pred0.get("classifications") or {}
            classes = classifications.get("classes") or []
            scores = classifications.get("scores") or []
            logger.info(
                "[SpeciesNet] prediction[0]: detections=%d top_detection=%s "
                "classifications_count=%d top_label=%r top_score=%s",
                len(detections),
                detections[0] if detections else None,
                len(classes),
                classes[0] if classes else None,
                f"{scores[0]:.4f}" if scores else None,
            )
            if len(classes) > 1:
                logger.debug(
                    "[SpeciesNet] top-3 labels: %s",
                    list(zip(classes[:3], [f"{s:.4f}" for s in scores[:3]])),
                )

        if not predictions:
            result = ImageAnalysisResult(
                is_relevant=False,
                relevancy_reason="SpeciesNet returned no predictions.",
                image_url=image_url,
                latitude=latitude,
                longitude=longitude,
            )
            return result

        prediction = predictions[0]

        # ── 3. Relevancy check ────────────────────────────────────────────────
        relevancy = check_relevancy(prediction)
        logger.info(
            "Relevancy: is_relevant=%s reason=%r",
            relevancy.is_relevant, relevancy.reason,
        )

        if not relevancy.is_relevant:
            result = ImageAnalysisResult(
                is_relevant=False,
                relevancy_reason=relevancy.reason,
                image_url=image_url,
                latitude=latitude,
                longitude=longitude,
            )
            await _write_cache(image_url, result)
            return result

        # ── 4. Species identification ──────────────────────────────────────────
        identification = await identify_species(prediction, image_url)
        logger.info(
            "Identification: species=%r confidence=%.2f source=%s",
            identification.scientific_name,
            identification.confidence,
            identification.source,
        )

        # ── 5. Assemble final result ──────────────────────────────────────────
        result = ImageAnalysisResult(
            is_relevant=True,
            relevancy_reason=relevancy.reason,
            scientific_name=identification.scientific_name,
            confidence=identification.confidence,
            identification_source=identification.source,
            family=identification.family,
            genus=identification.genus,
            image_url=image_url,
            latitude=latitude,
            longitude=longitude,
        )

        # ── 6. Write to cache ─────────────────────────────────────────────────
        await _write_cache(image_url, result)

        return result


# ── Singleton factory ─────────────────────────────────────────────────────────

_agent_instance: Optional[ImageAnalysisAgent] = None


def get_image_agent() -> ImageAnalysisAgent:
    """Return a singleton ImageAnalysisAgent, constructing it on first call."""
    global _agent_instance  # noqa: PLW0603
    if _agent_instance is None:
        _agent_instance = ImageAnalysisAgent()
    return _agent_instance

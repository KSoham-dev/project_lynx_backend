"""
agents/layer1/identifier.py
============================
Layer 1 — Two-stage species identification.

Converted from Identifier.txt (Semantic Kernel → async Python + AzureChatOpenAI).

Stage 1 — SpeciesNet model path (high confidence):
  Parse scientific name from semicolon taxonomy label.
  Use if score >= CONFIDENCE_THRESHOLD (0.80) AND genus not in priority list.

Stage 2 — GPT-4o-mini vision fallback:
  Triggered by ANY of these 6 conditions (checked on top-1, except condition 2):

  #1 (top-1) — confidence < 0.80
  #2 (top-3) — ANY of top-3 labels has genus in PRIORITY_GENERA
  #3 (top-1) — genus or family cannot be parsed from label
  #4 (top-1) — scientific name cannot be derived (non-species epithet)
  #5 (top-1) — label is a bare detection token (blank/animal/vehicle/…)
  #6 (top-1) — no label at all (empty classifications)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from agents.layer1.gpt_vision import gpt_identify
from agents.layer1.schemas import IdentificationResult
from agents.layer1.species_list import PRIORITY_GENERA



logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.80

_NON_SPECIES_TOKENS: frozenset[str] = frozenset({
    "blank", "vehicle", "animal", "human", "person",
    "other", "unknown", "no_cv_result", "",
})


# ── Taxonomy label parser ─────────────────────────────────────────────────────

def _parse_taxonomy(label: str) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Extract (family, genus, scientific_name, common_name) from a SpeciesNet taxonomy label.

    SpeciesNet label format (semicolon-separated):
        "uuid;class;order;family;genus;species_epithet;common_name"
        e.g. "aa73e0ac-...;mammalia;carnivora;felidae;panthera;pardus;leopard"
        → family="FELIDAE", genus="Panthera", scientific_name="Panthera pardus",
          common_name="leopard"

    The last part is always the common name, second-to-last is the species
    epithet, and third-to-last is the genus.

    Returns (None, None, None, None) when:
      - Fewer than 3 parts exist (can't get genus + epithet + common_name)
      - Epithet or genus is a known non-species token
    """
    parts = [p.strip() for p in label.split(";") if p.strip()]
    if len(parts) < 3:
        return None, None, None, None

    raw_common_name = parts[-1].lower()
    raw_epithet     = parts[-2].lower()
    raw_genus       = parts[-3].lower()
    raw_family      = parts[-4].upper() if len(parts) >= 4 else None

    if raw_epithet in _NON_SPECIES_TOKENS or raw_genus in _NON_SPECIES_TOKENS:
        return raw_family, None, None, None   # family may still be parseable

    genus           = raw_genus.capitalize()
    scientific_name = f"{genus} {raw_epithet}"
    common_name     = raw_common_name if raw_common_name not in _NON_SPECIES_TOKENS else None
    return raw_family, genus, scientific_name, common_name


# ── GPT routing decision ──────────────────────────────────────────────────────

def _should_use_gpt(
    top_label: Optional[str],
    top_score: float,
    family: Optional[str],
    genus: Optional[str],
    scientific_name: Optional[str],
    all_labels: list[str],
) -> tuple[bool, str]:
    """
    Return (use_gpt: bool, reason: str).

    Conditions checked in order — first match wins.
    Condition 2 (priority genus) scans top-3 of all_labels.
    All other conditions use top-1 label only.
    """
    # Condition 6 — no label at all
    if not top_label:
        return True, "No species label returned by SpeciesNet."

    # Condition 5 — bare detection token, no taxonomy
    if top_label.lower().strip() in _NON_SPECIES_TOKENS:
        return True, f"Label '{top_label}' is a detection class, not a species."

    # Condition 1 — low confidence
    if top_score < CONFIDENCE_THRESHOLD:
        return True, f"Confidence {top_score:.1%} is below the {CONFIDENCE_THRESHOLD:.0%} threshold."

    # Condition 3 — genus or family unparseable
    if genus is None or family is None:
        return True, "Could not parse genus or family from SpeciesNet label."

    # Condition 4 — no valid scientific name
    if scientific_name is None:
        return True, "Could not derive a valid scientific name from label."

    # Condition 2 — priority genus in ANY of top-3 labels
    for candidate_label in all_labels[:3]:
        _, candidate_genus, _, _ = _parse_taxonomy(candidate_label)
        if candidate_genus and candidate_genus in PRIORITY_GENERA:
            return True, (
                f"Genus '{candidate_genus}' found in top-3 predictions "
                f"— routing to GPT for accuracy on priority species."
            )

    return False, "SpeciesNet label accepted."


# ── Public identification function ────────────────────────────────────────────

async def identify_species(
    prediction: dict,
    image_url: str,
) -> IdentificationResult:
    """
    Two-stage species identifier.

    Parameters
    ----------
    prediction:
        Single prediction entry from SpeciesNet output dict.
    image_url:
        Original image URL — passed to GPT-5-mini vision if fallback is needed.

    Returns
    -------
    IdentificationResult
    """
    classifications: dict = prediction.get("classifications") or {}
    classes: list[str]   = classifications.get("classes") or []
    scores: list[float]  = classifications.get("scores") or []

    top_label = classes[0] if classes else None
    top_score = float(scores[0]) if scores else 0.0

    family, genus, scientific_name, model_common_name = _parse_taxonomy(top_label or "")

    use_gpt, reason = _should_use_gpt(
        top_label=top_label,
        top_score=top_score,
        family=family,
        genus=genus,
        scientific_name=scientific_name,
        all_labels=classes,
    )

    logger.info(
        "Identification decision: use_gpt=%s reason=%r label=%r score=%.2f",
        use_gpt, reason, top_label, top_score,
    )

    if use_gpt:
        return await gpt_identify(image_url)

    return IdentificationResult(
        scientific_name=scientific_name,   # type: ignore[arg-type]
        confidence=top_score,
        source="model",
        top_label_raw=top_label,
        family=family,
        genus=genus,
        common_name=model_common_name,
    )

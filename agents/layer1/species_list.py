"""
agents/layer1/species_list.py
==============================
Hardcoded Indian wildlife family→genus mapping.

When any of the top-3 SpeciesNet classification labels resolves to a genus
in PRIORITY_GENERA, the identifier routes to GPT-4o-mini vision for
more accurate identification of these critical / high-risk species.

Covers: large felids, elephants, bears, crocodiles, cobras & vipers,
        bovids, canids, pufferfish, and sharks found in or around India.
"""
from __future__ import annotations

# family → set of genera that belong to the priority list
PRIORITY_SPECIES: dict[str, set[str]] = {
    "BOVIDAE": {
        "Antilope", "Bos", "Boselaphus", "Bubalus", "Budorcas",
        "Capricornis", "Naemorhedus", "Pseudois", "Tetracerus",
    },
    "CANIDAE": {"Canis", "Cuon", "Vulpes"},
    "CARCHARHINIDAE": {"Carcharhinus", "Glyphis"},
    "CROCODYLIDAE": {"Crocodylus"},
    "ELAPIDAE": {
        "Bungarus", "Calliophis", "Hydrophis", "Laticauda",
        "Naja", "Ophiophagus", "Sinomicrurus",
    },
    "ELEPHANTIDAE": {"Elephas"},
    "FELIDAE": {
        "Catopuma", "Felis", "Lynx", "Neofelis", "Otocolobus",
        "Panthera", "Prionailurus",
    },
    "TETRAODONTIDAE": {
        "Arothron", "Carinotetraodon", "Chelonodontops",
        "Dichotomyctere", "Lagocephalus", "Leiodon",
    },
    "URSIDAE": {"Helarctos", "Melursus", "Ursus"},
    "VIPERIDAE": {
        "Azemiops", "Daboia", "Echis", "Gloydius", "Hypnale",
        "Macrovipera", "Ovophis", "Protobothrops", "Trimeresurus",
    },
}

# Flat frozen sets for O(1) membership tests
PRIORITY_GENERA: frozenset[str] = frozenset(
    genus for genera in PRIORITY_SPECIES.values() for genus in genera
)
PRIORITY_FAMILIES: frozenset[str] = frozenset(PRIORITY_SPECIES.keys())

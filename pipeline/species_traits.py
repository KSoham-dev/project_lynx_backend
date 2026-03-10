"""
pipeline/species_traits.py
==========================
Synchronous pipeline that:
  1. Resolves a scientific name → Azure Blob filename prefix match
  2. Downloads the blob (IUCN JSON) into memory
  3. Concurrently fetches Wikipedia text + iNaturalist photo via ThreadPoolExecutor
  4. Calls Groq LLM (llama-3.3-70b-versatile) to extract structured traits
  5. Returns merged result; caches per-species for 24 h

Environment variables
---------------------
GROQ_API_KEY                    – Groq API secret key
AZURE_STORAGE_ACCOUNT_URL       – e.g. https://myaccount.blob.core.windows.net
                                  (used with Managed Identity on ACA)
AZURE_STORAGE_CONTAINER_NAME    – blob container holding the IUCN .txt files
AZURE_STORAGE_BLOB_FOLDER       – virtual folder inside the container, e.g. "iucn/species"
                                  leave unset if files are at the container root
AZURE_STORAGE_CONNECTION_STRING – local-dev alternative to account URL
"""

from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests as _requests
from cachetools import TTLCache
from groq import Groq

logger = logging.getLogger(__name__)

# ── Groq ────────────────────────────────────────────────────────────────────────
# API key and model name read at call time so .env loaded in main.py is visible.
_GROQ_MODEL = "llama-3.3-70b-versatile"

_SYSTEM_PROMPT = """You are a rigorous biological data extractor and taxonomist with deep expertise in zoology, ecology, and conservation biology.

Your task is to extract or estimate species trait data using the following strict priority order:
1. WIKIPEDIA TEXT (provided) — extract directly if mentioned. This is the highest priority source.
2. IUCN DATA (provided as JSON) — use assessment fields, references, and supplementary info.
3. PEER-REVIEWED KNOWLEDGE — only if above sources are silent. Do NOT add any citation or parenthetical note; output only the plain value.
4. If completely unknown across all sources, output exactly: "Unknown"

RULES:
- Be maximally precise. Use numeric ranges when available (e.g., "3–7 years", "1.2–2.4 kg").
- Never hallucinate. Every estimate must be scientifically defensible.
- For human_risk_level, choose strictly one of: "Very High", "High", "Caution", "Low".
- For fun facts, prioritize unusual, counterintuitive, or ecologically significant facts.
- short_description must be 1 concise sentence covering taxonomy, habitat, and conservation status.
- Output strictly valid JSON, no extra text."""

_USER_PROMPT_TEMPLATE = (
    "Extract traits.\n"
    "IUCN DATA: {payload}\n"
    "OUTPUT FORMAT (strict JSON):\n"
    '{{"lifespan_years","mass","length","short_description",'
    '"human_risk_level","human_threat_level","fun_fact_1","fun_fact_2","fun_fact_3"}}'
)

# ── Azure Blob ──────────────────────────────────────────────────────────────────
# All env vars read at call time (not import time) so .env loaded by main.py is always visible.

# ── In-memory TTL cache (24 h) — thread-safe ─────────────────────────────────
_cache: TTLCache[str, dict[str, Any]] = TTLCache(maxsize=2048, ttl=86400)
_cache_lock = threading.Lock()

# Shared HTTP session (connection pooling, thread-safe)
_http_session = _requests.Session()
_http_session.headers.update({"User-Agent": "Prahari/1.0"})


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _normalize(scientific_name: str) -> str:
    """'Caridina typus' → 'Caridinatypus' (blob prefix)."""
    return scientific_name.replace(" ", "")


def _safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    for k in keys:
        try:
            d = d[k]
        except (KeyError, TypeError, IndexError):
            return default
    return d or default


def _wiki_extract(name: str) -> str:
    """Return the plain-text Wikipedia extract for *name*, or 'Not found'."""
    words = name.split()
    preview = " ".join(words[:2]) + (" ..." if len(words) > 2 else "")
    logger.info("[_wiki_extract] called with: %s", preview)
    try:
        search = _http_session.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "format": "json",
                "search": name,
                "limit": 1,
            },
            timeout=10,
        ).json()
        if not search[1]:
            return "Not found"

        page_resp = _http_session.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "titles": search[1][0],
                "prop": "extracts",
                "explaintext": True,
            },
            timeout=10,
        ).json()
        pages = page_resp["query"]["pages"]
        return next(iter(pages.values())).get("extract") or "Not found"
    except Exception as exc:
        logger.warning("Wikipedia fetch failed for %r: %s", name, exc)
        return "Not found"


def _inaturalist_photo(scientific_name: str) -> tuple[str | None, str | None]:
    """Return (medium_url, attribution) from iNaturalist, or (None, None)."""
    try:
        results = _http_session.get(
            "https://api.inaturalist.org/v1/taxa",
            params={"q": scientific_name},
            timeout=10,
        ).json().get("results") or []
        if not results:
            return None, None
        photo = results[0].get("default_photo") or {}
        return photo.get("medium_url"), photo.get("attribution")
    except Exception as exc:
        logger.warning("iNaturalist fetch failed for %r: %s", scientific_name, exc)
        return None, None


def _build_blob_service_client():
    """Return a sync BlobServiceClient using Managed Identity (ACA) or
    connection string (local dev fallback).

    Env vars are read here (not at module level) so values loaded from .env
    by load_dotenv() in main.py are always visible.
    """
    from azure.storage.blob import BlobServiceClient  # lazy import keeps startup lean

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL", "")

    if conn_str:
        return BlobServiceClient.from_connection_string(conn_str)

    if account_url:
        from azure.identity import DefaultAzureCredential  # type: ignore[import]
        return BlobServiceClient(account_url, credential=DefaultAzureCredential())

    raise RuntimeError(
        "Set AZURE_STORAGE_ACCOUNT_URL (+ Managed Identity) "
        "or AZURE_STORAGE_CONNECTION_STRING."
    )


def _fetch_blob(normalized_name: str) -> dict[str, Any]:
    """
    Find the first blob whose name starts with *normalized_name* and return
    its JSON content as a dict.

    The optional AZURE_STORAGE_BLOB_FOLDER env var is prepended as a path
    prefix so files can live inside a virtual folder, e.g.:
      container: iucn-species
      folder:    iucn/assessments
      blob path: iucn/assessments/Caridinatypus_LC_..._2013

    Raises FileNotFoundError if no matching blob exists.
    """
    container = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "iucn-species")
    folder    = os.getenv("AZURE_STORAGE_BLOB_FOLDER", "").strip("/")

    # Build the full prefix: "folder/Caridinatypus" or just "Caridinatypus"
    prefix = f"{folder}/{normalized_name}" if folder else normalized_name

    service = _build_blob_service_client()
    container_client = service.get_container_client(container)
    blob_name: str | None = None

    for blob in container_client.list_blobs(name_starts_with=prefix):
        blob_name = blob.name
        break  # first match is sufficient

    if blob_name is None:
        raise FileNotFoundError(
            f"No blob found with prefix '{prefix}' in container '{container}'."
        )

    logger.info("Downloading blob: %s", blob_name)
    content = container_client.get_blob_client(blob_name).download_blob().readall()
    return json.loads(content)


def _call_groq(payload: dict[str, Any]) -> dict[str, Any]:
    """Call Groq LLM synchronously and return parsed JSON traits dict."""
    groq = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
    completion = groq.chat.completions.create(
        model=_GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_PROMPT_TEMPLATE.format(payload=json.dumps(payload)),
            },
        ],
        response_format={"type": "json_object"},
        timeout=30,
    )
    return json.loads(completion.choices[0].message.content)


# ── Public entry point ──────────────────────────────────────────────────────────

def run_pipeline(
    scientific_name: str,
    iucn_data: dict[str, Any] | None = None,
    wiki_text_override: str | None = None,
    iucn_summary_override: str | None = None,
) -> dict[str, Any]:
    """
    Full synchronous pipeline for one scientific name.

    Returns a merged dict containing IUCN fields, photo URL/credit,
    and LLM-extracted traits.

    Parameters
    ----------
    scientific_name:
        Scientific name, e.g. "Panthera tigris".
    iucn_data:
        Pre-fetched IUCN dict (e.g. from the live IUCN API fallback).  When
        provided (or when iucn_summary_override is set) the Azure Blob fetch
        is skipped entirely.
    wiki_text_override:
        Wikipedia extract fetched by the caller.  Used in the caller-supplied
        path so Groq gets the same Wikipedia context as the blob path.
    iucn_summary_override:
        Compact plain-text IUCN summary built by ExplorerAgent._iucn_summary().
        When set, the heavy IUCN dict is not included in the Groq payload;
        only scientific_name + this summary + wiki_extract are sent.
        This keeps the context to ~4 lines of conservation data instead of
        kilobytes of nested JSON, reducing token cost and improving signal/noise.

    Results are cached in-process for 24 hours — cache is thread-safe.
    FastAPI runs plain `def` routes in a thread-pool executor automatically,
    so this never blocks the event loop.
    """
    cache_key = scientific_name.strip().lower()

    # ── Cache read ────────────────────────────────────────────────────────────
    with _cache_lock:
        cached = _cache.get(cache_key)
    if cached is not None:
        logger.info("Cache hit for %r", scientific_name)
        return cached

    # ── Blob fetch (skipped when caller has data or provides a summary) ────────
    caller_supplied = iucn_data is not None or iucn_summary_override is not None
    if not caller_supplied:
        normalized = _normalize(scientific_name.strip())
        iucn_data = _fetch_blob(normalized)

    taxon: dict = (iucn_data or {}).get("taxon") or {}
    inaturalist_name = taxon.get("scientific_name", scientific_name)

    common_names = taxon.get("common_names") or []
    main_names = [c["name"] for c in common_names if c.get("main")]

    # ── Wikipedia fetch (only when blob was the source; skip when iucn_data was
    #    pre-fetched by the caller, e.g. from the live IUCN API, to avoid an
    #    extra 10-20 s blocking call in the hot path) ──────────────────────────
    wiki_text: str
    photo_url: str | None = None
    photo_credit: str | None = None

    if not caller_supplied:
        # iucn_data was fetched above from blob — also fetch wiki & inat photo
        wiki_name = main_names[0] if main_names else inaturalist_name
        with ThreadPoolExecutor(max_workers=2) as ex:
            wiki_future = ex.submit(_wiki_extract, wiki_name)
            photo_future = ex.submit(_inaturalist_photo, inaturalist_name)
            wiki_text = wiki_future.result()
            photo_url, photo_credit = photo_future.result()
    else:
        # iucn_data was supplied by the caller; iNaturalist is skipped since
        # ExplorerAgent fetches the photo independently.  Use the Wikipedia
        # text forwarded by the caller when available, otherwise fall back to
        # an informational placeholder.
        wiki_text = wiki_text_override or "Not fetched (pre-supplied IUCN data)"

    if iucn_summary_override:
        # Caller supplied a pre-built IUCN summary; send only the essential
        # fields to Groq so the context stays minimal and focused.
        payload: dict[str, Any] = {
            "scientific_name": scientific_name,
            "iucn_summary"   : iucn_summary_override,
            "wiki_extract"   : wiki_text,
        }
    else:
        payload = {
            "assessment_id"  : iucn_data.get("assessment_id"),
            "year_published" : iucn_data.get("year_published"),
            "scientific_name": taxon.get("scientific_name"),
            "common_names"   : main_names,
            "category"       : _safe_get(iucn_data, "red_list_category", "description", "en"),
            "references"     : iucn_data.get("references") or [],
            "url"            : iucn_data.get("url"),
            "sis_taxon_id"   : iucn_data.get("sis_taxon_id"),
            "photo_url"      : photo_url or "Not available",
            "photo_credit"   : photo_credit or "Not available",
            "wiki_extract"   : wiki_text,
        }

    # ── LLM trait extraction ──────────────────────────────────────────────────
    try:
        traits = _call_groq(payload)
    except Exception as exc:
        logger.error("Groq call failed for %r: %s", scientific_name, exc)
        traits = {}

    # Strip heavy wiki text from the final response
    payload.pop("wiki_extract", None)
    result: dict[str, Any] = {**payload, **traits}

    # ── Cache write ───────────────────────────────────────────────────────────
    with _cache_lock:
        _cache[cache_key] = result

    return result

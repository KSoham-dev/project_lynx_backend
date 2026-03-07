"""
dev_server.py
=============
Standalone FastAPI dev server for testing individual functions in:
  - agents/layer2/iucn_fetcher.py
  - pipeline/species_traits.py

Run with:
    uvicorn dev_server:app --reload --port 8001

Does NOT require the SpeciesNet model — only Azure Blob + Groq credentials.
"""
from dotenv import load_dotenv
load_dotenv()

import asyncio
import logging
from typing import Any

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Species API Dev Bench", version="1.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


# ── Shared helper ──────────────────────────────────────────────────────────────

async def _fetch_raw(scientific_name: str) -> dict[str, Any]:
    from agents.layer2.iucn_fetcher import get_iucn_raw
    try:
        return await get_iucn_raw(scientific_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── IUCN Fetcher endpoints ─────────────────────────────────────────────────────

@app.get("/dev/iucn/raw", tags=["iucn_fetcher"])
async def dev_iucn_raw(scientific_name: str = Query(...)):
    """Return the full raw IUCN blob for the given species."""
    return await _fetch_raw(scientific_name)


@app.get("/dev/iucn/scientific-name", tags=["iucn_fetcher"])
async def dev_iucn_scientific_name(scientific_name: str = Query(...)):
    from agents.layer2.iucn_fetcher import iucn_scientific_name
    raw = await _fetch_raw(scientific_name)
    return {"result": iucn_scientific_name(raw)}


@app.get("/dev/iucn/common-names", tags=["iucn_fetcher"])
async def dev_iucn_common_names(scientific_name: str = Query(...)):
    from agents.layer2.iucn_fetcher import iucn_common_names
    raw = await _fetch_raw(scientific_name)
    return {"result": iucn_common_names(raw)}


@app.get("/dev/iucn/taxon-field", tags=["iucn_fetcher"])
async def dev_iucn_taxon_field(
    scientific_name: str = Query(...),
    key: str = Query(..., description="Taxon field key, e.g. genus, family, kingdom"),
):
    from agents.layer2.iucn_fetcher import iucn_taxon_field
    raw = await _fetch_raw(scientific_name)
    return {"result": iucn_taxon_field(raw, key)}


@app.get("/dev/iucn/red-list-category", tags=["iucn_fetcher"])
async def dev_iucn_red_list_category(scientific_name: str = Query(...)):
    from agents.layer2.iucn_fetcher import iucn_red_list_category
    raw = await _fetch_raw(scientific_name)
    return {"result": iucn_red_list_category(raw)}


@app.get("/dev/iucn/red-list-code", tags=["iucn_fetcher"])
async def dev_iucn_red_list_code(scientific_name: str = Query(...)):
    from agents.layer2.iucn_fetcher import iucn_red_list_code
    raw = await _fetch_raw(scientific_name)
    return {"result": iucn_red_list_code(raw)}


@app.get("/dev/iucn/population-trend", tags=["iucn_fetcher"])
async def dev_iucn_population_trend(scientific_name: str = Query(...)):
    from agents.layer2.iucn_fetcher import iucn_population_trend
    raw = await _fetch_raw(scientific_name)
    return {"result": iucn_population_trend(raw)}


@app.get("/dev/iucn/threat-titles", tags=["iucn_fetcher"])
async def dev_iucn_threat_titles(
    scientific_name: str = Query(...),
    limit: int = Query(4, ge=1, le=20),
):
    from agents.layer2.iucn_fetcher import iucn_threat_titles
    raw = await _fetch_raw(scientific_name)
    return {"result": iucn_threat_titles(raw, limit)}


@app.get("/dev/iucn/habitat-names", tags=["iucn_fetcher"])
async def dev_iucn_habitat_names(
    scientific_name: str = Query(...),
    limit: int = Query(4, ge=1, le=20),
):
    from agents.layer2.iucn_fetcher import iucn_habitat_names
    raw = await _fetch_raw(scientific_name)
    return {"result": iucn_habitat_names(raw, limit)}


@app.get("/dev/iucn/taxonomy", tags=["iucn_fetcher"])
async def dev_iucn_taxonomy(scientific_name: str = Query(...)):
    """Return all taxonomy fields used by the reporter: genus, family, kingdom, class, order."""
    from agents.layer2.iucn_fetcher import iucn_taxon_field, iucn_scientific_name, iucn_common_names
    raw = await _fetch_raw(scientific_name)
    return {
        "scientific_name": iucn_scientific_name(raw),
        "common_names":    iucn_common_names(raw),
        "genus":           iucn_taxon_field(raw, "genus"),    # resolves genus_name
        "family":          iucn_taxon_field(raw, "family"),   # resolves family_name
        "order":           iucn_taxon_field(raw, "order"),    # resolves order_name
        "class":           iucn_taxon_field(raw, "class"),    # resolves class_name
        "kingdom":         iucn_taxon_field(raw, "kingdom"),  # resolves kingdom_name
        "phylum":          iucn_taxon_field(raw, "phylum"),   # resolves phylum_name
    }


# ── Species Traits endpoints ───────────────────────────────────────────────────

@app.get("/dev/traits/normalize", tags=["species_traits"])
async def dev_traits_normalize(name: str = Query(...)):
    from pipeline.species_traits import _normalize
    return {"result": _normalize(name)}


@app.get("/dev/traits/wiki", tags=["species_traits"])
async def dev_traits_wiki(name: str = Query(...)):
    from pipeline.species_traits import _wiki_extract
    result = await asyncio.to_thread(_wiki_extract, name)
    return {"result": result}


@app.get("/dev/traits/inaturalist-photo", tags=["species_traits"])
async def dev_traits_inaturalist_photo(name: str = Query(...)):
    from pipeline.species_traits import _inaturalist_photo
    photo_url, attribution = await asyncio.to_thread(_inaturalist_photo, name)
    return {"photo_url": photo_url, "attribution": attribution}


class GroqPayload(BaseModel):
    payload: dict[str, Any]


@app.post("/dev/traits/groq", tags=["species_traits"])
async def dev_traits_groq(body: GroqPayload):
    from pipeline.species_traits import _call_groq
    try:
        result = await asyncio.to_thread(_call_groq, body.payload)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/dev/traits/fetch-blob", tags=["species_traits"])
async def dev_traits_fetch_blob(scientific_name: str = Query(...)):
    from pipeline.species_traits import _fetch_blob, _normalize
    try:
        result = await asyncio.to_thread(_fetch_blob, _normalize(scientific_name))
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/dev/traits/run-pipeline", tags=["species_traits"])
async def dev_traits_run_pipeline(scientific_name: str = Query(...)):
    from pipeline.species_traits import run_pipeline
    try:
        result = await asyncio.to_thread(run_pipeline, scientific_name)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/dev/traits/full-context", tags=["species_traits"])
async def dev_traits_full_context(scientific_name: str = Query(...)):
    """
    Full pipeline context inspector: shows every piece of data assembled by
    run_pipeline() — IUCN blob summary, Wikipedia extract (with the exact search
    name used), iNaturalist photo, the exact Groq input payload, Groq-extracted
    traits, and the final forwarded result sent to ExplorerAgent.
    Bypasses the 24-hour cache so data is always live.
    """
    from pipeline.species_traits import (
        _normalize, _fetch_blob, _wiki_extract, _inaturalist_photo,
        _call_groq, _safe_get,
    )
    from agents.layer2.iucn_fetcher import (
        iucn_scientific_name, iucn_common_names, iucn_taxon_field,
        iucn_red_list_category, iucn_red_list_code, iucn_population_trend,
        iucn_threat_titles, iucn_habitat_names,
    )

    # 1. Fetch IUCN blob
    try:
        iucn_data = await asyncio.to_thread(_fetch_blob, _normalize(scientific_name))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    taxon = iucn_data.get("taxon") or {}
    inaturalist_name = taxon.get("scientific_name", scientific_name)
    common_names_raw = taxon.get("common_names") or []
    main_names = [c["name"] for c in common_names_raw if c.get("main")]
    wiki_name = main_names[0] if main_names else inaturalist_name

    # 2. Concurrent Wikipedia extract + iNaturalist photo (mirrors run_pipeline)
    wiki_text, photo_pair = await asyncio.gather(
        asyncio.to_thread(_wiki_extract, wiki_name),
        asyncio.to_thread(_inaturalist_photo, inaturalist_name),
    )
    photo_url, photo_credit = photo_pair

    # 3. Build Groq input payload — identical to run_pipeline()
    groq_payload = {
        "assessment_id":   iucn_data.get("assessment_id"),
        "year_published":  iucn_data.get("year_published"),
        "scientific_name": taxon.get("scientific_name"),
        "common_names":    main_names,
        "category":        _safe_get(iucn_data, "red_list_category", "description", "en"),
        "references":      iucn_data.get("references") or [],
        "url":             iucn_data.get("url"),
        "sis_taxon_id":    iucn_data.get("sis_taxon_id"),
        "photo_url":       photo_url or "Not available",
        "photo_credit":    photo_credit or "Not available",
        "wiki_extract":    wiki_text,
    }

    # 4. Call Groq
    groq_traits: dict = {}
    groq_error: str | None = None
    try:
        groq_traits = await asyncio.to_thread(_call_groq, groq_payload)
    except Exception as exc:
        groq_error = str(exc)

    # 5. Final forwarded result — same as run_pipeline() output
    #    (wiki_extract stripped, Groq traits merged in)
    forwarded = {k: v for k, v in groq_payload.items() if k != "wiki_extract"}
    forwarded.update(groq_traits)

    result: dict = {
        "iucn_summary": {
            "scientific_name":   iucn_scientific_name(iucn_data),
            "common_names":      iucn_common_names(iucn_data),
            "genus":             iucn_taxon_field(iucn_data, "genus"),
            "family":            iucn_taxon_field(iucn_data, "family"),
            "order":             iucn_taxon_field(iucn_data, "order"),
            "class":             iucn_taxon_field(iucn_data, "class"),
            "kingdom":           iucn_taxon_field(iucn_data, "kingdom"),
            "phylum":            iucn_taxon_field(iucn_data, "phylum"),
            "red_list_category": iucn_red_list_category(iucn_data),
            "red_list_code":     iucn_red_list_code(iucn_data),
            "population_trend":  iucn_population_trend(iucn_data),
            "top_threats":       iucn_threat_titles(iucn_data, 6),
            "habitats":          iucn_habitat_names(iucn_data, 6),
        },
        "wikipedia": {
            "search_name_used": wiki_name,
            "extract":          wiki_text,
        },
        "inaturalist_photo": {
            "photo_url":   photo_url,
            "attribution": photo_credit,
        },
        "groq_input_payload": groq_payload,
        "groq_traits":        groq_traits,
        "forwarded_result":   forwarded,
    }
    if groq_error:
        result["groq_error"] = groq_error

    return result


@app.get("/dev/traits/llm-input-explore", tags=["species_traits"])
async def dev_traits_llm_input_explore(scientific_name: str = Query(...)):
    """
    Show the exact payload sent to the Groq LLM during an EXPLORE request.

    Builds the full context (IUCN blob → Wikipedia extract → iNaturalist photo)
    exactly as run_pipeline() would, and returns both the raw payload and a
    human-readable summary of each section.  No Groq call is made — this is
    purely for inspection.
    """
    from pipeline.species_traits import (
        _normalize, _fetch_blob, _wiki_extract, _inaturalist_photo, _safe_get,
        _SYSTEM_PROMPT, _USER_PROMPT_TEMPLATE,
    )

    try:
        iucn_data = await asyncio.to_thread(_fetch_blob, _normalize(scientific_name))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    taxon = iucn_data.get("taxon") or {}
    inaturalist_name = taxon.get("scientific_name", scientific_name)
    common_names_raw = taxon.get("common_names") or []
    main_names = [c["name"] for c in common_names_raw if c.get("main")]
    wiki_name = main_names[0] if main_names else inaturalist_name

    wiki_text, photo_pair = await asyncio.gather(
        asyncio.to_thread(_wiki_extract, wiki_name),
        asyncio.to_thread(_inaturalist_photo, inaturalist_name),
    )
    photo_url, photo_credit = photo_pair

    payload = {
        "assessment_id":   iucn_data.get("assessment_id"),
        "year_published":  iucn_data.get("year_published"),
        "scientific_name": taxon.get("scientific_name"),
        "common_names":    main_names,
        "category":        _safe_get(iucn_data, "red_list_category", "description", "en"),
        "references":      iucn_data.get("references") or [],
        "url":             iucn_data.get("url"),
        "sis_taxon_id":    iucn_data.get("sis_taxon_id"),
        "photo_url":       photo_url or "Not available",
        "photo_credit":    photo_credit or "Not available",
        "wiki_extract":    wiki_text,
    }

    rendered_user_prompt = _USER_PROMPT_TEMPLATE.format(payload=__import__("json").dumps(payload))

    return {
        "model":              "llama-3.3-70b-versatile (Groq)",
        "system_prompt":      _SYSTEM_PROMPT,
        "user_prompt":        rendered_user_prompt,
        "payload":            payload,
        "payload_sections": {
            "iucn_fields":       {k: payload[k] for k in ("assessment_id", "year_published", "scientific_name", "common_names", "category", "url", "sis_taxon_id")},
            "wiki_extract":      {"search_name_used": wiki_name, "length_chars": len(wiki_text), "preview": wiki_text[:500]},
            "inaturalist_photo": {"photo_url": photo_url, "attribution": photo_credit},
            "references_count":  len(payload["references"]),
        },
    }


@app.get("/dev/traits/llm-input-encyclopedia", tags=["species_traits"])
async def dev_traits_llm_input_encyclopedia(scientific_name: str = Query(...)):
    """
    Show the context used by the Encyclopedia agent.

    Encyclopedia makes NO LLM call — it passes the raw IUCN blob directly to
    the frontend.  This endpoint shows the complete IUCN data plus the
    iNaturalist photo that will be included in the response, so you can verify
    what the frontend will receive without running the full pipeline.
    """
    from agents.layer2.iucn_fetcher import (
        iucn_scientific_name, iucn_common_names, iucn_taxon_field,
        iucn_red_list_category, iucn_red_list_code, iucn_population_trend,
        iucn_threat_titles, iucn_habitat_names,
    )
    from agents.layer2.inaturalist import get_inaturalist_photo

    raw = await _fetch_raw(scientific_name)

    photo_url, photo_credit = await get_inaturalist_photo(
        iucn_scientific_name(raw) or scientific_name
    )

    return {
        "note": "Encyclopedia agent makes NO LLM call. This is the full context passed to the frontend.",
        "iucn_summary": {
            "scientific_name":   iucn_scientific_name(raw),
            "common_names":      iucn_common_names(raw),
            "genus":             iucn_taxon_field(raw, "genus"),
            "family":            iucn_taxon_field(raw, "family"),
            "order":             iucn_taxon_field(raw, "order"),
            "class":             iucn_taxon_field(raw, "class"),
            "kingdom":           iucn_taxon_field(raw, "kingdom"),
            "phylum":            iucn_taxon_field(raw, "phylum"),
            "red_list_category": iucn_red_list_category(raw),
            "red_list_code":     iucn_red_list_code(raw),
            "population_trend":  iucn_population_trend(raw),
            "top_threats":       iucn_threat_titles(raw, 6),
            "habitats":          iucn_habitat_names(raw, 6),
            "iucn_url":          raw.get("url"),
            "assessment_id":     raw.get("assessment_id"),
            "year_published":    raw.get("year_published"),
            "total_keys":        len(raw),
        },
        "inaturalist_photo": {
            "photo_url":   photo_url,
            "attribution": photo_credit,
        },
        "full_iucn_raw": raw,
    }


# ── Reporter / Assessment endpoints ───────────────────────────────────────────

@app.get("/dev/reporter/threat-level", tags=["reporter"])
async def dev_reporter_threat_level(scientific_name: str = Query(...)):
    """
    Derive the conservation threat level deterministically from the IUCN red list
    code — no LLM needed. Returns one of: Critical, High, Moderate, Low, Unknown.
    """
    from agents.layer2.reporter import _threat_level_from_iucn
    raw = await _fetch_raw(scientific_name)
    from agents.layer2.iucn_fetcher import iucn_red_list_code, iucn_red_list_category
    return {
        "threat_level":      _threat_level_from_iucn(raw),
        "red_list_code":     iucn_red_list_code(raw),
        "red_list_category": iucn_red_list_category(raw),
    }


@app.get("/dev/reporter/safety-data", tags=["reporter"])
async def dev_reporter_safety_data(scientific_name: str = Query(...)):
    """
    Call the Azure OpenAI GPT safety assessment for a species.
    Returns risk_level and user_safety_precautions (plain-language items for the reporter).
    """
    from agents.layer2.reporter import get_species_safety_data
    raw = await _fetch_raw(scientific_name)
    try:
        result = await get_species_safety_data(scientific_name, raw)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/dev/reporter/report-summary", tags=["reporter"])
async def dev_reporter_report_summary(scientific_name: str = Query(...)):
    """
    Simulate all the fields the reporter PDF will populate for a species:
    taxonomy, confidence, threat level, risk level, red list category.
    Does NOT generate a PDF — just shows what the data layer will provide.
    """
    from agents.layer2.iucn_fetcher import (
        iucn_scientific_name, iucn_common_names, iucn_taxon_field,
        iucn_red_list_category, iucn_red_list_code, iucn_population_trend,
    )
    from agents.layer2.reporter import _threat_level_from_iucn
    raw = await _fetch_raw(scientific_name)
    return {
        "scientific_name":   iucn_scientific_name(raw),
        "common_names":      iucn_common_names(raw),
        "genus":             iucn_taxon_field(raw, "genus"),
        "family":            iucn_taxon_field(raw, "family"),
        "order":             iucn_taxon_field(raw, "order"),
        "class":             iucn_taxon_field(raw, "class"),
        "kingdom":           iucn_taxon_field(raw, "kingdom"),
        "red_list_code":     iucn_red_list_code(raw),
        "red_list_category": iucn_red_list_category(raw),
        "population_trend":  iucn_population_trend(raw),
        "threat_level":      _threat_level_from_iucn(raw),
        # confidence comes from the identification model, not IUCN — shown as note
        "confidence":        "N/A — set by SpeciesNet model or GPT Vision at runtime",
        # severity comes from text_agent at runtime
        "severity":          "N/A — derived from user report text at runtime",
    }


# ── Pipeline Inspector endpoints ───────────────────────────────────────────────
# These endpoints simulate or expose key data-handoff points between agents,
# without running the full pipeline or any LLM calls.


@app.get("/dev/pipeline/tool-selection", tags=["pipeline"])
async def dev_pipeline_tool_selection(
    has_message: bool = Query(False, description="Whether the request has a text message"),
    has_image: bool = Query(False, description="Whether the request has an image_url"),
    has_gps: bool = Query(False, description="Whether the request has latitude + longitude"),
):
    """
    Simulate the Orchestrator's _available_tools() gate.

    Shows exactly which Layer 1 tools would be offered to the LLM
    for a given combination of inputs — before any LLM call is made.
    Also shows the [Context] block the LLM would receive.
    """
    tools_selected = []
    reasons = {}

    if has_message:
        tools_selected.append("analyse_text")
        reasons["analyse_text"] = "message is present → text analysis tool offered"
    else:
        reasons["analyse_text"] = "SKIPPED — message is [not provided]"

    if has_image:
        tools_selected.append("analyse_image")
        reasons["analyse_image"] = "image_url is present → image analysis tool offered"
    else:
        reasons["analyse_image"] = "SKIPPED — image_url is [not provided]"

    if has_gps:
        tools_selected.append("location_context")
        reasons["location_context"] = "latitude + longitude both present → location tool offered"
    else:
        reasons["location_context"] = "SKIPPED — GPS coordinates are [not provided]"

    context_block_preview = "\n".join([
        f"Text message: {'<user_message>' if has_message else '[not provided]'}",
        f"Image URL: {'<image_url>' if has_image else '[not provided]'}",
        f"GPS coordinates: {'latitude=<lat>, longitude=<lon>' if has_gps else '[not provided]'}",
    ])

    return {
        "tools_selected": tools_selected,
        "tool_count": len(tools_selected),
        "will_abort": len(tools_selected) == 0,
        "abort_reason": "No applicable tools — orchestrator returns empty dict" if not tools_selected else None,
        "selection_rules": reasons,
        "context_block_sent_to_llm": context_block_preview,
    }


@app.get("/dev/pipeline/name-resolution", tags=["pipeline"])
async def dev_pipeline_name_resolution(
    image_scientific_name: str = Query("", description="scientific_name from analyse_image result"),
    text_scientific_name: str = Query("", description="scientific_name from analyse_text result"),
):
    """
    Simulate SpeciesInfoAgent._extract_scientific_name().

    Shows the priority order used to pick the scientific name that drives
    the IUCN blob fetch: image_analysis wins over text_analysis.
    Unknown/unidentified strings are treated as absent.
    """
    _SKIP = {"unknown", "unidentified", "", "none"}

    candidates = [
        ("analyse_image", image_scientific_name.strip()),
        ("analyse_text",  text_scientific_name.strip()),
    ]

    chosen_source: str | None = None
    chosen_name:   str | None = None
    trace = []

    for source, name in candidates:
        if name and name.lower() not in _SKIP:
            if chosen_name is None:
                chosen_name   = name
                chosen_source = source
                trace.append({"source": source, "value": name, "selected": True,  "reason": "First valid candidate"})
            else:
                trace.append({"source": source, "value": name, "selected": False, "reason": "Lower priority — image wins"})
        else:
            trace.append({"source": source, "value": name or "(empty)", "selected": False,
                           "reason": "Rejected — value is empty or in skip-list"})

    return {
        "resolved_name":    chosen_name,
        "resolved_source":  chosen_source,
        "iucn_fetch_will_run": chosen_name is not None,
        "iucn_fetch_skipped_reason": "No valid scientific name extracted" if not chosen_name else None,
        "resolution_trace": trace,
        "priority_order":   ["analyse_image.scientific_name", "analyse_text.scientific_name"],
        "skip_list":        sorted(_SKIP),
    }


@app.get("/dev/pipeline/severity-chain", tags=["pipeline"])
async def dev_pipeline_severity_chain(
    text_severity: str = Query("", description="severity string from analyse_text (critical/high/medium/low/informational)"),
    risk_level: str = Query("", description="risk_level string from GPT safety call (Very High/High/Caution/Low)"),
):
    """
    Simulate the Reporter's severity inference chain.

    Shows which branch is taken and what the final IncidentSeverity enum
    value will be — matching the exact logic in ReporterAgent.run().
    """
    from agents.enums import IncidentSeverity

    _SEV_MAP = {"critical": "CRITICAL", "high": "HIGH", "medium": "MEDIUM"}
    _RISK_MAP = {"very high": "critical", "high": "high", "caution": "medium"}

    ts = text_severity.strip().lower()
    rl = risk_level.strip().lower()

    if ts:
        branch = "text_severity"
        final_sev = _SEV_MAP.get(ts, "LOW")
        explanation = (
            f"text_severity={text_severity!r} is present → mapped directly via _map_severity()"
        )
    elif rl:
        branch = "risk_level_inference"
        inferred_sev = _RISK_MAP.get(rl, "low")
        final_sev = _SEV_MAP.get(inferred_sev, "LOW")
        explanation = (
            f"text_severity is empty → inferred from risk_level={risk_level!r} → "
            f"intermediate={inferred_sev!r} → mapped via _map_severity()"
        )
    else:
        branch = "default"
        final_sev = "LOW"
        explanation = "Both text_severity and risk_level are absent → defaults to LOW"

    return {
        "incident_severity":    final_sev,
        "branch_taken":         branch,
        "explanation":          explanation,
        "inputs": {
            "text_severity":  text_severity or "(empty)",
            "risk_level":     risk_level    or "(empty)",
        },
        "severity_map":   {"critical": "CRITICAL", "high": "HIGH", "medium": "MEDIUM", "(else)": "LOW"},
        "risk_level_map": {"very high": "critical→CRITICAL", "high": "high→HIGH",
                           "caution": "medium→MEDIUM", "(else)": "low→LOW"},
    }


@app.get("/dev/pipeline/layer1-parse", tags=["pipeline"])
async def dev_pipeline_layer1_parse(
    image_result_json: str = Query("", description="Raw JSON string from analyse_image tool"),
    text_result_json: str = Query("", description="Raw JSON string from analyse_text tool"),
    location_result_json: str = Query("", description="Raw JSON string from location_context tool"),
):
    """
    Parse and validate raw Layer 1 tool output strings exactly as Layer 2 receives them.

    The Orchestrator passes tool results as a dict of raw JSON strings.
    This endpoint shows what fields are actually present after parsing,
    and flags any missing or unexpected values.
    """
    import json as _json

    def _parse(label: str, raw: str):
        if not raw.strip():
            return {"status": "not_provided", "fields": {}}
        try:
            parsed = _json.loads(raw)
            return {"status": "ok", "fields": parsed}
        except Exception as e:
            return {"status": "parse_error", "error": str(e), "raw_preview": raw[:200]}

    image_parsed  = _parse("analyse_image",    image_result_json)
    text_parsed   = _parse("analyse_text",     text_result_json)
    loc_parsed    = _parse("location_context", location_result_json)

    # Validate expected fields
    _IMAGE_EXPECTED  = ["is_relevant", "scientific_name", "confidence", "identification_source", "family", "genus"]
    _TEXT_EXPECTED   = ["severity", "scientific_name", "common_name", "identification_confidence", "size"]
    _LOC_EXPECTED    = ["district", "state", "country", "protected_area", "geohash", "formatted"]

    def _check_fields(parsed: dict, expected: list):
        if parsed["status"] != "ok":
            return {}
        fields = parsed["fields"]
        return {
            f: {"present": f in fields and fields[f] is not None, "value": fields.get(f)}
            for f in expected
        }

    return {
        "analyse_image": {
            **image_parsed,
            "field_check": _check_fields(image_parsed, _IMAGE_EXPECTED),
        },
        "analyse_text": {
            **text_parsed,
            "field_check": _check_fields(text_parsed, _TEXT_EXPECTED),
        },
        "location_context": {
            **loc_parsed,
            "field_check": _check_fields(loc_parsed, _LOC_EXPECTED),
        },
    }


@app.get("/dev/pipeline/iucn-explorer-fields", tags=["pipeline"])
async def dev_pipeline_iucn_explorer_fields(scientific_name: str = Query(...)):
    """
    Show which IUCN fields ExplorerAgent passes as the base dict, and which
    fields Groq is expected to overlay on top.

    The IUCN base dict is the full raw blob. Groq overlays only the 9
    _GPT_TRAIT_FIELDS. This helps verify no important IUCN field is being
    silently dropped or overwritten.
    """
    from agents.layer2.iucn_fetcher import (
        iucn_scientific_name, iucn_common_names, iucn_taxon_field,
        iucn_red_list_category, iucn_red_list_code, iucn_population_trend,
        iucn_threat_titles, iucn_habitat_names,
    )

    _GPT_TRAIT_FIELDS = (
        "lifespan_years", "mass", "length", "short_description",
        "human_risk_level", "human_threat_level",
        "fun_fact_1", "fun_fact_2", "fun_fact_3",
    )

    raw = await _fetch_raw(scientific_name)

    # Check for key collisions: any IUCN top-level key that Groq would overwrite
    iucn_top_level_keys = list(raw.keys())
    collision_risk = [k for k in _GPT_TRAIT_FIELDS if k in iucn_top_level_keys]

    return {
        "iucn_base": {
            "scientific_name":   iucn_scientific_name(raw),
            "common_names":      iucn_common_names(raw),
            "genus":             iucn_taxon_field(raw, "genus"),
            "family":            iucn_taxon_field(raw, "family"),
            "order":             iucn_taxon_field(raw, "order"),
            "class":             iucn_taxon_field(raw, "class"),
            "kingdom":           iucn_taxon_field(raw, "kingdom"),
            "red_list_category": iucn_red_list_category(raw),
            "red_list_code":     iucn_red_list_code(raw),
            "population_trend":  iucn_population_trend(raw),
            "top_threats":       iucn_threat_titles(raw, 5),
            "habitats":          iucn_habitat_names(raw, 5),
            "top_level_key_count": len(iucn_top_level_keys),
            "all_top_level_keys": iucn_top_level_keys,
        },
        "groq_overlay_fields": list(_GPT_TRAIT_FIELDS),
        "collision_check": {
            "collisions_found": len(collision_risk),
            "colliding_keys": collision_risk,
            "note": "Groq output overwrites these IUCN keys if present in both" if collision_risk
                    else "No collision — Groq fields are additive",
        },
    }


@app.get("/dev/pipeline/species-cache", tags=["pipeline"])
async def dev_pipeline_species_cache(scientific_name: str = Query(...)):
    """
    Check the Cosmos DB species_context_cache for both cache buckets.

    Shows whether the 'iucn' and 'traits' cache entries exist for a species,
    their expected document IDs, and a summary of what's cached.
    This reveals whether a live pipeline run will be a cache hit or miss.
    """
    from agents.layer2.species_cache import read_species_cache, _cache_id

    iucn_id   = _cache_id("iucn",   scientific_name)
    traits_id = _cache_id("traits", scientific_name)

    iucn_cached   = await read_species_cache("iucn",   scientific_name)
    traits_cached = await read_species_cache("traits", scientific_name)

    # Summarise traits cache without dumping huge blob
    traits_summary = None
    if traits_cached:
        from agents.layer2.explorer import _GPT_TRAIT_FIELDS
        traits_summary = {
            "total_keys": len(traits_cached),
            "gpt_traits_present": {k: k in traits_cached and bool(traits_cached[k]) for k in _GPT_TRAIT_FIELDS},
            "photo_url_present": bool(traits_cached.get("photo_url")),
        }

    iucn_summary = None
    if iucn_cached:
        from agents.layer2.iucn_fetcher import iucn_scientific_name, iucn_red_list_category
        iucn_summary = {
            "total_keys": len(iucn_cached),
            "scientific_name": iucn_scientific_name(iucn_cached),
            "red_list_category": iucn_red_list_category(iucn_cached),
        }

    return {
        "iucn_cache": {
            "document_id": iucn_id,
            "hit": iucn_cached is not None,
            "summary": iucn_summary,
        },
        "traits_cache": {
            "document_id": traits_id,
            "hit": traits_cached is not None,
            "summary": traits_summary,
        },
        "cache_ttl_seconds": 86400,
        "cache_ttl_hours": 24,
    }


@app.get("/dev/pipeline/location-geocode", tags=["pipeline"])
async def dev_pipeline_location_geocode(
    latitude: float = Query(..., description="GPS latitude"),
    longitude: float = Query(..., description="GPS longitude"),
):
    """
    Run the location_context tool directly and show the full geocode result.

    Exposes all fields returned by Nominatim reverse-geocode + the protected
    area detection logic. Uses the same caching layer as the live pipeline
    (Cosmos location_cache, 30-day TTL).
    """
    from agents.layer1.location import get_location_context
    try:
        result = await get_location_context(latitude, longitude)
        return {
            "inputs": {"latitude": latitude, "longitude": longitude},
            "location_context": result,
            "fields_present": {k: v is not None for k, v in result.items()},
            "protected_area_detected": result.get("protected_area") is not None,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/dev/pipeline/data-handoff", tags=["pipeline"])
async def dev_pipeline_data_handoff(
    scientific_name: str = Query(...),
    query_type: str = Query("explore", description="explore | report | encyclopedia"),
):
    """
    Full data-handoff trace for a species — no LLM calls, no PDF.

    Shows every field at every boundary in the pipeline:
      IUCN blob → SpeciesInfoAgent output
      ↓
      ExplorerAgent inputs │ ReporterAgent inputs │ EncyclopediaAgent inputs
      (which IUCN fields each agent reads and what it adds)

    This is the canonical map of data movement across agents.
    """
    from agents.layer2.iucn_fetcher import (
        iucn_scientific_name, iucn_common_names, iucn_taxon_field,
        iucn_red_list_category, iucn_red_list_code, iucn_population_trend,
        iucn_threat_titles, iucn_habitat_names,
    )
    from agents.layer2.reporter import _threat_level_from_iucn
    from agents.layer2.species_cache import _cache_id

    raw = await _fetch_raw(scientific_name)

    # ── SpeciesInfoAgent output ──
    species_info_out = {
        "scientific_name":   iucn_scientific_name(raw),
        "common_names":      iucn_common_names(raw),
        "red_list_category": iucn_red_list_category(raw),
        "red_list_code":     iucn_red_list_code(raw),
        "population_trend":  iucn_population_trend(raw),
        "top_threats":       iucn_threat_titles(raw, 5),
        "habitats":          iucn_habitat_names(raw, 5),
        "assessment_id":     raw.get("assessment_id"),
        "year_published":    raw.get("year_published"),
        "top_level_keys":    list(raw.keys()),
    }

    # ── Per-agent field maps ──
    qt = query_type.lower().strip()

    explorer_fields = {
        "reads_from_iucn":   ["all keys via **iucn_data spread", "scientific_name via iucn_scientific_name()"],
        "adds_groq_overlay":  ["lifespan_years", "mass", "length", "short_description",
                               "human_risk_level", "human_threat_level",
                               "fun_fact_1", "fun_fact_2", "fun_fact_3"],
        "adds_photo":         ["photo_url", "photo_credit"],
        "adds_location":      ["location.latitude", "location.longitude", "location.district",
                               "location.state", "location.country", "location.protected_area",
                               "location.formatted"],
        "groq_source":        "run_pipeline() → Groq llama-3.3-70b-versatile",
        "photo_source":       "inaturalist.get_inaturalist_photo()",
        "cache_keys_checked": [_cache_id("iucn", scientific_name), _cache_id("traits", scientific_name)],
    }

    reporter_fields = {
        "reads_from_iucn":    ["scientific_name", "common_names", "genus (via alias)", "family (via alias)",
                               "red_list_category", "red_list_code", "population_trend",
                               "threat_titles", "habitat_names"],
        "reads_from_image":   ["scientific_name (fallback)", "common_name", "genus", "family",
                               "confidence", "identification_source"],
        "reads_from_text":    ["severity", "size", "colour", "behaviour", "distinctive_features"],
        "reads_from_location":["district", "state", "country", "protected_area", "formatted"],
        "adds_gpt_safety":    ["user_safety_precautions", "risk_level"],
        "adds_deterministic": ["threat_level (from IUCN code)", "incident_severity (from text or risk_level)"],
        "adds_metadata":      ["report_id (UUID)", "submitted_at (UTC timestamp)", "pdf_url", "severity"],
        "gpt_call":           "Azure OpenAI — user safety precautions + risk_level",
        "threat_level_now":   _threat_level_from_iucn(raw),
        "iucn_code_now":      iucn_red_list_code(raw),
    }

    encyclopedia_fields = {
        "reads_from_iucn":    ["all keys via **iucn_data spread"],
        "adds_photo":         ["photo_url", "photo_credit"],
        "no_llm_calls":       True,
        "photo_source":       "inaturalist.get_inaturalist_photo()",
        "note":               "Lightest agent — IUCN blob + iNaturalist photo only",
    }

    agent_map = {
        "explore":      explorer_fields,
        "report":       reporter_fields,
        "encyclopedia": encyclopedia_fields,
    }

    return {
        "resolved_name":       iucn_scientific_name(raw),
        "query_type":          qt,
        "species_info_agent_output": species_info_out,
        "layer2_agent":        qt,
        "agent_field_map":     agent_map.get(qt, {"error": f"Unknown query_type {qt!r}"}),
        "shared_iucn_data":    True,
        "iucn_data_is_none_if": "No blob found in Azure / scientific_name unknown/unidentified",
    }


# ── SpeciesNet endpoints ───────────────────────────────────────────────────────
# These endpoints exercise every stage of the SpeciesNet → identification pipeline
# without needing the SpeciesNet model loaded (except /dev/speciesnet/full-analysis).


@app.get("/dev/speciesnet/parse-label", tags=["speciesnet"])
async def dev_speciesnet_parse_label(label: str = Query(..., description="Raw SpeciesNet taxonomy label string")):
    """
    Parse a raw SpeciesNet taxonomy label into its taxonomy components.

    SpeciesNet label format: "uuid;class;order;family;genus;epithet;common_name"
    e.g.  "aa73e0ac;mammalia;carnivora;felidae;panthera;tigris;tiger"
    → family="FELIDAE", genus="Panthera", scientific_name="Panthera tigris", common_name="tiger"
    """
    from agents.layer1.identifier import _parse_taxonomy
    family, genus, scientific_name, common_name = _parse_taxonomy(label)
    return {
        "input_label":     label,
        "family":          family,
        "genus":           genus,
        "scientific_name": scientific_name,
        "common_name":     common_name,
        "valid":           scientific_name is not None,
        "note":            "valid=False means the label is a bare detection token (blank/animal/vehicle/…) or has fewer than 3 semicolon-parts",
    }


@app.get("/dev/speciesnet/routing-decision", tags=["speciesnet"])
async def dev_speciesnet_routing_decision(
    top_label: str  = Query("",  description="Top-1 SpeciesNet label string"),
    top_score: float = Query(0.0, ge=0.0, le=1.0, description="Confidence score for top-1 label (0.0–1.0)"),
    label_2:   str  = Query("",  description="Top-2 label (optional — used for priority genus scan)"),
    label_3:   str  = Query("",  description="Top-3 label (optional — used for priority genus scan)"),
):
    """
    Show the full GPT routing decision that identifier._should_use_gpt() would produce.

    Returns which of the 6 conditions triggered (or none) and whether GPT vision
    would be called instead of using the SpeciesNet label directly.

    Conditions checked (first match wins):
      #6 — no label at all
      #5 — label is a bare detection token
      #1 — confidence < 0.80
      #3 — genus or family unparseable
      #4 — no valid scientific name
      #2 — priority genus in ANY of top-3 labels
    """
    from agents.layer1.identifier import _parse_taxonomy, _should_use_gpt, CONFIDENCE_THRESHOLD

    all_labels = [lbl for lbl in [top_label, label_2, label_3] if lbl.strip()]
    family, genus, scientific_name, common_name = _parse_taxonomy(top_label or "")
    use_gpt, reason = _should_use_gpt(
        top_label=top_label or None,
        top_score=top_score,
        family=family,
        genus=genus,
        scientific_name=scientific_name,
        all_labels=all_labels,
    )
    return {
        "use_gpt":             use_gpt,
        "reason":              reason,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "parsed_taxonomy": {
            "family":          family,
            "genus":           genus,
            "scientific_name": scientific_name,
            "common_name":     common_name,
        },
        "inputs": {
            "top_label":        top_label or None,
            "top_score":        top_score,
            "all_labels_used":  all_labels,
        },
    }


@app.get("/dev/speciesnet/priority-genera", tags=["speciesnet"])
async def dev_speciesnet_priority_genera():
    """
    List every priority genus and family that triggers GPT vision fallback (condition #2).

    When ANY of the top-3 SpeciesNet classification labels resolves to a genus
    in this list, the identifier always routes to GPT vision regardless of score.
    """
    from agents.layer1.species_list import PRIORITY_SPECIES, PRIORITY_GENERA, PRIORITY_FAMILIES
    return {
        "total_genera":  len(PRIORITY_GENERA),
        "total_families": len(PRIORITY_FAMILIES),
        "by_family":     {fam: sorted(genera) for fam, genera in sorted(PRIORITY_SPECIES.items())},
        "flat_genera":   sorted(PRIORITY_GENERA),
    }


@app.get("/dev/speciesnet/check-genus", tags=["speciesnet"])
async def dev_speciesnet_check_genus(genus: str = Query(..., description="Genus name to check, e.g. Panthera")):
    """
    Check whether a genus will trigger the GPT vision fallback (priority genus condition #2).
    """
    from agents.layer1.species_list import PRIORITY_GENERA, PRIORITY_SPECIES
    genus_cap    = genus.strip().capitalize()
    is_priority  = genus_cap in PRIORITY_GENERA
    family_match = next((fam for fam, genera in PRIORITY_SPECIES.items() if genus_cap in genera), None)
    return {
        "genus":          genus_cap,
        "is_priority":    is_priority,
        "family":         family_match,
        "will_trigger_gpt": is_priority,
        "reason": (
            f"'{genus_cap}' is in PRIORITY_GENERA (family {family_match}) "
            "→ GPT vision is always used when it appears in top-3."
            if is_priority else
            f"'{genus_cap}' is not a priority genus — GPT fallback only if confidence < 80% or parse fails."
        ),
    }


@app.get("/dev/speciesnet/gpt-vision", tags=["speciesnet"])
async def dev_speciesnet_gpt_vision(
    image_url: str = Query(..., description="Publicly accessible image URL to identify"),
):
    """
    Call GPT-5-mini vision directly on an image URL.

    Uses the same gpt_identify() function invoked by the live pipeline when
    SpeciesNet confidence is below threshold or a priority genus appears in top-3.
    Requires Azure OpenAI credentials in .env.
    """
    from agents.layer1.gpt_vision import gpt_identify
    try:
        result = await gpt_identify(image_url)
        return {
            "image_url":       image_url,
            "scientific_name": result.scientific_name,
            "common_name":     result.common_name,
            "confidence":      result.confidence,
            "source":          result.source,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class _PredictionPayload(BaseModel):
    prediction: dict[str, Any]
    image_url:  str = ""


@app.post("/dev/speciesnet/relevancy", tags=["speciesnet"])
async def dev_speciesnet_relevancy(body: _PredictionPayload):
    """
    Run the relevancy gate on a SpeciesNet prediction dict.

    No model needed — purely rule-based logic (detection label + confidence threshold).

    Example body:
    ```json
    {
      "prediction": {
        "detections": [{"label": "animal", "conf": 0.92}],
        "classifications": {
          "classes": ["uuid;mammalia;carnivora;felidae;panthera;tigris;tiger"],
          "scores": [0.91]
        }
      }
    }
    ```
    """
    from agents.layer1.relevancy import check_relevancy
    result = check_relevancy(body.prediction)
    return {
        "is_relevant":         result.is_relevant,
        "reason":              result.reason,
        "top_detection_label": result.top_detection_label,
        "top_detection_score": result.top_detection_score,
    }


@app.post("/dev/speciesnet/identify", tags=["speciesnet"])
async def dev_speciesnet_identify(body: _PredictionPayload):
    """
    Run the full two-stage identification on a SpeciesNet prediction dict.

    Stage 1 (label parse + routing) is purely deterministic — no model, no LLM.
    Stage 2 (GPT vision) fires only when routing conditions are met; Azure credentials
    must be configured if you want to test the GPT fallback path.

    Provide image_url only when you expect GPT vision to be triggered.

    Example body (high-confidence model path — no GPT):
    ```json
    {
      "prediction": {
        "detections": [{"label": "animal", "conf": 0.95}],
        "classifications": {
          "classes": ["uuid;mammalia;carnivora;felidae;neofelis;nebulosa;clouded leopard"],
          "scores": [0.87]
        }
      },
      "image_url": ""
    }
    ```
    """
    from agents.layer1.identifier import identify_species
    try:
        result = await identify_species(body.prediction, body.image_url or "")
        return {
            "scientific_name": result.scientific_name,
            "confidence":      result.confidence,
            "source":          result.source,
            "top_label_raw":   result.top_label_raw,
            "family":          result.family,
            "genus":           result.genus,
            "common_name":     result.common_name,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/dev/speciesnet/full-analysis", tags=["speciesnet"])
async def dev_speciesnet_full_analysis(
    image_url:  str   = Query(..., description="Publicly accessible image URL"),
    latitude:   float = Query(None),
    longitude:  float = Query(None),
):
    """
    Run the complete ImageAnalysisAgent pipeline (cache → SpeciesNet → relevancy → identifier).

    Requires the SpeciesNet model to be loaded. The dev_server does not load the model
    by default — start the main app (uvicorn main:app --port 8000) and call the /predict
    endpoint instead, or supply a pre-built prediction dict to /dev/speciesnet/identify.

    Returns 503 if the SpeciesNet model has not been injected.
    """
    from agents.layer1.image_agent import get_image_agent
    try:
        agent = get_image_agent()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail=(
                "SpeciesNet model not loaded in this process. "
                "Use /dev/speciesnet/identify with a manually supplied prediction dict, "
                "or start the main server (uvicorn main:app) and call /predict."
            ),
        )
    try:
        result = await agent.run(image_url, latitude, longitude)
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/dev/stats/reports", tags=["stats"])
async def dev_stats_reports(
    latitude:  float = Query(..., description="Centre latitude (decimal degrees)"),
    longitude: float = Query(..., description="Centre longitude (decimal degrees)"),
    radius_km: float = Query(50.0, ge=1.0, le=2000.0, description="Radius in kilometres"),
):
    """
    Proxy to GET /stats/reports with radius filtering.
    Returns the full ReportStatsResponse that the frontend consumes:
    totals, unique_species list, recent_sightings (one card per species),
    location breakdowns, and distribution dicts.
    """
    from agents.layer0.stats_router import get_report_stats
    return await get_report_stats(latitude=latitude, longitude=longitude, radius_km=radius_km)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dev_server:app", host="0.0.0.0", port=8001, reload=True)


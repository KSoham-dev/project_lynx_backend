"""
agents/layer2/reporter.py
==========================
Layer 2 — Reporter Agent  (query_type = report)

Responsibilities
----------------
1. get_species_safety_data()  [public, reusable by other agents]
      Single Azure OpenAI GPT call that returns:
        - user_safety_precautions  (4–6 plain-language actionable items)
        - risk_level               (Very High | High | Caution | Low)
        - length                   (e.g. "120-190 cm")
        - lifespan_years           (e.g. "10-15 Years")
      Note: threat_level is derived deterministically from the IUCN red list
      code via _threat_level_from_iucn() — no GPT call needed for this.

2. _generate_pdf_report()  [private]
      ReportLab A4 PDF with:
        - Embedded user-submitted image
        - Scientific / common / genus / family names + confidence score
        - User message (when present)
        - Location enriched with state / district / protected area
        - Safety + precaution guidelines
        - Risk / threat assessment

3. _upload_pdf_blob()  [private]
      Upload PDF bytes to Azure Blob Storage (container: AZURE_REPORTS_CONTAINER_NAME).
      Returns the blob URL.

4. ReporterAgent.run()
      Orchestrates steps 1-3, persists a ReportDocument to Cosmos DB, and
      returns a structured response dict.

Environment variables
---------------------
AZURE_REPORTS_CONTAINER_NAME  – blob container for PDFs (default: "report-pdfs")
AZURE_OPENAI_ENDPOINT         – via config (same as orchestrator)
AZURE_OPENAI_API_KEY          – via config
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from io import BytesIO
from typing import Any, Optional

from agents.enums import IncidentSeverity, QueryType, ReportStatus
from agents.layer0.receiver import AgentRequest
from fastapi import HTTPException
from agents.layer2.iucn_fetcher import (
    iucn_common_names,
    iucn_habitat_names,
    iucn_population_trend,
    iucn_rationale,
    iucn_red_list_category,
    iucn_red_list_code,
    iucn_scientific_name,
    iucn_taxon_field,
    iucn_threat_titles,
    is_iucn_not_found,
)
from agents.state.data_store import upsert_report

logger = logging.getLogger(__name__)

# ── Risk-level colour map (for PDF) ──────────────────────────────────────────
_RISK_COLOURS = {
    "very high": "#C0392B",
    "high":      "#E67E22",
    "caution":   "#F39C12",
    "low":       "#27AE60",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  GPT Safety Data  (public — usable by any agent)
# ═══════════════════════════════════════════════════════════════════════════════

_SAFETY_SYSTEM_PROMPT = """\
You are a wildlife safety advisor for Prahari, India's national wildlife
management system.

Given a species name and optional IUCN context, generate plain-language safety
precautions for the member of the public who reported this sighting.

Return ONLY valid JSON — no markdown fences, no extra text — with this exact structure:
{
  "user_safety_precautions": [
    "...",
    "..."
  ],
  "risk_level": "Very High | High | Caution | Low",
  "length": "<min>-<max> <unit>",
  "lifespan_years": "<min>-<max> Years"
}

user_safety_precautions: 4–6 clear, actionable items written for a non-expert.
  - What to do immediately (back away calmly, do not run, etc.)
  - What NOT to do (do not feed, do not corner, do not use flash)
  - Who to call (local forest department, wildlife warden)
  - Any species-specific behaviour the person should know
  - Be concise — one sentence per item

risk_level — HUMAN SAFETY ONLY, NOT conservation status:
  Very High — actively attacks / kills humans  (tiger, elephant, crocodile, king cobra)
  High      — serious injury if threatened     (sloth bear, leopard, Indian gaur, wild boar)
  Caution   — minor injury possible            (monitor lizard, Indian python, deer, wild dog)
  Low       — negligible human danger          (tortoise, turtle, small bird, most fish)

  A turtle or tortoise is ALWAYS "Low".
  Endangered / Critically Endangered conservation status does NOT raise risk_level.
  risk_level must be exactly one of the four strings above.

length: typical body length as a numeric range with unit, e.g. "120-190 cm" or "30-50 cm".
  Use the most authoritative published range. Unit must be cm or m.

lifespan_years: typical wild lifespan as a numeric range, e.g. "10-15 Years".
  Always capitalise "Years". If only a single well-known value exists, repeat it: "20-20 Years".

Do not add any text outside the JSON object.
"""


# IUCN red list code → conservation threat level (deterministic, no GPT needed)
_IUCN_CODE_TO_THREAT: dict[str, str] = {
    "LC": "Low",
    "NT": "Moderate",
    "VU": "Moderate",
    "EN": "High",
    "CR": "Critical",
    "EW": "Critical",
    "EX": "Critical",
    "DD": "Unknown",
    "NE": "Unknown",
}


def _threat_level_from_iucn(iucn_data: Optional[dict[str, Any]]) -> str:
    """
    Derive conservation threat level deterministically from the IUCN red list code.
    This is always accurate and requires no LLM call.
    """
    code = iucn_red_list_code(iucn_data or {})
    return _IUCN_CODE_TO_THREAT.get(code.upper(), "Unknown")


async def get_species_safety_data(
    scientific_name: str,
    iucn_data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Call GPT-5-mini to generate safety guidelines and precautions for a species.

    Parameters
    ----------
    scientific_name : e.g. "Panthera tigris"
    iucn_data       : optional IUCN dict to give the model context

    Returns
    -------
    dict with keys: safety_guidelines, precaution_guidelines,
                    risk_level, threat_level, emergency_action
    Falls back to an empty dict with placeholder text on any failure.
    """
    from openai import AsyncAzureOpenAI
    from agents.layer0.config import get_settings
    s = get_settings()

    # Build a concise IUCN context snippet — avoid overwhelming the prompt
    iucn_summary = ""
    if iucn_data:
        iucn_summary = (
            f"Red List category: {iucn_red_list_category(iucn_data) or 'Unknown'}\n"
            f"Population trend: {iucn_population_trend(iucn_data) or 'Unknown'}\n"
            f"Top threats: {', '.join(iucn_threat_titles(iucn_data)) or 'Unknown'}\n"
            f"Habitats: {', '.join(iucn_habitat_names(iucn_data)) or 'Unknown'}"
        )

    user_content = (
        f"Species: {scientific_name}\n"
        + (f"\nIUCN context:\n{iucn_summary}" if iucn_summary else "")
    )

    try:
        client = AsyncAzureOpenAI(
            azure_endpoint=s.azure_openai_endpoint,
            api_key=s.azure_openai_api_key,
            api_version="2025-01-01-preview",
            timeout=60.0,
        )
        completion = await asyncio.wait_for(
            client.chat.completions.create(
                model=s.azure_openai_deployment,
                messages=[
                    {"role": "system", "content": _SAFETY_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                max_completion_tokens=2048,
                response_format={"type": "json_object"},
            ),
            timeout=90.0,
        )
        raw = completion.choices[0].message.content or "{}"
        # Extract the JSON object robustly — strip markdown fences if present
        import re as _re
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        if not m:
            logger.warning(
                "[reporter] No JSON found in GPT response for %r: %.200s",
                scientific_name, raw,
            )
            raise ValueError("no_json_in_response")
        result = json.loads(m.group(0))
        if not result.get("user_safety_precautions") and not result.get("risk_level"):
            logger.warning(
                "[reporter] GPT returned empty safety object for %r: %r",
                scientific_name, result,
            )
            raise ValueError("empty_safety_result")
        logger.info(
            "[reporter] Safety data generated for %r: risk=%s precautions=%d",
            scientific_name,
            result.get("risk_level"),
            len(result.get("user_safety_precautions") or []),
        )
        return result
    except Exception as exc:
        logger.error(
            "[reporter] Safety GPT call failed for %r: %s", scientific_name, exc, exc_info=True
        )
        return {
            "user_safety_precautions": [
                "Move away calmly — do not run or make sudden movements.",
                "Keep a safe distance of at least 50 metres from the animal.",
                "Do not attempt to feed, corner, or interact with the animal in any way.",
                "Alert the nearest forest ranger or wildlife warden immediately.",
                "If in immediate danger, make loud noise and back away slowly.",
                "Do not return to the area until cleared by forest department personnel.",
            ],
            "risk_level": "Low",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PDF Report Generation  (private — built with ReportLab)
# ═══════════════════════════════════════════════════════════════════════════════

def _download_image_bytes(url: str) -> Optional[BytesIO]:
    """Download an image URL and return BytesIO, or None on failure."""
    try:
        import requests as _req
        resp = _req.get(url, timeout=15, stream=True)
        resp.raise_for_status()
        buf = BytesIO(resp.content)
        buf.seek(0)
        return buf
    except Exception as exc:
        logger.warning("[reporter] Could not download image %s: %s", url, exc)
        return None


def _fmt_coords(lat: Optional[float], lon: Optional[float]) -> str:
    """Format decimal coordinates into deg-min-sec with correct N/S/E/W."""
    if lat is None or lon is None:
        return "—"

    def _dms(deg: float) -> tuple[int, int, float]:
        d = int(abs(deg))
        m = int((abs(deg) - d) * 60)
        s = (abs(deg) - d - m / 60) * 3600
        return d, m, s

    ld, lm, ls = _dms(lat)
    od, om, os = _dms(lon)
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    return (
        f"{ld}\u00b0{lm}\u2032{ls:.1f}\u2033{lat_dir}, "
        f"{od}\u00b0{om}\u2032{os:.1f}\u2033{lon_dir}"
        f"  ({lat:.5f}, {lon:.5f})"
    )


def _id_source_label(source: Optional[str]) -> str:
    return {
        "model": "SpeciesNet AI Model",
        "gpt":   "GPT Vision Analysis",
    }.get((source or "").lower(), source or "—")


def _detail_table(
    rows: list[list[str]],
    s_label,
    s_value,
    colors,
    col_widths: Optional[list] = None,
) -> Any:
    """Reusable two-column label/value table."""
    from reportlab.platypus import Paragraph, Table, TableStyle
    cw = col_widths or [4.5, 12.5]
    from reportlab.lib.units import cm
    return Table(
        [[Paragraph(r[0], s_label), Paragraph(str(r[1]), s_value)] for r in rows],
        colWidths=[w * cm for w in cw],
        style=TableStyle([
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [colors.HexColor("#F5F5F5"), colors.white]),
            ("BOX",  (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#E0E0E0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ]),
    )


def _build_pdf(report_data: dict[str, Any]) -> bytes:
    """
    Build an official A4 wildlife incident report for forest officers.
    Returns raw PDF bytes. (Synchronous — call via asyncio.to_thread)
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable,
        Image as RLImage,
        KeepTogether,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    # A4 usable width = 21cm - 2.5cm left - 2.5cm right = 16cm
    PAGE_W = 16.0  # cm

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
        title="Prahari Wildlife Incident Report",
    )

    base = getSampleStyleSheet()

    def S(name, **kw) -> ParagraphStyle:
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    DARK_GREEN  = colors.HexColor("#1B4332")
    MID_GREEN   = colors.HexColor("#2D6A4F")
    LIGHT_GREEN = colors.HexColor("#D8F3DC")
    GREY_TEXT   = colors.HexColor("#555555")
    DARK_TEXT   = colors.HexColor("#1A1A1A")
    RULE_COLOR  = colors.HexColor("#AAAAAA")
    WHITE       = colors.white

    risk_colour = colors.HexColor(
        _RISK_COLOURS.get((report_data.get("risk_level") or "low").lower(), "#27AE60")
    )

    s_doc_title = S("DocTitle",  fontSize=18, fontName="Helvetica-Bold",
                    textColor=WHITE, alignment=TA_CENTER, spaceAfter=2)
    s_doc_sub   = S("DocSub",    fontSize=9,  textColor=WHITE,
                    alignment=TA_CENTER, spaceAfter=0)
    s_section   = S("Section",   fontSize=11, fontName="Helvetica-Bold",
                    textColor=DARK_GREEN, spaceBefore=14, spaceAfter=3)
    s_label     = S("Label",     fontSize=9,  fontName="Helvetica-Bold",
                    textColor=GREY_TEXT)
    s_value     = S("Value",     fontSize=9,  textColor=DARK_TEXT)
    s_italic    = S("Italic",    fontSize=9,  fontName="Helvetica-Oblique",
                    textColor=GREY_TEXT)
    s_bullet    = S("Bullet",    fontSize=9,  textColor=DARK_TEXT,
                    leftIndent=16, spaceBefore=3)
    s_obs       = S("Obs",       fontSize=9,  textColor=DARK_TEXT,
                    leftIndent=8, borderPad=6,
                    backColor=colors.HexColor("#F7F7F7"),
                    borderColor=colors.HexColor("#CCCCCC"),
                    borderWidth=0.5, leading=14)
    s_footer    = S("Footer",    fontSize=7,  textColor=colors.HexColor("#888888"),
                    alignment=TA_CENTER)
    s_conf_label = S("ConfLabel", fontSize=7, fontName="Helvetica-Bold",
                    textColor=colors.HexColor("#CC0000"), alignment=TA_RIGHT)

    story = []

    # ── Document header banner ────────────────────────────────────────────────
    header_table = Table(
        [[Paragraph("PRAHARI WILDLIFE MANAGEMENT SYSTEM", s_doc_title)],
         [Paragraph("Wildlife Incident Report — Restricted Official Document", s_doc_sub)]],
        colWidths=[PAGE_W * cm],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), DARK_GREEN),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ]),
    )
    story.append(header_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph("RESTRICTED — For authorised forest department personnel only",
                            s_conf_label))
    story.append(Spacer(1, 6))

    # ── Report metadata ───────────────────────────────────────────────────────
    story.append(Paragraph("Report Details", s_section))
    story.append(HRFlowable(width="100%", thickness=1, color=MID_GREEN, spaceAfter=4))

    severity_val = (report_data.get("severity") or "low").upper()
    meta_rows = [
        ["Report No.",      report_data.get("report_id", "—")],
        ["Date / Time",     report_data.get("submitted_at", "—")],
        ["Reported By",     report_data.get("user_id", "—")],
        ["Incident Severity", severity_val],
        ["Query Type",      "Wildlife Sighting — Incident Report"],
    ]
    story.append(_detail_table(meta_rows, s_label, s_value, colors))

    # ── Sighting / Observation ────────────────────────────────────────────────
    message = report_data.get("message")
    if message:
        story.append(Paragraph("Field Observation Note", s_section))
        story.append(HRFlowable(width="100%", thickness=1, color=MID_GREEN, spaceAfter=4))
        story.append(Paragraph(str(message), s_obs))
        story.append(Spacer(1, 4))

    # ── Species identification ────────────────────────────────────────────────
    story.append(Paragraph("Species Identification", s_section))
    story.append(HRFlowable(width="100%", thickness=1, color=MID_GREEN, spaceAfter=4))

    scientific_name = report_data.get("scientific_name") or "Unidentified"
    common_names    = report_data.get("common_names") or []
    common_name_str = ", ".join(common_names) if isinstance(common_names, list) else str(common_names)
    genus      = report_data.get("genus")   or "—"
    family     = report_data.get("family")  or "—"
    confidence = report_data.get("confidence")
    id_source  = _id_source_label(report_data.get("identification_source"))
    iucn_cat   = report_data.get("red_list_category") or "—"

    # Image — full width above taxonomy, max 8cm tall
    image_url = report_data.get("image_url")
    if image_url:
        img_bytes = _download_image_bytes(image_url)
        if img_bytes:
            try:
                from PIL import Image as PILImage
                _pil = PILImage.open(img_bytes)
                _orig_w, _orig_h = _pil.size
                img_bytes.seek(0)
                _max_w = PAGE_W * cm          # full usable width
                _max_h = 10 * cm              # hard ceiling height
                _scale = min(_max_w / _orig_w, _max_h / _orig_h)
                _img_w = _orig_w * _scale
                _img_h = _orig_h * _scale
                story.append(RLImage(img_bytes, width=_img_w, height=_img_h))
                story.append(Paragraph(
                    "Figure 1: Submitted field photograph",
                    S("Cap", fontSize=7, textColor=GREY_TEXT, alignment=TA_CENTER, spaceBefore=2),
                ))
                story.append(Spacer(1, 6))
            except Exception as exc:
                logger.warning("[reporter] ReportLab image error: %s", exc)

    taxo_rows = [
        ["Scientific Name",      f"{scientific_name}"],
        ["Common Name(s)",        common_name_str or "—"],
        ["Genus",                 genus],
        ["Family",                family],
        ["IUCN Red List",         iucn_cat],
        ["Identification Method", id_source],
    ]
    story.append(_detail_table(taxo_rows, s_label, s_value, colors))

    # ── Location ──────────────────────────────────────────────────────────────
    loc = report_data.get("location") or {}
    loc_has_data = any(v for v in loc.values() if v is not None)
    if loc_has_data:
        story.append(Paragraph("Sighting Location", s_section))
        story.append(HRFlowable(width="100%", thickness=1, color=MID_GREEN, spaceAfter=4))
        loc_rows = []
        coords_str = _fmt_coords(loc.get("latitude"), loc.get("longitude"))
        if coords_str != "—":
            loc_rows.append(["GPS Coordinates", coords_str])
        if loc.get("district") or loc.get("state"):
            loc_rows.append(["District / State",
                              ", ".join(filter(None, [loc.get("district"), loc.get("state")]))])
        if loc.get("country"):
            loc_rows.append(["Country", loc["country"]])
        if loc.get("protected_area"):
            loc_rows.append(["Protected / Reserve Area", loc["protected_area"]])
        if loc.get("formatted"):
            loc_rows.append(["Full Address", loc["formatted"]])
        if loc_rows:
            story.append(_detail_table(loc_rows, s_label, s_value, colors))

    # ── Threat & Risk Assessment ──────────────────────────────────────────────
    story.append(Paragraph("Threat and Risk Assessment", s_section))
    story.append(HRFlowable(width="100%", thickness=1, color=MID_GREEN, spaceAfter=4))

    risk_level   = report_data.get("risk_level")   or "—"
    threat_level = report_data.get("threat_level") or "—"

    # 3-cell summary row
    def _risk_cell(label: str, value: str, bg_colour) -> list:
        ls = S(f"RC_{label}", fontSize=8, fontName="Helvetica-Bold",
               textColor=WHITE, alignment=TA_CENTER)
        vs = S(f"RV_{label}", fontSize=11, fontName="Helvetica-Bold",
               textColor=WHITE, alignment=TA_CENTER)
        return [
            Table([[Paragraph(label, ls)], [Paragraph(value, vs)]],
                  colWidths=[(PAGE_W / 3) * cm],
                  style=TableStyle([
                      ("BACKGROUND", (0, 0), (-1, -1), bg_colour),
                      ("TOPPADDING",    (0, 0), (-1, -1), 8),
                      ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                      ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                      ("BOX", (0, 0), (-1, -1), 0.5, WHITE),
                  ])),
        ]

    risk_summary = Table(
        [[
            _risk_cell("Human Risk Level",       risk_level,   risk_colour)[0],
            _risk_cell("Threat Level",            threat_level, colors.HexColor("#5C4033"))[0],
            _risk_cell("IUCN Conservation Status", iucn_cat,    MID_GREEN)[0],
        ]],
        colWidths=[(PAGE_W / 3) * cm] * 3,
        style=TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]),
    )
    story.append(risk_summary)
    story.append(Spacer(1, 8))

    # ── Safety Precautions (user-facing) ─────────────────────────────────────
    user_precautions = report_data.get("user_safety_precautions") or []
    if user_precautions:
        story.append(Paragraph("Safety Precautions", s_section))
        story.append(HRFlowable(width="100%", thickness=1, color=MID_GREEN, spaceAfter=4))
        story.append(Paragraph(
            "The following safety precautions are recommended for anyone in the vicinity of this sighting:",
            s_italic,
        ))
        story.append(Spacer(1, 4))
        for i, item in enumerate(user_precautions, 1):
            story.append(Paragraph(f"{i}.\xa0\xa0{item}", s_bullet))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=RULE_COLOR, spaceAfter=4))
    story.append(Paragraph(
        f"Prahari Wildlife Management System  \u2022  Report No.: {report_data.get('report_id', '—')}"
        f"  \u2022  Generated: {report_data.get('submitted_at', '—')}"
        f"  \u2022  RESTRICTED",
        s_footer,
    ))

    doc.build(story)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Azure Blob Upload  (private)
# ═══════════════════════════════════════════════════════════════════════════════

def _pdf_blob_name(
    report_id: str,
    scientific_name: str,
    risk_level: Optional[str],
    threat_level: Optional[str],
    fsm_id: str,
) -> str:
    """Build a sanitised PDF blob name in the format:
    reportid_scientificname_risklevel_threatlevel_fsmid_timestamp.pdf
    """
    def _slug(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "-", (s or "unknown").strip()).strip("-")

    from datetime import datetime, timezone
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
    return "_".join([
        report_id,
        _slug(scientific_name),
        _slug(risk_level or "unknown"),
        _slug(threat_level or "unknown"),
        _slug(fsm_id),
        ts,
    ]) + ".pdf"


def _upload_pdf_sync(pdf_bytes: bytes, blob_name: str) -> Optional[str]:
    """
    Upload PDF bytes to Azure Blob Storage.
    Returns the blob URL on success, None on failure.
    Runs synchronously — call via asyncio.to_thread.
    """
    from pipeline.species_traits import _build_blob_service_client  # reuse helper

    container_name = os.getenv("AZURE_REPORTS_CONTAINER_NAME", "report-pdfs")

    try:
        service = _build_blob_service_client()
        container_client = service.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)
        from azure.storage.blob import ContentSettings
        blob_client.upload_blob(
            pdf_bytes,
            overwrite=True,
            content_settings=ContentSettings(content_type="application/pdf"),
        )
        url = blob_client.url
        logger.info("[reporter] PDF uploaded: %s", url)
        return url
    except Exception as exc:
        logger.error("[reporter] PDF upload failed for %s: %s", blob_name, exc, exc_info=True)
        return None


async def _upload_pdf_blob(pdf_bytes: bytes, blob_name: str) -> Optional[str]:
    """Async wrapper for PDF blob upload."""
    return await asyncio.to_thread(_upload_pdf_sync, pdf_bytes, blob_name)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Reporter Agent
# ═══════════════════════════════════════════════════════════════════════════════

class ReporterAgent:
    """
    Full incident report pipeline:
      iucn_data + safety GPT → PDF → blob upload → Cosmos save → response dict.
    """

    async def run(
        self,
        request: AgentRequest,
        tool_results: dict[str, Any],
        iucn_data: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        request      : normalised AgentRequest
        tool_results : Layer 1 outputs
        iucn_data    : output of SpeciesInfoAgent — may be None

        Returns
        -------
        Response dict with report_id, pdf_url, safety data, species info.
        """
        from datetime import datetime, timezone

        image_data: dict = tool_results.get("analyse_image") or {}
        text_data:  dict = tool_results.get("analyse_text")  or {}
        loc_data:   dict = tool_results.get("location_context") or {}

        # Resolve the best available names
        scientific_name: str = (
            iucn_scientific_name(iucn_data or {})
            or image_data.get("scientific_name")
            or (text_data.get("scientific_name") if isinstance(text_data, dict) else None)
            or "Unknown"
        )
        common_names = (
            iucn_common_names(iucn_data or {})
            or ([image_data.get("common_name")] if image_data.get("common_name") else [])
        )

        # Guard: if both scientific name and common name are unresolved, abort.
        _sci_unknown = not scientific_name or scientific_name.lower() in ("unknown", "unidentified")
        _common_unknown = not common_names or all(
            not n or n.lower() in ("unknown", "unidentified") for n in common_names
        )
        if is_iucn_not_found(iucn_data):
            logger.warning(
                "[reporter] Species not in any database: %r",
                iucn_data.get("scientific_name"),
            )
            raise HTTPException(
                status_code=404,
                detail=iucn_data["message"],
            )

        if not iucn_data and _sci_unknown and _common_unknown:
            logger.warning(
                "[reporter] Aborting — species could not be identified (scientific=%r, common_names=%r) session=%s",
                scientific_name, common_names, request.session_id,
            )
            raise HTTPException(
                status_code=404,
                detail="Could not identify the species from the provided input. Please retake the photo with the animal clearly visible and try again.",
            )

        report_id    = str(uuid.uuid4())
        submitted_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        logger.info(
            "[reporter] START session=%s species=%r report_id=%s",
            request.session_id, scientific_name, report_id,
        )

        # ── Fetch safety data + iNaturalist photo concurrently ──────────────────
        from agents.layer2.inaturalist import get_inaturalist_photo

        safety_data: dict[str, Any] = {}
        inaturalist_photo_url: Optional[str] = None
        inaturalist_photo_credit: Optional[str] = None

        if scientific_name.lower() not in ("unknown", "unidentified"):
            safety_task = get_species_safety_data(scientific_name, iucn_data)
            # Use the primary IUCN common name for the iNaturalist lookup so the
            # photo result is more recognisable to end users (e.g. "Tiger" rather
            # than "Panthera tigris").  Fall back to scientific name if absent.
            photo_task  = get_inaturalist_photo(scientific_name)
            (safety_data, (inaturalist_photo_url, inaturalist_photo_credit)) = await asyncio.gather(
                safety_task, photo_task
            )

        # threat_level is derived deterministically from IUCN red list code —
        # never from GPT, which confuses conservation status with human danger.
        threat_level_val = _threat_level_from_iucn(iucn_data)

        # ── Severity: from text_agent if present, else infer from human risk level ──
        text_severity = (text_data.get("severity") or "") if isinstance(text_data, dict) else ""
        if text_severity:
            incident_severity = _map_severity(text_severity)
        else:
            # No user text — infer severity from the human risk level returned by GPT
            _risk_to_sev = {"very high": "critical", "high": "high", "caution": "medium"}
            _inferred_sev = _risk_to_sev.get((safety_data.get("risk_level") or "").lower(), "low")
            incident_severity = _map_severity(_inferred_sev)

        # ── Assemble PDF data dict ────────────────────────────────────────────
        # Genus/family: prefer SpeciesNet image result, fall back to IUCN taxonomy
        # iucn_taxon_field accepts short aliases (genus, family) and resolves to
        # the actual blob key names (genus_name, family_name).
        iucn_raw   = iucn_data or {}
        genus_val  = image_data.get("genus")  or iucn_taxon_field(iucn_raw, "genus")
        family_val = image_data.get("family") or iucn_taxon_field(iucn_raw, "family")

        pdf_data: dict[str, Any] = {
            "report_id":             report_id,
            "submitted_at":          submitted_at,
            "user_id":               request.user_id,
            "scientific_name":       scientific_name,
            "common_names":          common_names,
            "genus":                 genus_val,
            "family":                family_val,
            "confidence":            image_data.get("confidence"),
            "identification_source": image_data.get("identification_source"),
            "image_url":             request.image_url or image_data.get("image_url"),
            "message":               request.message or None,
            "severity":              incident_severity.value,
            "location": {
                "latitude":       request.latitude,
                "longitude":      request.longitude,
                "district":       loc_data.get("district"),
                "state":          loc_data.get("state"),
                "country":        loc_data.get("country"),
                "protected_area": loc_data.get("protected_area"),
                "formatted":      loc_data.get("formatted"),
            },
            # IUCN fields for risk section
            "red_list_category":       iucn_red_list_category(iucn_data or {}),
            # Safety GPT fields
            "user_safety_precautions": safety_data.get("user_safety_precautions") or [],
            "risk_level":              safety_data.get("risk_level"),
            "threat_level":            threat_level_val,
        }

        # ── Generate PDF (sync, run in thread) ───────────────────────────────
        pdf_url: Optional[str] = None
        try:
            pdf_bytes = await asyncio.to_thread(_build_pdf, pdf_data)
            logger.info("[reporter] PDF generated: %d bytes", len(pdf_bytes))
            blob_name = _pdf_blob_name(
                report_id,
                scientific_name,
                safety_data.get("risk_level"),
                threat_level_val,
                request.user_id,
            )
            pdf_url = await _upload_pdf_blob(pdf_bytes, blob_name)
        except Exception as exc:
            logger.error("[reporter] PDF generation failed: %s", exc, exc_info=True)

        # ── Save ReportDocument to Cosmos ─────────────────────────────────────
        report_doc = {
            "id":              report_id,
            "user_id":         request.user_id,
            "session_id":      request.session_id,
            "query_type":      QueryType.REPORT.value,
            "scientific_name": scientific_name,
            "common_name":     common_names[0] if common_names else None,
            "family":          family_val,
            "genus":           genus_val,
            "confidence":      image_data.get("confidence") or None,
            "identification_source": image_data.get("identification_source"),
            "risk_level":      safety_data.get("risk_level"),
            "threat_level":    threat_level_val,
            "red_list_category": iucn_red_list_category(iucn_data or {}),
            "status":          ReportStatus.SUBMITTED.value,
            "severity":        incident_severity.value,
            "pdf_url":         pdf_url,
            "location": {
                "latitude":       request.latitude,
                "longitude":      request.longitude,
                "district":       loc_data.get("district"),
                "state":          loc_data.get("state"),
                "country":        loc_data.get("country"),
                "protected_area": loc_data.get("protected_area"),
                "formatted":      loc_data.get("formatted"),
            },
            "image_url": inaturalist_photo_url or request.image_url or image_data.get("image_url"),
            "animal_traits": {
                "size":      text_data.get("size")      if isinstance(text_data, dict) else None,
                "colour":    text_data.get("colour")    if isinstance(text_data, dict) else None,
                "behaviour": text_data.get("behaviour") if isinstance(text_data, dict) else None,
                "distinctive_features": (
                    text_data.get("distinctive_features") if isinstance(text_data, dict) else []
                ),
            },
            "message": request.message,
        }
        try:
            await upsert_report(report_doc)
            logger.info("[reporter] Report saved to Cosmos: id=%s species=%s", report_id, scientific_name)
        except Exception as exc:
            logger.error("[reporter] Cosmos save failed: %s", exc, exc_info=True)

        # ── Return structured response ────────────────────────────────────────
        response = {
            "report_id":   report_id,
            "pdf_url":     pdf_url,
            "submitted_at": submitted_at,
            # Species
            "scientific_name":       scientific_name,
            "common_names":          common_names,
            "genus":                 genus_val,
            "family":                family_val,
            "confidence":            image_data.get("confidence") or None,
            "identification_source": image_data.get("identification_source"),
            # IUCN
            "red_list_category":     iucn_red_list_category(iucn_data or {}),
            "red_list_code":         iucn_red_list_code(iucn_data or {}),
            "population_trend":      iucn_population_trend(iucn_data or {}),
            "rationale":             iucn_rationale(iucn_data or {}),
            # Safety
            "risk_level":              safety_data.get("risk_level"),
            "threat_level":            threat_level_val,
            "user_safety_precautions": safety_data.get("user_safety_precautions") or [],
            # Physical traits
            "length":         safety_data.get("length"),
            "lifespan_years": safety_data.get("lifespan_years"),
            # Incident
            "severity":  incident_severity.value,
            "location":  pdf_data["location"],
            "message":   request.message,
            "image_url": inaturalist_photo_url or request.image_url or image_data.get("image_url"),
            # Source links
            "iucn_url":                  (iucn_data or {}).get("url"),
            "inaturalist_photo_credit":  inaturalist_photo_credit,
        }
        logger.info("[reporter] DONE: report_id=%s pdf_url=%s", report_id, pdf_url)

        # ── Trigger Incident Agent (log + notify) ─────────────────────────────
        try:
            from agents.layer2.incident_agent import IncidentAgent
            await IncidentAgent().run(response, request)
        except Exception as exc:
            logger.error("[reporter] IncidentAgent.run failed: %s", exc, exc_info=True)

        return response


# ── Helpers ───────────────────────────────────────────────────────────────────

def _map_severity(text_severity: str) -> IncidentSeverity:
    return {
        "critical": IncidentSeverity.CRITICAL,
        "high":     IncidentSeverity.HIGH,
        "medium":   IncidentSeverity.MEDIUM,
    }.get((text_severity or "").lower(), IncidentSeverity.LOW)

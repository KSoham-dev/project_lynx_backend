# Prahari Backend — In-Depth Architecture & Technical Documentation

> **System Purpose:** Prahari is an Indian wildlife management backend for forest rangers. It accepts images, GPS coordinates, and/or wildlife text messages, runs a multi-layer AI agent pipeline, and produces one of three structured outputs: an **Incident Report** (PDF), a **Species Explorer** profile, or an **Encyclopedia** entry.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Infrastructure Dependencies](#infrastructure-dependencies)
3. [Cosmos DB Databases](#cosmos-db-databases)
4. [Caching Strategy](#caching-strategy)
5. [Entry Point — main.py](#entry-point--mainpy)
6. [Layer 0 — Orchestration Layer](#layer-0--orchestration-layer)
   - [Receiver](#step-1-receiver)
   - [Orchestrator](#step-2-orchestrator)
   - [Router (HTTP Adapter)](#router--http-adapter)
7. [Layer 1 — Analysis Agents](#layer-1--analysis-agents)
   - [Image Analysis Agent](#image-analysis-agent)
   - [Relevancy Check](#relevancy-check)
   - [Species Identifier](#species-identifier)
   - [GPT Vision Fallback](#gpt-vision-fallback)
   - [Text Analysis Agent](#text-analysis-agent)
   - [Location Agent](#location-agent)
8. [Layer 2 — Output Agents](#layer-2--output-agents)
   - [SpeciesInfoAgent (Always First)](#speciesinfoagent--always-first)
   - [Reporter Agent](#reporter-agent)
   - [Explorer Agent](#explorer-agent)
   - [Encyclopedia Agent](#encyclopedia-agent)
9. [State Layer — Persistence](#state-layer--persistence)
10. [Pipeline — species_traits.py](#pipeline--species_traitspy)
11. [Shared Enumerations](#shared-enumerations)
12. [Data Schemas Reference](#data-schemas-reference)
13. [Complete End-to-End Data Flow](#complete-end-to-end-data-flow)
14. [Technologies Used](#technologies-used)
15. [Agent × Technology Matrix](#agent--technology-matrix)

---

## System Overview

```
Ranger App (React / Mobile)
          │
          │  POST /agent/chat
          │  { user_id, query_type, message?, image_url?, latitude?, longitude? }
          │  Headers: X-Device-ID, X-FCM-Token, User-Agent
          ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 0 — Receiver                                     │
│  Validate + normalise → load/create session             │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 0 — Orchestrator                                 │
│                                                         │
│  1st GPT call  →  tool selection                        │
│  asyncio.gather → parallel Layer 1 execution            │
│  Relevancy gate                                         │
│  2nd GPT call  →  Layer 2 routing (forced tool_choice)  │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 1  (all run concurrently)                        │
│  Image Analysis │ Text Analysis │ Location Geocoding    │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 2 — SpeciesInfoAgent (always first)              │
│  then one of: Reporter │ Explorer │ Encyclopedia         │
└─────────────────────────────────────────────────────────┘
          │
          ▼
     Structured JSON Response
```

---

## Infrastructure Dependencies

| Service | Purpose |
|---|---|
| **Azure OpenAI (GPT-4o-mini)** | Orchestration tool selection, L2 routing, GPT vision species ID fallback, text analysis, safety report generation |
| **Azure Cosmos DB** (2 databases) | Session state, all caches, users, reports, incidents, SOS events |
| **Azure Blob Storage** | Pre-processed IUCN species JSON files + PDF report uploads |
| **Azure DefaultAzureCredential** | Managed Identity authentication on Azure Container Apps (production) |
| **Groq (llama-3.3-70b-versatile)** | Species trait extraction via `run_pipeline()` |
| **SpeciesNet (PyTorch)** | On-device wildlife detection + classification, loaded once at startup |
| **LangChain** | Tool definitions, message history format, StructuredTool wrappers |
| **LangSmith** | Tracing and span management for orchestrator calls |
| **Nominatim (OpenStreetMap)** | GPS reverse geocoding |
| **iNaturalist API** | Representative species photos for all three output types |
| **IUCN Red List API v4** | Live fallback when species blob not found in Azure Blob |
| **Wikipedia API** | Species text context fed into Groq prompt |

---

## Cosmos DB Databases

### `prahari-state` — Operational State

| Container | Key | Contents | TTL |
|---|---|---|---|
| `sessions` | `session_id` | Full conversation history per user session | None |
| `agent_traces` | auto | Per-agent invocation logs (latency, success, I/O) | None |
| `image_analysis_cache` | MD5(image_url) | SpeciesNet prediction results | 7 days |
| `species_context_cache` | `iucn:<name>` / `traits:<name>` | IUCN raw data + GPT/Groq traits | 24 hours |
| `location_cache` | geohash (precision 5) | Nominatim geocode results (~5 km² cells) | 30 days |

### `prahari-data` — Persistent Business Data

| Container | Key | Contents |
|---|---|---|
| `users` | `user_id` | User profiles with role, forest zone, district, device FCM tokens |
| `reports` | `report_id` | Incident report records + PDF blob URLs |
| `explorations` | `exploration_id` | Species exploration sessions with multiple observations |
| `encyclopedia` | `species_id` | Permanent species knowledge base entries |
| `incidents` | `incident_id` | Formal incidents with type, severity, status lifecycle |
| `sos_events` | `sos_id` | SOS alerts with GPS, device ID, notification tracking, status lifecycle |

---

## Caching Strategy

| Cache Level | Key | TTL | Backend |
|---|---|---|---|
| Image analysis results | `MD5(image_url)` | 7 days | Cosmos `image_analysis_cache` |
| IUCN species data | `iucn:<normalized_name>` | 24 hours | Cosmos `species_context_cache` |
| Species traits (Groq) | `traits:<normalized_name>` | 24 hours | Cosmos `species_context_cache` |
| Location geocode | `geohash` at precision 5 | 30 days | Cosmos `location_cache` |
| `run_pipeline()` output | scientific name (lowercase) | 24 hours | In-memory `TTLCache` (maxsize 2048) |

All Cosmos caches use native Cosmos TTL (`ttl` field in document). Cache misses fall through to the live source automatically.

---

## Entry Point — `main.py`

### FastAPI Application Lifespan

**On startup:**
1. Loads `SpeciesNet` PyTorch model from `./model` directory.
2. Injects model into `ImageAnalysisAgent` via `set_speciesnet_model()`.
3. Silences verbose Azure/httpx/urllib3 log noise to WARNING-only.

**On shutdown:**
1. Calls `close_cosmos_clients()` — gracefully closes all Cosmos DB connection pools.

### HTTP Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health-check welcome message |
| `GET` | `/health` | Returns `{ "model_loaded": true/false }` |
| `POST` | `/agent/chat` | **Main agent pipeline** — mounted from Layer 0 router |
| `POST` | `/predict` | Raw single-image SpeciesNet classification |
| `POST` | `/predict/batch` | Batch SpeciesNet classification (max ~16 images) |
| `GET` | `/species/traits?scientific_name=` | Direct species traits pipeline via `run_pipeline()` |

### `/predict` Input/Output

```json
// Input: PredictRequest
{ "image_url": "...", "latitude": 20.5, "longitude": 78.9 }

// Output: PredictionResponse
{
  "top_prediction": { "label": "Panthera tigris", "score": 0.92 },
  "classifications": [ { "label": "...", "score": 0.92 }, ... ]
}
```

---

## Layer 0 — Orchestration Layer

Files: `agents/layer0/`

### Step 1: Receiver

**File:** `agents/layer0/receiver.py`

The front door of the agent system — validates, normalises, and enriches the raw frontend payload.

#### Input Schema — `ReceiverInput`

| Field | Type | Required | Description |
|---|---|---|---|
| `user_id` | string | ✅ | Unique user identifier |
| `query_type` | `QueryType` | ✅ | `report`, `explore`, or `encyclopedia` |
| `session_id` | string | ❌ | Existing session UUID; auto-generated if absent |
| `message` | string | ❌ | Ranger's text description |
| `image_url` | string | ❌ | URL to the wildlife image |
| `latitude` | float | ❌ | GPS latitude |
| `longitude` | float | ❌ | GPS longitude |

#### Output Schema — `AgentRequest`

Enriched request passed downstream to the orchestrator:

| Field | Source |
|---|---|
| `session_id` | Provided or UUID4 generated |
| `session` | Loaded `SessionDocument` from Cosmos or newly created |
| `device_info` | Extracted from HTTP headers (`X-Device-ID`, `X-FCM-Token`, `User-Agent`) |
| `geo_context` | `{ "latitude": ..., "longitude": ... }` |

#### Processing Steps

```
1. Extract DeviceContext from HTTP headers
2. Resolve session_id   (provided → or auto UUID4)
3. Load SessionDocument from prahari-state/sessions
   └── if not found → create_session()
4. Return fully-populated AgentRequest
```

---

### Step 2: Orchestrator

**File:** `agents/layer0/orchestrator.py`

The central coordinator of the entire multi-agent pipeline. Uses LangChain tooling with Azure OpenAI.

#### LangChain Tool Definitions (Layer 1 Tools)

| Tool Name | Calls | Arguments |
|---|---|---|
| `analyse_image` | `ImageAnalysisAgent.run()` | `image_url`, `latitude?`, `longitude?` |
| `analyse_text` | `TextAnalysisAgent.run()` | `message` |
| `location_context` | `get_location_context()` | `latitude`, `longitude` |

#### `_available_tools(request)` — Dynamic Filtering

Tools are offered to the LLM only when their required inputs are present:
- No `image_url` → `analyse_image` excluded
- No `message` → `analyse_text` excluded
- No `latitude`+`longitude` → `location_context` excluded

#### Step-by-Step Orchestration Flow

**1st LLM Call — Tool Selection**
```
system: _TOOL_SELECTION_PROMPT
        → "select only tools whose inputs are present; do not call tools with missing inputs"
options: tool_choice="required", parallel_tool_calls=True
retry:   if 0 tool calls returned → retry once with correction message
```

**Concurrent Layer 1 Execution**
```python
results = await asyncio.gather(
    analyse_image(image_url, lat, lon),   # SpeciesNet + identifier
    analyse_text(message),                 # GPT text analysis
    location_context(lat, lon)             # Nominatim geocoding
)
```

**Session Persistence**
- User message persisted to Cosmos session at this point.

**Relevancy Gate**
```
if image.is_relevant == False AND message is None:
    → return early: "Please retake the photo / ensure wildlife is visible"
    (no Layer 2 invoked)
```

**2nd LLM Call — Layer 2 Routing**
```
system: _LAYER2_ROUTING_PROMPT
        → "route to the appropriate Layer 2 tool based on query_type"
options: tool_choice=<forced to query_type tool name>
tools:   3 StructuredTool closures (each closes over request + tool_results):
         - REPORT       → runs ReporterAgent
         - EXPLORE      → runs ExplorerAgent
         - ENCYCLOPEDIA → runs EncyclopediaAgent
fallback: _direct_dispatch_layer2() if LLM routing fails
```

**SpeciesInfoAgent always runs before the query-specific agent**, feeding IUCN data downstream.

---

### Router — HTTP Adapter

**File:** `agents/layer0/router.py`

Thin FastAPI adapter. Mounts at `/agent`.

```
POST /agent/chat
    → run_receiver()
    → run_orchestrator()
    → ChatResponse { session_id, user_id, query_type, response, data }

Error mapping:
    ValueError   → HTTP 400
    RuntimeError → HTTP 503
    others       → HTTP 500
```

---

## Layer 1 — Analysis Agents

Files: `agents/layer1/`

All three agents run **concurrently** via `asyncio.gather` in the orchestrator.

---

### Image Analysis Agent

**File:** `agents/layer1/image_agent.py`

Full two-stage pipeline for a single image URL.

#### Pipeline

```
image_url
    │
    ├── 1. Cache check (MD5 hash → Cosmos image_analysis_cache)
    │       └── cache hit → return ImageAnalysisResult (cached=True)
    │
    ├── 2. SpeciesNet.predict(image_url)    [asyncio.to_thread — non-blocking]
    │       └── returns: detections[], classifications[]
    │
    ├── 3. check_relevancy(prediction)
    │       └── returns: RelevancyResult { is_relevant, reason, top_label, top_score }
    │
    ├── 4. identify_species(prediction, image_url)
    │       ├── Stage A: parse SpeciesNet label
    │       └── Stage B: GPT vision fallback (if triggered)
    │
    └── 5. Write to Cosmos cache (TTL 7 days)
            └── return ImageAnalysisResult
```

#### `ImageAnalysisResult` Fields

| Field | Description |
|---|---|
| `scientific_name` | e.g., `"Panthera tigris"` |
| `confidence` | 0.0–1.0 |
| `source` | `"model"` / `"gpt"` / `"gpt_error"` |
| `is_relevant` | Bool — wildlife in image? |
| `reason` | Human-readable relevancy explanation |
| `top_detection_label` | Raw SpeciesNet detection label |
| `top_detection_score` | Detection confidence score |
| `latitude`, `longitude` | Passed through from request |
| `family`, `genus`, `common_name` | Parsed from SpeciesNet label |
| `cached` | Bool — was this a cache hit? |

---

### Relevancy Check

**File:** `agents/layer1/relevancy.py`

Decision tree operating on `SpeciesNet.detections[0]`:

```
detections is empty?
    └── check classifications as fallback
        ├── classifications exist → relevant (True)
        └── no classifications   → not relevant (False)

top detection label in { blank, vehicle, human, person, no_cv_result }?
    └── not relevant (False)

top detection confidence < 0.15?
    └── not relevant (False)

otherwise → relevant (True)
```

**`DETECTION_SCORE_THRESHOLD` = 0.15**

---

### Species Identifier

**File:** `agents/layer1/identifier.py`

**`CONFIDENCE_THRESHOLD` = 0.80**

#### SpeciesNet Label Format

```
"uuid;class;order;family;genus;epithet;common_name"
```
Parsed by `_parse_taxonomy()` → `(family, genus, scientific_name, common_name)`.

#### 6 GPT Fallback Trigger Conditions (`_should_use_gpt`)

| # | Condition | Reason |
|---|---|---|
| 1 | No label returned | No model result |
| 2 | Top label is a detection token (human/vehicle) | Not a species label |
| 3 | Confidence < 0.80 | Low confidence |
| 4 | Unparseable genus or family | Malformed taxonomy label |
| 5 | No scientific name extracted | Incomplete label |
| 6 | Priority genus appears in top-3 predictions | High-stakes Indian wildlife species |

#### Priority Species (`agents/layer1/species_list.py`)

10 Indian wildlife families with priority genera:

| Family | Genera |
|---|---|
| `FELIDAE` | Panthera, Neofelis, Prionailurus |
| `ELEPHANTIDAE` | Elephas |
| `URSIDAE` | Melursus, Ursus |
| `CROCODYLIDAE` | Crocodylus, Gavialis |
| `ELAPIDAE` | Naja, Ophiophagus |
| `VIPERIDAE` | Daboia, Bungarus |
| `BOVIDAE` | Bos, Bubalus |
| `CANIDAE` | Cuon |
| ... | ... |

`PRIORITY_GENERA` is a flat `frozenset` for O(1) lookup.

---

### GPT Vision Fallback

**File:** `agents/layer1/gpt_vision.py`

- Uses raw `AsyncAzureOpenAI` client (`api_version="2025-01-01-preview"`).
- `response_format={"type": "json_object"}`.
- System prompt: expert zoologist for the Indian subcontinent.
- Returns JSON: `{ "species_name": "Genus species", "common_name": "..." }`.
- `_normalise_scientific_name()`: enforces strict two-word `"Genus species"` format — returns `"Unknown"` for invalid names.
- On any error → `IdentificationResult(source="gpt_error")`.

---

### Text Analysis Agent

**File:** `agents/layer1/text_agent.py`

Single GPT call on the ranger's free-text message.

#### Output — `TextAnalysisResult`

| Field | Description |
|---|---|
| `scientific_name?` | Species scientific name if mentioned |
| `common_name?` | Common name if mentioned |
| `severity` | `SeverityLevel` enum value |
| `identification_confidence` | 0–100 score |
| `size?` | Creature size if mentioned |

- Lazy singleton `AzureChatOpenAI` via `get_text_agent()`.
- Falls back to `SeverityLevel.LOW` on any parse error.
- `response_format={"type": "json_object"}`.

---

### Location Agent

**File:** `agents/layer1/location.py`

GPS reverse geocoding with Cosmos cache.

#### Pipeline

```
(latitude, longitude)
    │
    ├── 1. Compute geohash at precision 5 (~5 km² cells) as cache key
    ├── 2. Check Cosmos location_cache
    │       └── cache hit → return cached geocode dict
    ├── 3. HTTP GET → Nominatim OSM reverse geocoding API
    ├── 4. _parse_nominatim():
    │       extracts: district, state, country, protected_area, formatted
    │       protected_area detection: extratags, address tags
    │       (nature_reserve, national_park, etc.), display_name phrases
    └── 5. Write to Cosmos cache (TTL 30 days)
            └── return { district, state, country, protected_area,
                         geohash, latitude, longitude, formatted }
```

---

## Layer 2 — Output Agents

Files: `agents/layer2/`

All Layer 2 agents receive: `request`, `tool_results` (Layer 1 outputs), `iucn_data` (from `SpeciesInfoAgent`).

---

### SpeciesInfoAgent — Always First

**File:** `agents/layer2/species_info.py`

Resolves IUCN conservation data before any query-specific agent runs.

#### Scientific Name Priority

```
1. analyse_image result → scientific_name  (if not "Unknown"/"Unidentified")
2. analyse_text result  → scientific_name
3. None                 → _not_in_database_response() sentinel
```

#### Pipeline

```
scientific_name
    │
    ├── 1. Check species_context_cache (key: "iucn:<normalized_name>")
    │       └── cache hit → return cached IUCN dict
    │
    ├── 2. Azure Blob fetch (pre-processed IUCN JSON)
    │       └── get_iucn_raw(scientific_name) from iucn_fetcher.py
    │
    ├── 3. [FileNotFoundError] → live IUCN API v4 fallback
    │       _fetch_iucn_api():
    │           A. resolve scientific_name → assessment_id
    │           B. fetch full IUCN assessment JSON
    │
    ├── 4. [total failure] → _not_in_database_response() sentinel
    │       → { "_not_in_database": True, "message": "Species not found..." }
    │
    └── 5. Write to Cosmos cache (TTL 24h)
            └── return raw IUCN dict
```

#### IUCN Accessor Functions (`agents/layer2/iucn_fetcher.py`)

| Function | Returns |
|---|---|
| `iucn_scientific_name(raw)` | From `raw.taxon.scientific_name` |
| `iucn_common_names(raw)` | List where `main=True` |
| `iucn_red_list_category(raw)` | English description |
| `iucn_red_list_code(raw)` | `"LC"`, `"EN"`, `"CR"`, etc. |
| `iucn_population_trend(raw)` | Increasing / Decreasing / Stable |
| `iucn_threat_titles(raw, limit=4)` | Top threat titles |
| `iucn_habitat_names(raw, limit=4)` | Top habitat names |
| `iucn_rationale(raw)` | Assessment rationale text |
| `is_iucn_not_found(iucn_data)` | `True` if sentinel dict |

---

### Reporter Agent

**File:** `agents/layer2/reporter.py`

**Query type:** `report`

The most complex agent — produces a formal incident report PDF and persists it.

#### Concurrent Phase

```python
safety_data, photo = await asyncio.gather(
    get_species_safety_data(scientific_name, iucn_data),
    get_inaturalist_photo(scientific_name)
)
```

**`get_species_safety_data()` GPT output:**

| Field | Description |
|---|---|
| `user_safety_precautions` | 4–6 actionable safety items |
| `risk_level` | `HumanRiskLevel` (Very High / High / Caution / Low) |
| `length` | Species length string |
| `lifespan_years` | Species lifespan |

#### Threat Level — Deterministic from IUCN Code

```
LC  → Low
NT/VU → Moderate
EN  → High
CR/EW/EX → Critical
```
No LLM involved — purely rule-based.

#### PDF Generation (`_build_pdf` via ReportLab, A4)

| Section | Contents |
|---|---|
| Header banner | Prahari logo, report title |
| Report metadata | Report ID, ranger ID, timestamp, query type |
| Field observation | Ranger's text message, severity badge |
| Species identification | Embedded wildlife image, method label, confidence %, scientific + common name |
| Sighting location | GPS coordinates (decimal + DMS format), district, state, protected area |
| Threat & risk assessment | 3-cell colour table: IUCN category, population trend, human risk level |
| Safety precautions | Numbered list of GPT-generated precautions |
| Footer | Report reference, submission timestamp |

#### Storage & Persistence

```
PDF bytes
    → _upload_pdf_blob()
        → Azure Blob "report-pdfs/<report_id>.pdf"
        → returns blob URL

ReportDocument
    → upsert_report()
        → prahari-data/reports
```

#### Final Response Keys

```json
{
  "report_id": "...",
  "pdf_url": "https://...",
  "scientific_name": "Panthera tigris",
  "common_name": "Bengal Tiger",
  "risk_level": "Very High",
  "threat_level": "Critical",
  "safety_precautions": ["...", "..."],
  "iucn_category": "EN",
  "population_trend": "Decreasing",
  "threats": ["Poaching", "Habitat Loss"],
  "habitats": ["Tropical Forest"],
  "photo_url": "...",
  "photo_credit": "..."
}
```

---

### Explorer Agent

**File:** `agents/layer2/explorer.py`

**Query type:** `explore`

Produces a rich species profile card.

#### Concurrent Phase

```python
gpt_traits, (photo_url, photo_credit) = await asyncio.gather(
    _get_gpt_traits(scientific_name, iucn_data),
    get_inaturalist_photo(scientific_name)
)
```

**`_get_gpt_traits()` — Trait Extraction Flow:**

```
1. Check species_context_cache (key: "traits:<name>")
2. run_pipeline() [in thread — Groq llama-3.3-70b]
   → lifespan_years, mass, length, short_description,
     human_risk_level, human_threat_level, fun_fact_1/2/3
3. Write to Cosmos cache (TTL 24h)
```

#### Final Response

```python
{
    **iucn_data,                   # Full IUCN assessment
    **gpt_overlay,                 # 9 Groq-extracted traits
    "photo_url": ...,
    "photo_credit": ...,
    "location": {
        "district": ...,
        "state": ...,
        "protected_area": ...,
        ...
    }
}
```

---

### Encyclopedia Agent

**File:** `agents/layer2/encyclopedia.py`

**Query type:** `encyclopedia`

Simplest agent — no additional LLM calls required.

```
if is_iucn_not_found(iucn_data):
    → return { "message": "Species not in our database" }

if iucn_data is None:
    → return { "error": "Could not retrieve species data" }

get_inaturalist_photo(scientific_name)
    → return { **iucn_data, "photo_url": ..., "photo_credit": ... }
```

---

## State Layer — Persistence

Files: `agents/state/`

### Cosmos Client (`cosmos_client.py`)

Two separate `CosmosClient` singletons:

```
_state_store  →  prahari-state
_data_store   →  prahari-data
```

Lazy initialisation with module-level caching. `close_cosmos_clients()` called on FastAPI shutdown.

### Session Store (`session_store.py`)

| Function | Description |
|---|---|
| `create_session(user_id, query_type, session_id?, metadata?)` | Creates new `SessionDocument` |
| `get_session(session_id)` | Returns `SessionDocument \| None` |
| `upsert_session(doc)` | Stamps `updated_at`, upserts |
| `append_message(session_id, role, content, tool_name?, layer?)` | Loads, appends `MessageRecord`, upserts |

### Data Store (`data_store.py`)

| Function | Container | Operation |
|---|---|---|
| `get_user / upsert_user / register_device` | `users` | Read / write / auto-create |
| `create_report / get_report / update_report_status / upsert_report` | `reports` | Full CRUD |
| `create_exploration / get_exploration / append_observation` | `explorations` | CRUD |
| `get_encyclopedia_entry / upsert_encyclopedia_entry` | `encyclopedia` | Read / write |
| `create_incident / get_incident / update_incident_status` | `incidents` | CRUD |
| `create_sos_event / get_sos_event / acknowledge_sos / resolve_sos / add_notified_device` | `sos_events` | Full lifecycle |

---

## Pipeline — `species_traits.py`

**File:** `pipeline/species_traits.py`

Synchronous end-to-end species traits pipeline. Used by:
- `GET /species/traits` HTTP endpoint
- `ExplorerAgent._get_gpt_traits()`

### `run_pipeline(scientific_name, iucn_data?)`

```
1. In-memory TTLCache check (24h, maxsize 2048)

2. If iucn_data not supplied:
       _fetch_blob(normalized_name)   → Azure Blob (IUCN JSON)
       ThreadPoolExecutor(max_workers=2):
           _wiki_extract(name)        → Wikipedia OpenSearch + Extract API
           _inaturalist_photo(name)   → iNaturalist taxa API

3. If iucn_data supplied (from live IUCN API):
       Skip blob + Wikipedia (ExplorerAgent fetches photo separately)

4. _call_groq(payload)  →  Groq llama-3.3-70b-versatile
       system: expert wildlife biologist prompt
       response_format: json_object
       extracts: lifespan_years, mass, length, short_description,
                 human_risk_level, human_threat_level, fun_fact_1/2/3

5. Merge IUCN fields + photo + Groq traits → result dict
6. Strip heavy wiki_extract from result
7. Cache result (in-memory TTLCache)
8. Return
```

### Azure Blob Connection Strategy

| Environment | Authentication |
|---|---|
| Local dev | `AZURE_STORAGE_CONNECTION_STRING` (env var) |
| Azure Container Apps | `AZURE_STORAGE_ACCOUNT_URL` + `DefaultAzureCredential` (Managed Identity) |

---

## Shared Enumerations

**File:** `agents/enums.py`

| Enum | Values |
|---|---|
| `QueryType` | `report`, `explore`, `encyclopedia` |
| `AgentLayer` | `layer0`, `layer1`, `layer2`, `layer3` |
| `MessageRole` | `user`, `assistant`, `system`, `tool` |
| `HumanRiskLevel` | `Very High`, `High`, `Caution`, `Low` |
| `SeverityLevel` | `critical`, `high`, `medium`, `low`, `informational` |
| `NotificationChannel` | `web_push`, `email`, `sms` |
| `SOSStatus` | `active`, `acknowledged`, `resolved` |
| `IncidentStatus` | `open`, `investigating`, `resolved` |
| `IncidentType` | `poaching`, `human_wildlife_conflict`, `injured_animal`, `illegal_entry`, `other` |
| `IncidentSeverity` | `critical`, `high`, `medium`, `low` |
| `ReportStatus` | `draft`, `submitted`, `archived` |
| `UserRole` | `ranger`, `officer`, `admin`, `public` |

---

## Data Schemas Reference

### `prahari-state` Schemas (`agents/state/schemas.py`)

#### `MessageRecord`
```python
role: MessageRole
content: str
timestamp: datetime
tool_name: str | None
layer: AgentLayer | None
```

#### `SessionDocument`
```python
id / session_id: str
user_id: str
query_type: QueryType
created_at / updated_at: datetime
messages: list[MessageRecord]
active_layer: AgentLayer
metadata: dict
```

#### `ImageAnalysisCacheDoc`
```python
id: str               # MD5(image_url)
result: dict          # ImageAnalysisResult serialized
ttl: int              # 604800 (7 days)
```

#### `LocationCacheDoc`
```python
id: str               # geohash
geocode: dict
ttl: int              # 2592000 (30 days)
```

---

### `prahari-data` Schemas (`agents/state/data_schemas.py`)

#### `DeviceInfo`
```python
device_id: str
fcm_token: str | None
user_agent: str | None
platform: str
last_seen: datetime
registered_at: datetime
```

#### `UserDocument`
```python
id / user_id: str
name: str
role: UserRole
forest_zone: str | None
district: str | None
state: str | None
devices: list[DeviceInfo]
created_at / updated_at: datetime
```
Method: `.upsert_device(device_info)` — replaces by `device_id`.

#### `ReportDocument`
```python
id / report_id: str
session_id: str
user_id: str
species_name: str
location_context: dict
species_context: dict
risk_level: HumanRiskLevel
pdf_url: str
status: ReportStatus
created_at: datetime
```

#### `SOSEventDocument`
```python
id / sos_id: str
user_id: str
latitude: float
longitude: float
device_id: str
status: SOSStatus
notified_devices: list[str]
created_at: datetime
acknowledged_at: datetime | None
resolved_at: datetime | None
```

---

## Complete End-to-End Data Flow

```
╔═══════════════════════════════════════════════════════════════╗
║  Ranger App                                                   ║
║  POST /agent/chat                                             ║
║  { user_id, query_type="report",                              ║
║    message="spotted tiger near camp",                         ║
║    image_url="https://...",                                   ║
║    latitude=20.5, longitude=78.9 }                            ║
╚═══════════════════════════════════════════════════════════════╝
                          │
                          ▼
╔═══════════════════════════════════════════════════════════════╗
║  LAYER 0 — RECEIVER                                           ║
║  • Extract X-Device-ID, X-FCM-Token, User-Agent from headers  ║
║  • session_id not provided → generate UUID4                   ║
║  • Cosmos GET prahari-state/sessions/<id>                     ║
║    └── not found → create_session(user_id, "report")          ║
║  • Return AgentRequest                                        ║
╚═══════════════════════════════════════════════════════════════╝
                          │
                          ▼
╔═══════════════════════════════════════════════════════════════╗
║  LAYER 0 — ORCHESTRATOR                                       ║
║                                                               ║
║  [1st GPT Call — Tool Selection]                              ║
║  All 3 inputs present → select all 3 tools                    ║
║                                                               ║
║  [asyncio.gather — Parallel Layer 1]                          ║
║  ┌─────────────────────┬──────────────────┬─────────────────┐ ║
║  │ analyse_image       │ analyse_text     │ location_context│ ║
║  │                     │                  │                 │ ║
║  │ 1. cache check      │ GPT text call    │ geohash compute │ ║
║  │ 2. SpeciesNet.predict│ → severity=high │ → cache miss    │ ║
║  │ 3. relevancy: ✅    │   sci=Panthera   │ → Nominatim API │ ║
║  │ 4. confidence=0.72  │   tigris         │ → Pench NP,     │ ║
║  │    < 0.80 threshold │   confidence=85  │   MP, India     │ ║
║  │    → GPT fallback   │                  │ → write cache   │ ║
║  │ 5. GPT vision call  │                  │                 │ ║
║  │    → Panthera tigris│                  │                 │ ║
║  │ 6. write cache      │                  │                 │ ║
║  └─────────────────────┴──────────────────┴─────────────────┘ ║
║                                                               ║
║  [Relevancy Gate]                                             ║
║  image.is_relevant=True → continue                            ║
║                                                               ║
║  [Persist user message → Cosmos session]                      ║
║                                                               ║
║  [2nd GPT Call — Layer 2 Routing]                             ║
║  query_type="report" → forced tool_choice=REPORT              ║
╚═══════════════════════════════════════════════════════════════╝
                          │
                          ▼
╔═══════════════════════════════════════════════════════════════╗
║  LAYER 2 — SpeciesInfoAgent (always first)                    ║
║  • scientific_name = "Panthera tigris" (from image)           ║
║  • Check species_context_cache["iucn:panthera_tigris"] → miss ║
║  • Azure Blob fetch: iucn-species/Pantheratigris*.json        ║
║  • Returns raw IUCN dict (EN, Decreasing, poaching...)        ║
║  • Write to Cosmos cache (24h TTL)                            ║
╚═══════════════════════════════════════════════════════════════╝
                          │
                          ▼
╔═══════════════════════════════════════════════════════════════╗
║  LAYER 2 — ReporterAgent                                      ║
║                                                               ║
║  [asyncio.gather]                                             ║
║  ┌──────────────────────────┬──────────────────────────────┐  ║
║  │ get_species_safety_data  │ get_inaturalist_photo        │  ║
║  │ GPT safety advisor call  │ iNaturalist taxa API         │  ║
║  │ → risk_level=Very High   │ → photo_url, attribution     │  ║
║  │   precautions[4 items]   │                              │  ║
║  │   lifespan=10-15 years   │                              │  ║
║  └──────────────────────────┴──────────────────────────────┘  ║
║                                                               ║
║  threat_level = "Critical"  (IUCN code EN → High)            ║
║  severity = "high"          (from TextAnalysisAgent)          ║
║                                                               ║
║  _build_pdf() → ReportLab A4 PDF bytes                        ║
║  _upload_pdf_blob() → Azure Blob report-pdfs/<id>.pdf         ║
║  upsert_report() → Cosmos prahari-data/reports                ║
╚═══════════════════════════════════════════════════════════════╝
                          │
                          ▼
╔═══════════════════════════════════════════════════════════════╗
║  RESPONSE                                                     ║
║  {                                                            ║
║    "session_id": "...",                                       ║
║    "user_id": "ranger_001",                                   ║
║    "query_type": "report",                                    ║
║    "data": {                                                  ║
║      "report_id": "...",                                      ║
║      "pdf_url": "https://blob.../report-pdfs/....pdf",        ║
║      "scientific_name": "Panthera tigris",                    ║
║      "common_name": "Bengal Tiger",                           ║
║      "risk_level": "Very High",                               ║
║      "threat_level": "High",                                  ║
║      "safety_precautions": [...],                             ║
║      "iucn_category": "EN",                                   ║
║      "photo_url": "...",                                      ║
║      ...                                                      ║
║    }                                                          ║
║  }                                                            ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## Technologies Used

### AI / ML

| Technology | Version / Model | Used By |
|---|---|---|
| **SpeciesNet** (PyTorch) | v4.0.2b | `ImageAnalysisAgent` — wildlife detection + classification |
| **Azure OpenAI** | GPT-4o-mini (`gpt-5-mini` deployment) | Orchestrator, GPT Vision, TextAnalysisAgent, ReporterAgent |
| **Groq** | llama-3.3-70b-versatile | `run_pipeline()` — species trait extraction |
| **LangChain** | — | Orchestrator tool definitions, message history, StructuredTool |
| **LangSmith** | — | Orchestrator tracing + span management |

### Cloud Infrastructure

| Technology | Used By |
|---|---|
| **Azure Cosmos DB** | SessionStore, DataStore, all 5 cache types |
| **Azure Blob Storage** | IUCN species JSON (read) + PDF report uploads (write) |
| **Azure DefaultAzureCredential** | Managed Identity auth on Azure Container Apps |

### External APIs

| API | Used By |
|---|---|
| **IUCN Red List API v4** | `SpeciesInfoAgent` — live fallback |
| **Nominatim (OpenStreetMap)** | `LocationAgent` — GPS reverse geocoding |
| **iNaturalist API** | `ReporterAgent`, `ExplorerAgent`, `EncyclopediaAgent` — species photos |
| **Wikipedia API** | `pipeline/species_traits.py` — text context for Groq |

### Backend Framework & Libraries

| Library | Purpose |
|---|---|
| **FastAPI** | HTTP server, lifespan, routing |
| **Pydantic / Pydantic Settings** | All schemas, settings classes |
| **aiohttp** | Async IUCN API calls in `SpeciesInfoAgent` |
| **httpx** | General async HTTP |
| **requests** | iNaturalist shared session with connection pooling |
| **ReportLab** | A4 PDF generation in `ReporterAgent` |
| **pygeodesy** | Geohash computation for location cache keys |
| **cachetools TTLCache** | In-memory 24h species trait cache |
| **asyncio** | `asyncio.gather` for parallel Layer 1 execution |
| **ThreadPoolExecutor** | Concurrent Wikipedia + iNaturalist in `run_pipeline()` |

---

## Agent × Technology Matrix

| Agent | Layer | Primary Technologies |
|---|---|---|
| **Receiver** | 0 | FastAPI, Pydantic, Cosmos DB (sessions) |
| **Orchestrator** | 0 | Azure OpenAI, LangChain, LangSmith, asyncio |
| **Router** | 0 | FastAPI |
| **ImageAnalysisAgent** | 1 | SpeciesNet (PyTorch), Azure OpenAI (GPT vision), Cosmos DB (cache) |
| **TextAnalysisAgent** | 1 | Azure OpenAI (GPT-4o-mini) |
| **LocationAgent** | 1 | Nominatim OSM API, pygeodesy, Cosmos DB (cache) |
| **SpeciesInfoAgent** | 2 | Azure Blob Storage, IUCN Red List API v4, aiohttp, Cosmos DB (cache) |
| **ReporterAgent** | 2 | Azure OpenAI, iNaturalist API, ReportLab, Azure Blob Storage, Cosmos DB |
| **ExplorerAgent** | 2 | Groq llama-3.3-70b, iNaturalist API, Cosmos DB (cache) |
| **EncyclopediaAgent** | 2 | iNaturalist API |
| **SpeciesTraitsPipeline** | — | Azure Blob Storage, Groq, Wikipedia API, iNaturalist API, TTLCache, ThreadPoolExecutor |

---

*Document generated: March 6, 2026 — Prahari Backend*

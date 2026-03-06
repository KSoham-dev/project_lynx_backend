from dotenv import load_dotenv
load_dotenv()  # must be first — populates os.environ before any module reads it

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from contextlib import asynccontextmanager
from typing import Any, Optional
import os
import uvicorn

from pipeline.species_traits import run_pipeline
from agents.layer0.router import router as agent_router
from agents.layer0.stats_router import router as stats_router
from agents.state.cosmos_client import close_cosmos_clients
from agents.layer1.image_agent import set_speciesnet_model

# ── Model globals ──────────────────────────────────────────────────────────────
model = None
MODEL_PATH = os.getenv("MODEL_PATH", "./model")

import logging
logging.basicConfig(level=logging.INFO)

# Silence noisy SDK loggers — only show WARNING+ from these
for _noisy in (
    "azure.cosmos",
    "azure.core",
    "azure.cosmos._cosmos_http_logging_policy",
    "httpx",
    "httpcore",
    "urllib3",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the SpeciesNet model once on startup and clean up on shutdown."""
    global model
    from speciesnet import SpeciesNet  # lazy import keeps startup error clear

    print("Loading SpeciesNet model …")
    try:
        model = SpeciesNet(MODEL_PATH)
        print("SpeciesNet model loaded successfully.")
        set_speciesnet_model(model)         # inject into Layer 1 image agent
        print("Layer 1 ImageAnalysisAgent ready.")
    except Exception as exc:
        print(f"Failed to load SpeciesNet model: {exc}")
        raise RuntimeError(f"Model load failed: {exc}") from exc
    yield
    # Cleanup: close Cosmos DB connection pools
    await close_cosmos_clients()
    model = None


app = FastAPI(
    title="Prahari – SpeciesNet API",
    description="Identify wildlife species from an image URL using SpeciesNet.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Agent router (Layer 0: Orchestrator + Context/State) ───────────────────────
app.include_router(agent_router)
app.include_router(stats_router)


# ── Request / Response schemas ─────────────────────────────────────────────────


class PredictRequest(BaseModel):
    image_url: str  # direct URL to the image (jpg / png / …)
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "image_url": "https://inaturalist-open-data.s3.amazonaws.com/photos/605060662/large.jpg",
                "latitude": 34.08,
                "longitude": 74.8056,
            }
        }
    }


class SpeciesTraitsResponse(BaseModel):
    """Response returned by GET /species/traits."""
    assessment_id: Optional[Any] = None
    year_published: Optional[Any] = None
    scientific_name: Optional[str] = None
    common_names: list[str] = []
    category: Optional[str] = None
    url: Optional[str] = None
    sis_taxon_id: Optional[Any] = None
    photo_url: str = "Not available"
    photo_credit: str = "Not available"
    # LLM-extracted trait fields
    lifespan_years: Optional[str] = None
    mass: Optional[str] = None
    length: Optional[str] = None
    short_description: Optional[str] = None
    human_risk_level: Optional[str] = None
    human_threat_level: Optional[str] = None
    fun_fact_1: Optional[str] = None
    fun_fact_2: Optional[str] = None
    fun_fact_3: Optional[str] = None

    model_config = {"extra": "allow"}  # surface any extra LLM fields


class ClassificationResult(BaseModel):
    label: str
    score: float


class PredictionResponse(BaseModel):
    image_url: str
    top_prediction: Optional[str]
    classifications: Optional[list[ClassificationResult]]
    raw: dict  # full model output for power users


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/", tags=["Health"])
def root():
    """Health-check / welcome endpoint."""
    return {"status": "ok", "message": "Prahari SpeciesNet API is running."}


@app.get("/health", tags=["Health"])
def health():
    """Returns whether the model has been loaded successfully."""
    return {"model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Run species classification on a single image.

    - **image_url**: Publicly accessible URL of the image.
    - **latitude** / **longitude**: Optional GPS coordinates to refine predictions.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    instance: dict = {"filepath": request.image_url}
    if request.latitude is not None:
        instance["latitude"] = request.latitude
    if request.longitude is not None:
        instance["longitude"] = request.longitude

    try:
        result = model.predict(instances_dict={"instances": [instance]})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    if not result or "predictions" not in result or not result["predictions"]:
        raise HTTPException(status_code=500, detail="Model returned no predictions.")

    prediction = result["predictions"][0]
    classifications_raw = prediction.get("classifications", {})

    classes = classifications_raw.get("classes", [])
    scores = classifications_raw.get("scores", [])

    top_prediction = classes[0] if classes else None

    classification_list = [
        ClassificationResult(label=cls, score=round(float(sc), 6))
        for cls, sc in zip(classes, scores)
    ]

    return PredictionResponse(
        image_url=request.image_url,
        top_prediction=top_prediction,
        classifications=classification_list,
        raw=prediction,
    )


@app.post("/predict/batch", response_model=list[PredictionResponse], tags=["Prediction"])
def predict_batch(requests_list: list[PredictRequest]):
    """
    Run species classification on multiple images in a single call.

    Each item in the list follows the same schema as `/predict`.
    Maximum recommended batch size: 16.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    if not requests_list:
        raise HTTPException(status_code=400, detail="Request list is empty.")

    instances = []
    for req in requests_list:
        instance: dict = {"filepath": req.image_url}
        if req.latitude is not None:
            instance["latitude"] = req.latitude
        if req.longitude is not None:
            instance["longitude"] = req.longitude
        instances.append(instance)

    try:
        result = model.predict(instances_dict={"instances": instances})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    if not result or "predictions" not in result:
        raise HTTPException(status_code=500, detail="Model returned no predictions.")

    responses = []
    for req, prediction in zip(requests_list, result["predictions"]):
        classifications_raw = prediction.get("classifications", {})
        classes = classifications_raw.get("classes", [])
        scores = classifications_raw.get("scores", [])
        top_prediction = classes[0] if classes else None

        classification_list = [
            ClassificationResult(label=cls, score=round(float(sc), 6))
            for cls, sc in zip(classes, scores)
        ]

        responses.append(
            PredictionResponse(
                image_url=req.image_url,
                top_prediction=top_prediction,
                classifications=classification_list,
                raw=prediction,
            )
        )

    return responses


# ── Species Traits (IUCN + Wikipedia + iNaturalist + Groq) ────────────────────


@app.get(
    "/species/traits",
    response_model=SpeciesTraitsResponse,
    tags=["Species Traits"],
    summary="Get rich trait data for a species by scientific name",
)
def species_traits(
    scientific_name: str = Query(
        ...,
        description="Scientific name of the species (e.g. 'Caridina typus').",
        examples=["Caridina typus", "Panthera leo"],
        min_length=3,
    )
) -> SpeciesTraitsResponse:
    """
    Fetch IUCN data from Azure Blob Storage, enrich it with Wikipedia text and
    iNaturalist photo, then extract structured biological traits via LLM.

    - **scientific_name**: the species scientific name; spaces are fine.

    Results are cached in-process for **24 hours** — repeated calls are
    near-instant.
    """
    try:
        result = run_pipeline(scientific_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Pipeline error: {exc}"
        ) from exc

    return SpeciesTraitsResponse(**result)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

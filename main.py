from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from contextlib import asynccontextmanager
from typing import Optional
import os
import uvicorn

# ── Model globals ──────────────────────────────────────────────────────────────
model = None
MODEL_PATH = os.getenv("MODEL_PATH", "./model")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the SpeciesNet model once on startup and clean up on shutdown."""
    global model
    from speciesnet import SpeciesNet  # lazy import keeps startup error clear

    print("Loading SpeciesNet model …")
    try:
        model = SpeciesNet(MODEL_PATH)
        print("SpeciesNet model loaded successfully.")
    except Exception as exc:
        print(f"Failed to load SpeciesNet model: {exc}")
        raise RuntimeError(f"Model load failed: {exc}") from exc
    yield
    # Cleanup (if any) goes here
    model = None


app = FastAPI(
    title="Prahari – SpeciesNet API",
    description="Identify wildlife species from an image URL using SpeciesNet.",
    version="1.0.0",
    lifespan=lifespan,
)


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


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

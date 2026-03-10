# ── Stage 1: build dependencies ────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Pull uv binary from the official distroless image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /build

# System deps needed to compile / install Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN uv pip install --no-cache --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ─────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Runtime shared libraries (opencv-headless, shapely, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libexpat1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# ── Copy the SpeciesNet model directly ────────────────────────────────────────
COPY model/ /app/model/

# ── Copy application code ──────────────────────────────────────────────────────
COPY main.py .
COPY agents/ /app/agents/
COPY pipeline/ /app/pipeline/

# ── Security: run as non-root ──────────────────────────────────────────────────
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser \
    && chown -R appuser:appgroup /app
USER appuser

# ── Runtime configuration ──────────────────────────────────────────────────────
# Azure Container Apps injects PORT; default to 8000 if not set.
ENV MODEL_PATH=/app/model \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Use shell form so $PORT is evaluated at container start
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1

# ITOL — multi-stage CPU-only Docker build
#
# Stage 1 (builder): installs build tools and compiles wheels
# Stage 2 (runtime): python:3.11-slim with only runtime artefacts
#
# Usage:
#   docker build -t itol:latest -f deploy/Dockerfile .
#   docker run -p 8080:8080 -v /data/itol:/data itol:latest
#
# Environment variables:
#   ITOL_DATA_DIR   — persistent storage path (default: /data)
#   ITOL_MODE       — optimize | cache_only | observe_only | bypass (default: optimize)
#   ITOL_PORT       — TCP port for the proxy server (default: 8080)

# ────────────────────────────────────────────────
# Stage 1: builder
# ────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (wheel compilation only — not in final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only the files needed to install the package
COPY pyproject.toml ./
COPY itol/ ./itol/
COPY jobs/ ./jobs/
COPY calibration/ ./calibration/
COPY data/ ./data/

# Install the package and its dependencies into /install
RUN pip install --upgrade pip --no-cache-dir \
 && pip install --prefix=/install --no-cache-dir \
        ".[embed]" \
 && pip install --prefix=/install --no-cache-dir \
        gunicorn>=21.0 \
        pyyaml>=6.0

# ────────────────────────────────────────────────
# Stage 2: runtime (slim image, no build tools)
# ────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN addgroup --system itol && adduser --system --ingroup itol itol

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY --from=builder /build/itol /app/itol
COPY --from=builder /build/jobs /app/jobs
COPY --from=builder /build/calibration /app/calibration
COPY --from=builder /build/data /app/data

WORKDIR /app

# Persistent data volume (SQLite db, model cache, calibration outputs)
RUN mkdir -p /data && chown itol:itol /data
VOLUME ["/data"]

# Runtime configuration via environment
ENV ITOL_DATA_DIR=/data \
    ITOL_MODE=optimize \
    ITOL_PORT=8080 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

USER itol

# Health check — the /healthz endpoint returns 200 when the proxy is ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${ITOL_PORT}/healthz')" || exit 1

# Default command — run the ITOL proxy server
CMD ["python", "-m", "itol", "serve", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--data-dir", "/data"]

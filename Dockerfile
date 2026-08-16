# ── Stage 1: builder — compile C-extension wheels ─────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# lxml needs libxml2/libxslt headers to compile; everything else uses pre-built wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY req.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r req.txt


# ── Stage 2: runtime — lean final image ───────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Runtime libs for lxml (no dev headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./

# The graph .pkl cache files are committed to git and copied in here,
# so the server loads the Kolkata OSM graph in ~2s on boot — no re-download.
# Files: app/cache/graph.pkl (drive), graph_walk.pkl, graph_bike.pkl

# Writable dirs: SQLite DB lives on /data (Render persistent disk),
# hgnn/weights can be populated post-deploy if needed
RUN mkdir -p cache hgnn/weights

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Render injects $PORT automatically; fall back to 8000 for local dev
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1

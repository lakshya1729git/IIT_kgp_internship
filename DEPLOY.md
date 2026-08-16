# Deploying traffic-ai to Render (Free)

## What's been set up

| File | Purpose |
|------|---------|
| `Dockerfile` | Builds the app into a container (Python 3.11-slim, 2-stage) |
| `render.yaml` | Render Blueprint — service config + persistent disk |
| `.dockerignore` | Keeps the Docker image lean |
| `app/cache/*.pkl` | OSM graph pre-baked into the image (fast ~2s boot, no re-download) |
| `app/api.py` | Added `/health` endpoint + modernised lifespan startup |

---

## Step 1 — Push to GitHub

```bash
cd traffic-ai

# Stage everything new
git add Dockerfile render.yaml .dockerignore .gitignore app/api.py
git add app/cache/graph.pkl app/cache/graph_walk.pkl app/cache/graph_bike.pkl

git commit -m "chore: add Render deployment config and baked graph cache"
git push
```

> The 3 `.pkl` files are ~75 MB total — well under GitHub's 100 MB per-file limit.
> If your repo is private that's fine; Render connects to both public and private repos.

---

## Step 2 — Create the Render service

1. Go to [render.com](https://render.com) → **New** → **Blueprint**
2. Connect your GitHub account and select the `traffic-ai` repo
3. Render reads `render.yaml` automatically and proposes the service config
4. Click **Apply** — it will create:
   - A **Web Service** (free tier, Docker runtime)
   - A **1 GB persistent disk** mounted at `/data` (for SQLite DB)

Alternatively, create it manually:
1. **New** → **Web Service** → connect repo
2. **Runtime**: Docker
3. **Plan**: Free
4. Under **Advanced** → **Add Disk**: mount path `/data`, size 1 GB

---

## Step 3 — Set environment variables

In the Render dashboard → your service → **Environment**, add:

| Key | Value | Required |
|-----|-------|----------|
| `OPENROUTER_API_KEY` | your key from [openrouter.ai/keys](https://openrouter.ai/keys) | ✅ (or Gemini) |
| `GEMINI_API_KEY` | your key from [aistudio.google.com](https://aistudio.google.com/app/apikey) | ✅ (or OpenRouter) |
| `TOMTOM_API_KEY` | from [developer.tomtom.com](https://developer.tomtom.com) | Recommended |
| `OPENWEATHER_API_KEY` | from [openweathermap.org/api](https://openweathermap.org/api) | Optional |
| `NEWS_API_KEY` | from [newsapi.org/register](https://newsapi.org/register) | Optional |

The non-secret vars (`DATABASE_URL`, `GRAPH_CACHE_PATH`, etc.) are already set in `render.yaml`.

---

## Step 4 — Deploy

Click **Manual Deploy** → **Deploy latest commit** (or it triggers automatically on push).

Build takes ~5–8 minutes on the first run (pip installs osmnx, torch-free stack).  
Subsequent deploys are faster due to Docker layer caching.

Once live, Render will show a URL like `https://traffic-ai.onrender.com`.

---

## Free tier notes

| Limit | Impact |
|-------|--------|
| 512 MB RAM | `MAX_ROUTES=5` is set in render.yaml (lower = less RAM) |
| Spins down after 15 min inactivity | First request after sleep takes ~30s to wake |
| 750 hrs/month compute | Enough for one always-on service |
| 1 GB persistent disk | SQLite DB persists across restarts and redeploys |

The graph `.pkl` files are baked into the Docker image, so **the server loads the Kolkata road network in ~2 seconds** even after a cold wake — no Overpass re-download.

---

## Local dev (unchanged)

```bash
cd app
uvicorn api:app --reload --port 8000
```

---

## Verifying the deployment

- Health check: `GET https://your-app.onrender.com/health` → `{"status":"ok"}`
- Map UI: `https://your-app.onrender.com/`
- API docs: `https://your-app.onrender.com/docs`

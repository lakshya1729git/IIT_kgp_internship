# traffic-ai 🚦

**A Probabilistic and Explainable Tool for Context-Aware Multimodal Trip Planning Using Generative AI**

Research prototype — ANRF ARG grant proposal, IIT Kharagpur.  
Pilot city: **Kolkata, India** | Collaboration with Kolkata Traffic Police.

---

## What it does

Current trip planners are reactive — they only know about a disruption after sensors detect it.  
This system is **anticipatory**: it reads news, public advisories, and live traffic feeds *before* you travel, extracts structured disruption events using an LLM, scores your route using a probabilistic graph model, and renders everything on a live Leaflet.js map.

```
You pick:  Howrah Station  →  Salt Lake Sector V

System:
  Step 1 (~2s)     — Computes 2–3 alternative routes on cached Kolkata road/metro/bus graph
  Step 2 (background, ~30–60s)
                   — Fetches TomTom live incidents + Google News RSS + NewsAPI
                     + weather + Kolkata Police / KMRC / WB Disaster scrapers
                   — Rule-based severity correction (severity_rules.py)
                   — LLM extracts event_type, location, severity σ, confidence κ
                   — HGNN refines κ city-wide; cascade detection adds adjacent roads
                   — Bayesian fusion computes per-road posterior probabilities
                   — Scores each route: risk = Σ(σ × κ × duration_mult × recency_mult)
                   — Updates map with colour-coded routes + disruption markers
```

**The map draws immediately. Risk analysis fills in the background.**

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/sawarn-nik/traffic-ai.git
cd traffic-ai
```

### 2. Python environment

> Python 3.11 is the tested and recommended version.

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r req.txt
```

### 3. API keys

```bash
cp .env.example .env
# Open .env and fill in your keys
```

| Key | Where to get | Required? |
|-----|-------------|-----------|
| `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) — free, no card | ✅ Yes (or Gemini) |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com/app/apikey) — free | ✅ Yes (or OpenRouter) |
| `TOMTOM_API_KEY` | [developer.tomtom.com](https://developer.tomtom.com) — free, no card | ⭐ Recommended |
| `NEWS_API_KEY` | [newsapi.org/register](https://newsapi.org/register) — 100 req/day free | Optional |
| `OPENWEATHER_API_KEY` | [openweathermap.org/api](https://openweathermap.org/api) — free | Optional |

You need **at least one** of OpenRouter or Gemini for LLM extraction.

### 4. Start the server

```bash
cd app
uvicorn api:app --port 8000 --reload
```

### 5. Open the map

```
http://localhost:8000
```

Select source and destination → pick a transport mode → click **Get Routes**.

> **First run:** the Kolkata OSM graph downloads once (~30–60s) and caches to `cache/graph.pkl`. All subsequent runs load from disk in ~2s.

---

## Transport modes

| Mode | Description |
|------|-------------|
| `drive` | Car routing on Kolkata OSM road network |
| `walk` | Pedestrian routing |
| `bike` | Cycling routing |
| `metro` | Kolkata Metro only — all 5 lines |
| `metro+walk` | Metro with walking access/egress legs |
| `metro+bike` | Metro with cycling legs |
| `metro+drive` | Metro with driving legs |
| `bus` | Kolkata bus network (BFS graph) |

---

## Project structure

```
traffic-ai/
├── app/
│   ├── api.py                      # FastAPI server — 11 endpoints
│   ├── main.py                     # CLI pipeline (debugging only)
│   ├── config.py                   # All settings, reads from .env
│   │
│   ├── auto_label.py               # Rule-based ground-truth severity labeling
│   ├── backfill_coords.py          # Geocode NULL lat/lon rows in DB (Nominatim)
│   ├── evaluate_hgnn.py            # HGNN eval: P/R/F1, confusion matrix, ablation
│   ├── generate_training_data.py   # Synthetic balanced training data generator
│   ├── label_events.py             # Interactive CLI labeling tool
│   │
│   ├── static/
│   │   ├── index.html              # Leaflet.js map (entry point)
│   │   ├── scripts.js              # Map interaction and API calls
│   │   └── styles.css              # Frontend styles
│   │
│   ├── ingestion/
│   │   ├── rss_fetcher.py          # Google News RSS (when:2d Kolkata feeds)
│   │   ├── news_fetcher.py         # NewsAPI with date parsing
│   │   ├── tomtom_fetcher.py       # TomTom Traffic Incidents API v5 (live)
│   │   ├── weather_fetcher.py      # OpenWeatherMap
│   │   └── web_scraper.py          # Kolkata Police, KMRC, WB Disaster, Railways, KMC
│   │
│   ├── llm/
│   │   ├── extractor.py            # LangChain LCEL chain + JSON repair + retry
│   │   ├── prompts.py              # Kolkata-tuned extraction + location-retry prompts
│   │   ├── schema.py               # TrafficEventSchema (Pydantic)
│   │   ├── filter.py               # Pre/post-LLM filtering and deduplication
│   │   └── location_resolver.py    # Location enrichment and recovery
│   │
│   ├── hgnn/
│   │   ├── model.py                # Temporal HAN — 3 output heads, attention weights
│   │   ├── graph_builder.py        # Heterogeneous graph from DB events
│   │   ├── inference.py            # Lazy-load inference wrapper
│   │   ├── integration.py          # City-wide confidence enhance + cascade detection
│   │   ├── trainer.py              # V3 training loop (verified-label boosting)
│   │   └── weights/                # Trained model weights (model.pt)
│   │
│   ├── routing/
│   │   ├── route_engine.py         # OSMnx k-shortest paths, Kolkata graph cache
│   │   ├── multimodal.py           # Multimodal routing orchestration
│   │   ├── metro_timetable.py      # 5-line metro data + next-train (live IST)
│   │   ├── map_matcher.py          # Snap event coords → OSMnx road edges
│   │   ├── geocoder.py             # Location name → coordinates
│   │   └── cost_function.py        # Layer 3 generalised edge cost (Eq. 3)
│   │
│   ├── scoring/
│   │   ├── congestion_score.py     # σ × κ base computation + route impact
│   │   ├── confidence.py           # Multi-factor confidence (source + age + location)
│   │   ├── impact_duration.py      # Event duration estimation
│   │   └── severity_rules.py       # Rule-based severity correction layer
│   │
│   ├── transit/
│   │   ├── bus_graph.py            # Kolkata bus network graph
│   │   ├── bus_engine.py           # BFS bus route finder
│   │   ├── bus_overlay.py          # GeoJSON overlay for map
│   │   └── data/                   # bus_stops.json, bus_routes.json
│   │
│   ├── weather/
│   │   ├── route_weather.py        # Fetch weather at route sample points
│   │   └── weather_risk.py         # Weather Severity Index (WSI)
│   │
│   ├── fusion/
│   │   └── bayesian_fusion.py      # Layer 2 — Bayesian update per road edge
│   │
│   ├── database/
│   │   └── models.py               # SQLAlchemy schema + auto-migration
│   │
│   ├── results/
│   │   └── hgnn_eval_results.csv   # Latest HGNN evaluation output
│   │
│   └── utils/
│       └── helpers.py              # Text cleaning, dedup, timestamps
│
├── .env.example                    # Template — copy to .env
├── .gitignore
├── CONTRIBUTING.md
├── req.txt                         # Python dependencies (pinned)
└── README.md
```

---

## How routes are scored

```
severity_score σ:        low=2  medium=5  high=10
confidence κ:            LLM score × source reliability × age decay × location quality
weighted_score per event: σ × κ × duration_multiplier × recency_multiplier

route_risk = Σ weighted_scores for active matched events on route

Risk levels:  CRITICAL ≥ 25 (purple) | HIGH ≥ 12 (red) | MODERATE ≥ 5 (orange)
              LOW > 0 (yellow-green)  | CLEAR = 0 (green)

best route = lowest  travel_time × (1 + risk_score / 10)
```

HGNN multiplies each event's contribution by a per-road disruption probability, so two routes matching the same events score differently based on their graph topology.

---

## HGNN training pipeline

```bash
cd app

# 1. Generate synthetic balanced training data
python generate_training_data.py --count 400

# 2. Auto-label TomTom events with rule-based ground truth
python auto_label.py

# 3. (Optional) Interactive human labeling
python label_events.py --limit 100

# 4. Backfill NULL coordinates using Nominatim
python backfill_coords.py

# 5. Train the model
python -m hgnn.trainer --epochs 200 --lr 0.001

# 6. Evaluate against ground-truth labels
python evaluate_hgnn.py --export results/
```

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Leaflet.js map frontend |
| `GET` | `/api/locations` | Localities, metro stations, bus stops |
| `POST` | `/api/route` | Step 1 — fast route computation, no LLM |
| `POST` | `/api/disruptions` | Step 2 — LLM + HGNN + Bayesian scoring |
| `POST` | `/api/explain-route` | HGNN attention-based route explanation |
| `GET` | `/api/metro-overlay` | GeoJSON of all 5 metro lines |
| `GET` | `/api/metro-lines` | Metro timetable summary |
| `GET` | `/api/next-metro/{station}` | Next n trains at station (live IST) |
| `GET` | `/api/bus-overlay` | GeoJSON of all bus stops |
| `GET` | `/api/bus-network` | Full bus route and stop data |
| `GET` | `/api/hgnn-status` | HGNN model readiness |

---

## Configuration

All settings are read from `.env` (copy from `.env.example`).

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `openrouter` | `openrouter` \| `gemini` \| `ollama` |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `OPENROUTER_MODEL` | `meta-llama/llama-3.3-70b-instruct:free` | Model slug |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model |
| `OLLAMA_MODEL` | `phi3` | Local Ollama model (no key needed) |
| `NEWS_API_KEY` | — | NewsAPI key |
| `TOMTOM_API_KEY` | — | TomTom Traffic API key |
| `OPENWEATHER_API_KEY` | — | OpenWeatherMap key |
| `ENABLE_TOMTOM` | `true` | Toggle TomTom ingestion |
| `ENABLE_WEATHER` | `true` | Toggle weather ingestion |
| `ENABLE_NEWSAPI` | `true` | Toggle NewsAPI ingestion |
| `ENABLE_RSS` | `true` | Toggle RSS ingestion |
| `ENABLE_SCRAPER` | `true` | Toggle web scraper |
| `ENABLE_NOMINATIM` | `true` | Toggle Nominatim geocoding |
| `MAX_ROUTES` | `10` | Max candidate routes |
| `MIN_ROUTE_DIVERGENCE` | `0.20` | Min divergence between route alternatives |
| `DATABASE_URL` | `sqlite:///traffic_events.db` | SQLAlchemy DB URL |

---

## System layers

| Layer | Description | Status |
|-------|-------------|--------|
| **1** | LLM disruption extraction + HGNN confidence refinement | ✅ Complete |
| **2** | Bayesian probabilistic fusion per road edge | ✅ Complete |
| **3** | CVaR risk-aware routing + natural language explainability | 🔲 In progress |

---

## Research context

> *A Probabilistic and Explainable Tool for Context-Aware Multimodal Trip Planning Using Generative Artificial Intelligence (GenAI)*  
> ANRF ARG Pre-Proposal, IIT Kharagpur, 2026

---

## Contributors

| GitHub | Name | Contributions |
|--------|------|---------------|
| [@sawarn-nik](https://github.com/sawarn-nik) | **Nikhil Mishra** | Layer 1 LLM pipeline, HGNN architecture & training, Bayesian fusion (Layer 2), multimodal routing, FastAPI backend, Leaflet.js frontend |
| [@lakshya1729git](https://github.com/lakshya1729git) | **Lakshya Sharma** | Bus transit module, bus graph & routing engine, bus map overlay, multimodal bus integration, transit data curation |
| [@Ruchi-Kumarii](https://github.com/Ruchi-Kumarii) | **Ruchi Mishra** | HGNN training pipeline (V3), auto-labeling, ground-truth evaluation, severity rules, map matcher, cost function (Layer 3), coordinate backfilling, synthetic data generation |

---

## License

MIT

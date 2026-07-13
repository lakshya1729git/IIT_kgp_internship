# traffic-ai 🚦

**A Probabilistic and Explainable Tool for Context-Aware Multimodal Trip Planning Using Generative AI**

Research prototype — ANRF ARG grant proposal, IIT Kharagpur.  
Pilot city: **Kolkata, India** | Collaboration with Kolkata Traffic Police.

---

## What it does

Current trip planners are reactive — they only know about a disruption after sensors detect it.  
This system is **anticipatory**: it reads news, public advisories, and live traffic feeds *before* you travel, extracts structured disruption events using an LLM, and scores your route in real time.

```
You pick:  Howrah Station  →  Salt Lake Sector V

System:
  Step 1 (instant ~2s)  — Computes 2–3 alternative routes on Kolkata road graph
  Step 2 (background)   — Fetches Google News RSS + NewsAPI + TomTom Live incidents
                        — LLM extracts event_type, location, severity σ, confidence κ
                        — Scores each route: risk = Σ(σ × κ) for recent events
                        — Updates map with colour-coded routes + disruption markers
```

**The map draws immediately. Disruption analysis fills in the background.**

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/sawarn-nik/traffic-ai.git
cd traffic-ai
```

### 2. Python environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r req.txt

"Here python 3.11 is only compatible and working properly"
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

Select source and destination → click **Get Routes**.  
Routes appear on the map within ~2 seconds. Disruption analysis populates in the background (~30–60 seconds).

---

## Project structure

```
traffic-ai/
├── app/
│   ├── api.py                   # FastAPI server — run this with uvicorn
│   ├── main.py                  # CLI pipeline (optional, for debugging)
│   ├── config.py                # All settings, reads from .env
│   ├── static/
│   │   └── index.html           # Leaflet.js map frontend (single file)
│   ├── ingestion/
│   │   ├── rss_fetcher.py       # Google News RSS (when:2d Kolkata feeds)
│   │   ├── news_fetcher.py      # NewsAPI
│   │   ├── tomtom_fetcher.py    # TomTom Traffic Incidents API (live)
│   │   └── weather_fetcher.py   # OpenWeatherMap (optional)
│   ├── llm/
│   │   ├── extractor.py         # LangChain LCEL chain + JSON repair + retry
│   │   ├── prompts.py           # Kolkata-tuned system + location-retry prompts
│   │   ├── schema.py            # TrafficEventSchema (Pydantic)
│   │   ├── filter.py            # Pre/post LLM filtering and deduplication
│   │   └── location_resolver.py # Location resolution and enrichment
│   ├── routing/
│   │   ├── route_engine.py      # OSMnx routing + multi-route + GeoJSON export
│   │   └── cost_function.py     # Layer 3 placeholder (Eq. 3)
│   ├── scoring/
│   │   ├── congestion_score.py  # σ × κ weighted score
│   │   ├── confidence.py        # Enhanced multi-factor confidence scoring
│   │   └── impact_duration.py   # Event duration estimation
│   ├── fusion/
│   │   └── bayesian_fusion.py   # Layer 2 placeholder (Eq. 1)
│   ├── database/
│   │   └── models.py            # SQLAlchemy — traffic_events.db
│   └── utils/
│       └── helpers.py           # Text cleaning, dedup, timestamps
├── docs/
│   └── project-handoff.md       # Full technical handoff for teammates
├── data/                        # Datasets (not committed)
├── notebooks/                   # Jupyter analysis notebooks
├── .env.example                 # Template — copy to .env
├── .gitignore
├── req.txt                      # Python dependencies
└── README.md
```

---

## How routes are scored

```
severity_score σ:   low=2  medium=5  high=10
confidence κ:       0.0–1.0  (LLM certainty about the extraction)
weighted_score:     σ × κ

route_risk = Σ(weighted_score) for all recent events on route

risk levels:
  CRITICAL  ≥ 25   (deep purple)
  HIGH      ≥ 12   (red)
  MODERATE  ≥ 5    (orange)
  LOW       > 0    (yellow-green)
  CLEAR     = 0    (green)

best route = lowest  travel_time × (1 + risk_score / 10)
```

---

## HGNN — Heterogeneous Graph Neural Network

The HGNN layer refines LLM-assigned severities using graph-based context: multi-source corroboration, spatial road clustering, source reliability, and temporal decay.

### Architecture
- **Node types**: events, roads, sources, locations
- **Relations**: event→road, event→source, event→location, road↔road (spatial)
- **Outputs**: road disruption probability, adjusted event confidence, severity logits (3-class)

### Training workflow

```bash
cd app

# 1. Auto-label real TomTom events with rule-based ground truth
python auto_label.py               # labels all unlabeled real events
python auto_label.py --stats       # check coverage

# 2. (Optional) Generate synthetic data for class balance
python generate_training_data.py --count 400 --clear-synthetic

# 3. Train the model
#    80% of verified labels → train (with 5× loss boost)
#    20% of verified labels → val  (ground-truth validation signal)
#    All proxy-labeled events → train
python -m hgnn.trainer --epochs 200 --limit 1500

# 4. Evaluate against ground truth
python evaluate_hgnn.py --export results/
```

### Key design decisions

| Decision | Reason |
|----------|--------|
| 80% verified → train, 20% → val | Old approach (all verified → val) meant model never saw real medium/high patterns |
| `verified_label_weight = 5×` | Verified labels get higher gradient contribution even when mixed with proxy data |
| Class-weighted CE loss | DB is ~67% low — weighting prevents the model from collapsing to always predict low |
| Blind eval mode (`--blind`) | Zero-out LLM severity feature to measure pure graph reasoning vs. shortcut learning |

### Eval targets

| Metric | Min acceptable | Target |
|--------|---------------|--------|
| Overall accuracy | > LLM baseline (~70%) | ≥ 75% |
| Medium class F1 | > 0.30 | ≥ 0.50 |
| LLM-error correction rate | > 10% | ≥ 25% |

---

## Layers

| Layer | Description | Status |
|-------|-------------|--------|
| **1** | LLM disruption intelligence — this repo | ✅ Complete |
| **1b** | HGNN severity refinement | ✅ Implemented — retrain after `auto_label.py` |
| **2** | Bayesian probabilistic data fusion | 🔲 Planned |
| **3** | CVaR risk-aware routing + explainability | 🔲 Planned |

See [docs/project-handoff.md](docs/project-handoff.md) for full technical details.

---

## Team

| Member | Role |
|--------|------|
| Nikhil Kumar | Layer 1 — LLM pipeline, routing, frontend |
| TBD | Layer 2 — Bayesian fusion |
| TBD | Layer 3 — CVaR routing |
| TBD | UI / dashboard extensions |

---

## Research context

> *A Probabilistic and Explainable Tool for Context-Aware Multimodal Trip Planning Using Generative Artificial Intelligence (GenAI)*  
> ANRF ARG Pre-Proposal, IIT Kharagpur, 2026

---

## License

MIT

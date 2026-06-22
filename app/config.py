import os
from dotenv import load_dotenv

load_dotenv()

# ── Base directory — always the app/ folder regardless of cwd ────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ── LLM ──────────────────────────────────────────────────────────────────────
# Backends: "openrouter" (default) | "gemini" | "ollama"
LLM_BACKEND = os.getenv("LLM_BACKEND", "openrouter")

# OpenRouter — free tier, OpenAI-compatible, 300+ models
# Get key at: https://openrouter.ai/keys  (free, no credit card)
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Free models — swap via .env for speed vs quality tradeoff:
#   meta-llama/llama-3.3-70b-instruct:free   ← best quality, direct JSON (recommended)
#   qwen/qwen3-coder:free                    ← good at structured output
#   openai/gpt-oss-20b:free                  ← reasoning model, needs max_tokens≥1024
#   openai/gpt-oss-120b:free                 ← larger reasoning model
#   nvidia/nemotron-3-super-120b-a12b:free   ← 120B, highest quality but slowest
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "meta-llama/llama-3.3-70b-instruct:free"   # direct JSON, no reasoning overhead
)

# Gemini — fallback if OpenRouter key not set
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Ollama — local offline fallback
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")

# ── News / RSS ────────────────────────────────────────────────────────────────
NEWS_API_KEY   = os.getenv("NEWS_API_KEY", "")
NEWS_PAGE_SIZE = int(os.getenv("NEWS_PAGE_SIZE", "3"))

# ── Twitter / X ───────────────────────────────────────────────────────────────
# Removed — Twitter/X API free tier does not support search (402 Payment Required)

# ── HERE Traffic API ──────────────────────────────────────────────────────────
# Removed — HERE API key not available

# ── TomTom Traffic API ────────────────────────────────────────────────────────
# Real-time incidents: accidents, closures, road works, congestion
# Free tier: 2,500 requests/day — no credit card needed
# Get key at: https://developer.tomtom.com  → "Traffic Incidents" API
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "")

# ── OpenWeatherMap ────────────────────────────────────────────────────────────
# Real-time weather + government alerts (cyclone, flood warnings)
# Free tier: 1,000 calls/day — no credit card needed
# Get key at: https://openweathermap.org/api  → "Current Weather Data"
# Note: Weather Alerts require One Call API 3.0 (paid, ~$0.001/call)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# ── Routing ───────────────────────────────────────────────────────────────────
DEFAULT_CITY = os.getenv("DEFAULT_CITY", "Kolkata, India")

GRAPH_CACHE_PATH = os.getenv(
    "GRAPH_CACHE_PATH",
    os.path.join(_APP_DIR, "cache", "graph.pkl"),
)

# Route generation settings
MAX_ROUTES = int(os.getenv("MAX_ROUTES", "10"))

# Minimum percentage difference between routes
MIN_ROUTE_DIVERGENCE = float(
    os.getenv("MIN_ROUTE_DIVERGENCE", "0.20")
)

# Avoid repeated same route attempts
MAX_CONSECUTIVE_DUPES = int(
    os.getenv("MAX_CONSECUTIVE_DUPES", "3")
)

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(_APP_DIR, 'traffic_events.db')}",
)

# ── Scoring ───────────────────────────────────────────────────────────────────
SEVERITY_SCORES = {"low": 2, "medium": 5, "high": 10}
DEFAULT_SCORE   = 1

# ── Source enable/disable flags ───────────────────────────────────────────────
# Set any of these to "false" in .env to disable a source
ENABLE_TOMTOM      = os.getenv("ENABLE_TOMTOM",      "true").lower() == "true"
ENABLE_WEATHER     = os.getenv("ENABLE_WEATHER",     "true").lower() == "true"
ENABLE_SCRAPER     = os.getenv("ENABLE_SCRAPER",     "true").lower() == "true"
ENABLE_NEWSAPI     = os.getenv("ENABLE_NEWSAPI",     "true").lower() == "true"
ENABLE_RSS         = os.getenv("ENABLE_RSS",         "true").lower() == "true"

# Nominatim reverse geocoding — adds ~0.6s per unique TomTom coordinate
# Disable if you want faster runs and TomTom road names are sufficient
ENABLE_NOMINATIM   = os.getenv("ENABLE_NOMINATIM",   "true").lower() == "true"

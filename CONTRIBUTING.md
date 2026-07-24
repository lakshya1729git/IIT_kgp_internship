# Contributing to traffic-ai

Read this before writing any code.

---

## 1. First-time setup

```bash
git clone https://github.com/sawarn-nik/traffic-ai.git
cd traffic-ai

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r req.txt

cp .env.example .env
# Fill in your API keys — ask Nikhil for the shared dev set
```

**Never commit `.env`.** It's in `.gitignore`.

---

## 2. Running the server

```bash
cd app
uvicorn api:app --port 8000 --reload
# Open http://localhost:8000
```

`main.py` is the CLI version — use it for quick debugging without the browser.

---

## 3. Running the HGNN training pipeline

Run these from the `app/` directory in order:

```bash
# Generate synthetic balanced training data (run once or when DB is small)
python generate_training_data.py --count 400

# Auto-label TomTom events using deterministic rules
python auto_label.py

# (Optional) Label events by hand — quit anytime, progress is saved
python label_events.py --limit 100

# Backfill NULL lat/lon for non-TomTom events (Nominatim, ~1 req/s)
python backfill_coords.py

# Train the model
python -m hgnn.trainer --epochs 200 --lr 0.001 --patience 40

# Evaluate against verified labels
python evaluate_hgnn.py --export results/
```

---

## 4. Branch strategy

```
main        ← stable, protected — no direct pushes
 └── nikhil ← Nikhil's working branch
 └── dev    ← shared integration branch
      ├── feature/layer3-cvar-routing
      ├── feature/nlp-explanations
      └── fix/short-description
```

**Rules:**
- Branch off `dev`, merge back into `dev` via PR
- Never push directly to `main`
- `main` only updated after team review

Both remotes are updated with a single push (configured via `git remote`):
- `origin` → `github.com/sawarn-nik/traffic-ai`
- `lakshya` → `github.com/lakshya1729git/IIT_kgp_internship`

---

## 5. Daily workflow

```bash
# Sync with latest
git checkout dev
git pull origin dev
git checkout your-feature-branch
git rebase dev

# Work, then commit
git add app/your_file.py
git commit -m "feat(scope): what you did"

git push
```

---

## 6. Commit message format

```
feat(hgnn): add verified-label boosting to trainer V3
fix(rss_fetcher): scope queries to when:2d for recency
refactor(route_engine): extract bounds check into helper
docs(readme): update contributor credits
chore(deps): pin sqlalchemy to 2.0.50
```

| Prefix | When to use |
|--------|-------------|
| `feat` | New feature or module |
| `fix` | Bug fix |
| `refactor` | Restructure without behaviour change |
| `docs` | Docs, comments, docstrings |
| `chore` | Dependencies, config, tooling |

---

## 7. PR checklist

- [ ] `uvicorn api:app --port 8000` starts without errors
- [ ] No `.env` file in the diff
- [ ] No API keys hardcoded anywhere in the code
- [ ] New functions have docstrings
- [ ] PR description explains what changed and why
- [ ] If you touched the DB schema, the auto-migration in `models.py` is updated

---

## 8. What NOT to commit

| File / folder | Reason |
|---------------|--------|
| `.env` | API keys |
| `venv/` | ~200MB, everyone installs their own |
| `*.db` | Generated at runtime |
| `app/cache/*.pkl` | ~50MB OSM graph, auto-downloaded |
| `app/cache/*.json` | OSMnx cache, machine-specific |
| `__pycache__/` | Python bytecode |
| `.DS_Store` | macOS metadata |
| `docs/` | Local reports and LaTeX — not tracked |

---

## 9. Layer 3 — what to build next

**Files:**
- `app/routing/cost_function.py` — Equation 3 is fully implemented; integrate it into route ranking in `api.py`
- `app/routing/cvar_router.py` — **TODO**: CVaR path optimisation using the generalised cost
- `app/routing/travel_time.py` — **TODO**: travel time distributions (nominal + disruption mixture)

**Equation 3 (from proposal):**

```
c_e(t) = c_base(t)
       + λ1 · E[τ̃_e(t)]      ← expected (disruption-adjusted) travel time
       + λ2 · Var[τ̃_e(t)]    ← travel time variance (reliability)
       + λ3 · κ_e · σ_e       ← disruption risk
       + λ4 · CO2(e)          ← emissions (gCO2)
       + λ5 · Transfers(e)    ← mode-switch penalty (minutes)
```

Three weight presets are already defined in `cost_function.py`:
- `TIME_OPTIMAL` — minimise travel time, ignore variance and emissions
- `RELIABLE` — penalise variance heavily (risk-averse commuter)
- `ECO` — balance time with emissions

**Natural language explanations:**
- `app/llm/explain.py` — **TODO**: generate a plain-English justification per recommended route using HGNN attention weights + event summaries (H3 validation)

---

## 10. Module ownership

| Module | Owner |
|--------|-------|
| LLM extraction, HGNN, Bayesian fusion, API, frontend | Nikhil Mishra |
| Bus transit, multimodal routing, transit data | Lakshya Sharma |
| HGNN training pipeline, evaluation, severity rules, map matcher, Layer 3 cost function | Ruchi Mishra |

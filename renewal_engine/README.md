# Renewal Risk Intelligence Engine

A complete local-first Python project that predicts which customer accounts are likely to miss renewal and explains why, using structured data + unstructured context analyzed by a local Ollama model.

## Architecture

The engine is modular and follows a pipeline architecture:

1. **Ingestion (`features.py`, `utils.py`)**
   - Loads required files:
     - `accounts.csv`
     - `usage_metrics.csv`
     - `support_tickets.csv`
     - `nps_responses.csv`
     - `csm_notes.txt`
     - `changelog.md`
   - Cleans missing values.
   - Normalizes account names.
   - Uses fuzzy matching to align inconsistent account names before joins.
   - Filters to accounts renewing in next 90 days.

2. **Feature Engineering (`features.py`)**
   - **Usage features**: average API usage, trend, active-user percentage.
   - **Support features**: total tickets, P1 count, unresolved tickets.
   - **NPS features**: score, sentiment bucketing, LLM comment analysis.
   - **CSM features**: structured risk extraction from messy notes.

3. **LLM Layer (`llm.py`)**
   - Calls local Ollama API (`http://localhost:11434/api/generate`) using `llama3` by default.
   - Extracts structured account signals from CSM notes.
   - Translates and analyzes NPS comments.
   - Parses changelog risk events (breaking changes/deprecations/migrations).
   - Generates explanations (3 reasons + 2 actions) per account.

4. **Scoring (`scoring.py`)**
   - Weighted risk model over:
     - usage drop
     - ticket severity
     - NPS sentiment
     - CSM signals
     - changelog impact
   - Produces:
     - `risk_score` in [0, 1]
     - risk class: `LOW`, `MEDIUM`, `HIGH`

5. **Explanation + Insights (`explain.py`, `scoring.py`)**
   - Account-level reason/action generation via LLM, with robust fallbacks.
   - Advanced patterns (non-obvious insights) such as silent churn and low adoption seat risk.

6. **Orchestration (`main.py`)**
   - Runs the full pipeline.
   - Prints CLI report blocks per account.
   - Writes scored CSV output.

---

## Why an LLM is used

Traditional BI features capture numerical risk, but renewal outcomes are often influenced by qualitative signals buried in text.

LLM usage is critical for:
- Turning unstructured CSM notes into structured risk metadata.
- Interpreting multilingual NPS comments.
- Reading changelog semantics (deprecations, breakage, migration burden).
- Producing human-readable, actionable explanations.

All LLM calls run locally through Ollama (no cloud APIs).

---

## Tradeoffs

- **Pros**
  - Stronger signal extraction from messy text.
  - Explainable outputs for GTM teams.
  - Fully local deployment for data privacy.

- **Cons**
  - LLM output can vary; JSON parsing guards are included, but deterministic parsing is still imperfect.
  - Fuzzy matching improves joins but may create occasional false matches if account names are extremely ambiguous.
  - Weighted scoring is transparent and tunable, but less expressive than a trained ML model.

---

## Future Improvements

- Add historical renewal labels and train calibrated models (e.g., XGBoost + SHAP).
- Add retrieval over product docs + contracts for richer LLM context.
- Add confidence score per explanation.
- Add batch caching for Ollama calls to reduce latency.
- Add a full Streamlit UI for filtering, account drill-down, and timeline diagnostics.

---

## How to Run

### 1) Prerequisites

- Python 3.10+
- Ollama installed and running locally
- Model pulled locally:

```bash
ollama pull llama3
```

### 2) Install dependencies

```bash
cd renewal_engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Prepare input files

Place the six required input files in a folder, for example `./data`:

- `data/accounts.csv`
- `data/usage_metrics.csv`
- `data/support_tickets.csv`
- `data/nps_responses.csv`
- `data/csm_notes.txt`
- `data/changelog.md`

Expected important columns:
- `accounts.csv`: `account_name`, `renewal_date` (optional for changelog mapping: `sdk_version`, `not_upgraded`, `licensed_users`)
- `usage_metrics.csv`: `account_name`, `date`, `api_calls`, `active_users`, `licensed_users`
- `support_tickets.csv`: `account_name`, `severity`, `status`
- `nps_responses.csv`: `account_name`, `score`, `comment`

### 4) Execute

From repository root:

```bash
python -m renewal_engine.main --data-dir ./data --model llama3 --output renewal_risk_scores.csv
```

### 5) Output

- CLI output per account:

```text
----------------------------------------
Account: XYZ Corp
Risk: HIGH
Score: 0.78

Reasons:
- ...
- ...

Actions:
- ...
- ...
----------------------------------------
```

- CSV file with account-level risk scores and explanation fields.

---

## Optional Streamlit UI (Bonus)

You can quickly prototype a UI with Streamlit by reading the output CSV:
- table view
- risk filters
- account click-through for explanations

(Dependencies are already included in `requirements.txt`.)

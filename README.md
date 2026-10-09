# Universal Dataset Intelligence (UDI)

UDI is a conversational analytics and machine-learning assistant for tabular data (CSV / XLSX / XLS). You select a domain for context, upload a dataset and ask questions in plain language. Each question can lead to a different analysis: a plain aggregation, a group comparison, a trend test, anomaly detection, a correlation analysis, a regression or classification model trained on your data, a forecast, or a what-if scenario.

**Stack:** React (Vite) frontend · Flask / Python backend · pandas, scikit-learn, XGBoost, SciPy for all computation · Google Gemini only for language (planning and wording). No Streamlit.

## 1. Roles: who does what

| Layer | Responsibility | Never does |
|---|---|---|
| **Gemini** | Understands the question, resolves follow-ups, returns a *structured plan* (JSON). Words the explanation of a result that Python already verified. | Compute numbers, invent columns/targets/groups/dates/metrics, write or run code, answer from its own knowledge. |
| **Validator** (Python) | Checks the plan against the real dataset: columns exist, types fit the operation, filters match real values, enough data. | Silently substitute a column. A missing column is rejected with the list of available ones. |
| **Operations** (Python) | Compute everything: statistics, models, metrics, forecasts, charts. | Execute anything Gemini wrote. Only the ten operations below exist. |
| **React UI** | Shows operation, target, model, metrics, charts, tables, limitations, errors, Gemini/backend status. | |

Gemini is required for planning. There is **no keyword or rule-based fallback** and no mock mode: if Gemini is unavailable, the question fails with a structured `GEMINI_*` error and nothing is guessed. If only the *explanation* step fails (or Gemini's wording contains a number that is not in the verified result), the answer falls back to a Python-written summary of the same verified result, and the UI says so.

```text
question ─► Gemini plan (JSON) ─► validate against data ─► Python operation ─► verified result ─► wording (Gemini, checked) ─► UI
              schema + stats       INVALID_COLUMN,          pandas / sklearn       numbers,           every number must
              + last 4 turns       INSUFFICIENT_DATA…                              charts, tables     trace to the result
```

### What is sent to Gemini
Domain name and a short context sentence; dataset name; row count; per-column name, detected role, type, missing %, unique count, summary statistics and, for categorical columns, the distinct category values (capped for high-cardinality columns); up to `GEMINI_SAMPLE_ROWS` sample rows (default 3; set 0 to send no row values); the question; the last four turns (question, executed plan, short digest of the verified result). **The dataset itself is never sent.** For the wording step Gemini receives only the verified result digest.

## 2. Domains (context, not model locks)
Company · Hospital · College · Municipality · NGO · Community · Event Organizer.
The domain tells Gemini which vocabulary the user probably uses. It does not choose a target or a model; the dataset schema and the question decide that. The same engine serves every domain, and the plan is validated against the uploaded columns, not against the domain.

## 3. Operations (the complete controlled library)
| Operation | Typical question | Method |
|---|---|---|
| `get_summary` | "Give me an overview", "average X", "how many…" | Deterministic pandas aggregations and distributions. No ML. |
| `get_current_state` | "What is the latest X?" | Most recent observation versus the previous one. No ML. |
| `compare_groups` | "Compare X across Y", "which Y is highest?" | Group aggregation + Kruskal–Wallis / chi-square test. No ML. |
| `analyze_trend` | "Is X rising?" | Period aggregation, Mann–Kendall test, linear fit. No ML. |
| `detect_anomalies` | "Which records look unusual?" | IQR fences, z-score, Isolation Forest (flag needs ≥2 of 3 agreeing by default). Flags are *unusual*, not *wrong*. |
| `calculate_correlation` | "What is related to X?" | Pearson/Spearman (numeric), η (numeric vs category), Cramér's V (category vs category). Reports that correlation is not causation. |
| `predict_value` | "Predict X for …" | Trains and compares regression models on your data (below). |
| `predict_class` | "Will … be late/high-risk?" | Trains and compares classification models on your data. |
| `forecast` | "Forecast X for the next 14 days" | Time-aware candidates + rolling-origin backtest. |
| `run_scenario` | "What if distance grows 20%?" | Re-scores the dataset with the trained regression model; labelled model-based and non-causal. |

Not every question needs ML: averages, group comparisons and unusual-value checks are computed directly.

## 4. Dynamic model selection (uploaded data)
For `predict_value`, `predict_class`, `run_scenario` and `forecast` UDI works out the task from the plan and the data, then:
1. **Preprocesses**: drops duplicates, excludes identifier-like columns (name + uniqueness, or sequential index), constant/empty columns and date columns from tabular features, imputes missing values, one-hot encodes categories, scales numerics. High-cardinality *numeric* business variables (Sales, Revenue…) are kept.
2. **Trains candidates** — regression: Linear, Decision Tree, Random Forest, Gradient Boosting, XGBoost; classification: Logistic, Decision Tree, Random Forest, Gradient Boosting, XGBoost.
3. **Selects** by cross-validation on the training split (regression: RMSE; classification: macro-F1), then reports **honest hold-out metrics** (regression MAE / RMSE / R²; classification accuracy / precision / recall / F1 / ROC-AUC, confusion matrix, per-class table, majority-class or mean baseline) and refits the winner on all labelled rows.
4. **Explains** with permutation importance on held-out rows (SHAP is not implemented).
5. **Caches** the trained model per session (key: dataset, task, target, features, filters, config version), so a follow-up prediction or scenario reuses it. A new upload or "New chat" clears the cache.
6. **Warns** about leakage-like features (near-perfect correlation with the target), weak models (low R²), extrapolation outside the training range, and thin classes.

**Forecasting** needs a time column, a numeric target and at least 24 periods (horizon ≤ one third of history). Candidates: naive, seasonal naive, moving average, linear trend (+ seasonal profile), ridge autoregression, Random-Forest and Gradient-Boosting lag models. They are compared with a **time-ordered rolling-origin backtest** (data is never shuffled); naive baselines are always included and the report says whether the chosen model beats them. The 80% band is approximate (backtest error).

**Data-sufficiency checks** (otherwise `INSUFFICIENT_DATA` with the reason): regression ≥ 30 labelled rows; classification ≥ 40 rows, 2–30 classes, ≥ 5 rows per class; anomalies ≥ 20 values; trend ≥ 6 periods; forecast ≥ 24 periods.

## 5. Dataset requirements
- CSV (delimiter and encoding detected), XLSX, XLS (`.xls` loading relies on `xlrd` and is untested here); first non-empty sheet; header row detected automatically.
- One table, one row per record. Numeric text such as `₹1,200` and common date formats are converted.
- A date/time column is needed only for trend, current-state, forecast and date-range filters.
- Maximum upload size: `UDI_MAX_UPLOAD_MB` (default 50).

## 6. Install and run (Windows)
Prerequisites: Python 3.10+ and Node.js 18+ on PATH.
1. Unzip the project.
2. Double-click `run_windows.bat` (or run it from a terminal). It creates `backend\.venv`, installs requirements, creates `backend\.env` from `.env.example`, installs and builds the frontend, and starts Flask. It stops at the first error and shows it.
3. First run: when it reports `[ACTION NEEDED]`, open `backend\.env`, set your own `GEMINI_API_KEY` and `GEMINI_MODEL`, save, and run the script again.
4. Open http://localhost:5000.

Manual equivalent:
```bat
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      & rem then edit .env
cd ..\frontend
npm install
npm run build
cd ..\backend
python app.py
```
macOS/Linux: `./run_unix.sh`.

**Frontend development:** run Flask (`python app.py`) and, in `frontend`, `npm run dev` → http://localhost:5173 (Vite proxies `/api` to port 5000).

## 7. Environment variables (`backend/.env`)
| Variable | Meaning |
|---|---|
| `GEMINI_API_KEY` | Your key. Required for planning. Never commit it. |
| `GEMINI_MODEL` | Model name your key can call. **No default**: an unavailable or retired model is reported (`GEMINI_MODEL_UNAVAILABLE`) together with the models your key can use, never silently replaced. |
| `GEMINI_TIMEOUT_SECONDS` | Per-request timeout (default 60). |
| `GEMINI_SAMPLE_ROWS` | Sample rows sent for planning (default 3, 0 = none). |
| `FLASK_PORT` | Backend port (default 5000). |
| `UDI_MAX_UPLOAD_MB`, `UDI_SESSION_TTL_SECONDS`, `UDI_MAX_SESSIONS` | Upload limit, session lifetime, session cap. |

The status chip in the header shows Gemini state; click it to run a live check of the key and model.

## 8. Tests and checks
```bat
cd backend
python -m py_compile app.py config.py errors.py schemas.py data_profiler.py filtering.py preprocessing.py context.py result_formatter.py validator.py analytics_engine.py diagnostics_engine.py anomaly_engine.py ml_engine.py forecasting_engine.py executor.py session_manager.py explainer.py gemini_client.py planner.py reference_models.py train_models.py
python -m pytest -q
python reference_models.py --validate        & rem optional: reference models reproduce their stored metrics
cd ..\frontend
npm run build
```
The automated tests run offline. Gemini is replaced by `tests/fake_llm.py::ScriptedLLM`, a test double that returns the plan a correct Gemini would return for a scripted question; the SDK is replaced by a fake client in `tests/test_gemini_client.py`. They cover CSV/Excel loading, profiling, validation (invalid columns, insufficient data), every operation, dynamic model selection and caching, forecasting backtests, scenarios, follow-ups, clarification, new chat, error codes, strict-JSON responses, and that different questions on the same dataset produce different operations. **Live Gemini behaviour (plan quality for your phrasing) is not covered by automated tests.**

`GET /api/session/<id>/routing` returns the per-question log (question → operation, target, model, ML or not) for the session.

## 9. Sample data and reference models
`backend/sample_data/*.csv` are seven demo datasets (one per domain), for testing only. `backend/models/<domain>/` holds reference models trained on them (`MODEL_REPORT.md`). The ZIP ships the four core artifacts per domain (selected regression model, selected classification model, forecasting model, anomaly model); the 56 non-selected candidate models (14.5 MB, only for completeness) come in the optional `UDI_reference_candidates.zip` - unzip it into the project root to restore the full set, or rebuild everything with `run_windows.bat --retrain-reference`. `metadata.json` records the metrics of all candidates either way. **They are not loaded when you ask questions.** They are used for validation (`python reference_models.py --validate` re-scores them and compares with stored metrics) and as a benchmark for the dynamic pipeline (`tests/test_reference_models.py`).

## 10. Error codes
Every failure returns `{code, error, retryable}`; the UI shows a friendly card.
`GEMINI_UNAVAILABLE` · `GEMINI_MODEL_UNAVAILABLE` · `INVALID_PLAN` · `INVALID_COLUMN` · `INSUFFICIENT_DATA` · `UNSUPPORTED_OPERATION` · `MODEL_TRAINING_ERROR` · `DATASET_PARSE_ERROR` · `FILE_TOO_LARGE` · `BACKEND_ERROR` (plus `SESSION_NOT_FOUND`, `NO_DATASET`, `BAD_REQUEST`).

## 11. Security
- The API key is read from the environment / `backend/.env`; `.env` is git-ignored and must never be shared or zipped. **If a key has ever been in a shared archive, rotate it.**
- Gemini output is parsed as data only (pydantic-validated JSON). There is no `exec`/`eval`; a test scans the source for them.
- Uploads are parsed, held in memory for the session and the temporary file is deleted immediately. Sessions expire after `UDI_SESSION_TTL_SECONDS`.
- The backend has no authentication; run it on a trusted machine/network.

## 12. Limitations
- UDI works on a single flat table. It does not join tables, read images/text corpora or handle very wide/very large data efficiently (model training happens at question time; expect seconds to a minute on tens of thousands of rows).
- It is not guaranteed to work for every dataset. Messy headers, mixed-type columns, free text and unusual date formats can be mis-detected; the profile in the sidebar shows what was detected.
- Plan quality depends on Gemini and on the column names. Ambiguous questions trigger a clarification; unsupported ones are refused.
- Model-based outputs are statistical estimates: no causal claims; scenario results are only as good as the underlying model (R² is shown); forecast bands are approximate.
- Permutation importance is provided; SHAP is not.
- `.xls` loading and `run_windows.bat` have not been exercised on Windows by the author of this build.
- Sessions are in memory: restarting Flask loses them.

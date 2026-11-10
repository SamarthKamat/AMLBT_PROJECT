# UDI 2.0 architecture

```text
React UI (frontend/src)                                         Flask API (backend/app.py, served by waitress: serve.py)
  Ask · Relationships · EDA · Data quality · Reports · History
  upload / bundled dataset ────────────────────────────────►  /api/upload, /api/samples/load
                                                              load_dataset → Session.set_dataset: profile, roles, semantics,
                                                              relationships, LLM profile → SQLite + snapshot (storage.py)
  question ────────────────────────────────────────────────►  /api/chat
                                                                │
   Planner agent  planner.PlanningService  ── auto | gemini | local
        gemini_client.GeminiService.plan   (schema, semantics, relationship digest, ≤3 sample rows, last 4 turns)
        local_planner.LocalPlanner         (schema linking → masking → TF-IDF+LR intent model → slot filling)
                                                                ▼
   Validation gate  validator.validate_plan  columns exist (never substituted) · types · filters vs real values · sufficiency
                                                                ▼
   Analyst / ML agent  executor.execute → OPERATION_TABLE (18 operations)
        analytics_engine     get_summary · get_current_state · compare_groups
        diagnostics_engine   analyze_trend · calculate_correlation
        anomaly_engine       detect_anomalies
        ml_engine            predict_value · predict_class · run_scenario  (AutoML + explainability_engine SHAP)
        forecasting_engine   forecast  (9 candidates incl. ETS, ARIMA/SARIMA; rolling-origin backtest)
        business_engine      query_records · pivot_table · explain_change · segment_records
        eda_engine           run_eda · assess_data_quality (data_quality.py) · describe_relationships (relationships.py)
        report_engine        generate_report  → agents.build_report (multi-agent) → PDF / Excel / JSON
                                                                ▼
   Business Analyst agent  business_analyst.business_insights  findings · suggested actions · evidence · priority
                                                                ▼
   Narrator  planner.explain_result  Gemini wording accepted only if every number traces to the verified result
                                                                ▼
   response {answer, plan, result, planner, agent_trace, timing}  → ResultView (metrics, charts, insights, SHAP, trace)
```

## Modules
| File | Role |
|---|---|
| `config.py` | Environment, paths, domains, planner mode, auth, rate limits, bundled datasets. |
| `schemas.py` | `AnalysisPlan` and the 18-operation vocabulary. |
| `gemini_client.py` | Gemini prompts (all 18 operations), JSON schema, error mapping. |
| `local_planner.py`, `planner_corpus.py`, `train_planner.py` | Offline planner: schema linker, masking, intent classifier, slot filling; corpus and training. |
| `planner.py` | Planning modes and grounded explanation. |
| `data_profiler.py` | Loading, type coercion, role inference, profile. |
| `semantics.py` | Business semantic type of every column (money, quantity, rate, geography, outcome, …), additivity, business functions. |
| `relationships.py` | Association matrix, hierarchies, derived columns, duplicates, date rules, keys, data dictionary. |
| `data_quality.py` | Quality score, issues and treatments, cleaning actions, before/after comparison. |
| `validator.py`, `filtering.py` | Plan validation and filters. |
| `*_engine.py`, `executor.py` | The operations and their dispatcher. |
| `explainability_engine.py` | Permutation-sampling Shapley values (global + local, waterfall). |
| `business_analyst.py`, `agents.py`, `report_engine.py` | Business findings, multi-agent report orchestration, PDF/Excel rendering. |
| `session_manager.py`, `storage.py` | Sessions with owner, persistence in SQLite + dataset snapshots, restore after restart, deletion. |
| `ops.py` | Authentication/authorisation, rate limiting, background jobs, metrics. |
| `reference_models.py`, `train_models.py`, `models/<domain>/` | Phase 1 reference models (validation/demo only). |

## Key formulas
* Quality score: `Q = Σ_d w_d · 100 · (1 − min(1, k_d · r_d))`, weights (.30,.20,.20,.15,.15), sensitivities (1,4,4,4,2) for completeness, uniqueness, validity, consistency, outliers.
* Association: |Pearson| or |Spearman| (numeric–numeric), η = √(SS_between/SS_total) (numeric–categorical), bias-corrected Cramér's V (categorical–categorical).
* SHAP: `φ_j = 1/K Σ_k [f(z_{π_k,≤j}) − f(z_{π_k,<j})]` over K antithetic orderings paired with K background rows; `f(x) = φ_0 + Σ φ_j` exactly.
* Model selection: among candidates within one standard error of the best CV score (σ/√folds) choose the least complex.
* Root cause (sum KPI): `Δ = Σ_g (M_gB − M_gA)`; (mean KPI): `Δ = Σ (w_gB − w_gA) m_gA + Σ w_gB (m_gB − m_gA)`.

## Result contract
`{success, operation, title, summary, metrics, table, table_title, tables{}, charts[], insights[], business_insights[], evaluation, explainability{items, shap, local}, forecast, scenario, prediction, report, quality, relationships, preferred_chart, limitations[], assumptions[], warnings[], provenance}`.
Chart types: `bar` (horizontal / grouped / stacked), `line`, `area`, `pie`, `scatter` (highlight / group / identity), `forecast` (80 % and 95 % bands), `heatmap`, `box`, `waterfall`.

# UDI architecture

```text
React UI (frontend/src)                                   Flask API (backend/app.py)
  upload ─────────────────────────────────────────────►  /api/upload   load_dataset → profile_dataframe → roles → Session
  question ───────────────────────────────────────────►  /api/chat
                                                              │
                                                              ▼
                                                    planner.PlanningService
                                                    profile_for_llm (schema+stats+≤3 rows) + last 4 turns
                                                              │
                                                              ▼
                                                    gemini_client.GeminiService.plan  → AnalysisPlan (pydantic JSON)
                                                              │      (GEMINI_* errors stop here; no fallback)
                                                              ▼
                                                    validator.validate_plan
                                                      columns exist? types fit? filters match real values?
                                                      enough data? → ValidatedPlan | ClarificationNeeded | UDIError
                                                              │
                                                              ▼
                                                    executor.execute → OPERATION_TABLE[operation]
        get_summary / get_current_state / compare_groups ....... analytics_engine   (pandas, no ML)
        analyze_trend / calculate_correlation .................. diagnostics_engine (Mann–Kendall, Pearson/Spearman, η, Cramér's V)
        detect_anomalies ....................................... anomaly_engine     (IQR, z-score, Isolation Forest)
        predict_value / predict_class / run_scenario ........... ml_engine          (candidate models, CV selection, hold-out metrics, cache)
        forecast ............................................... forecasting_engine (rolling-origin backtest)
                                                              │
                                                              ▼
                                                    verified result (result_formatter.jsonsafe)
                                                              │
                                                              ▼
                                                    planner.explain_result → Gemini wording, accepted only if every number
                                                    traces to the result (explainer.check_grounded); else Python summary
                                                              │
                                                              ▼
React: ResultView (metrics, charts, tables, confusion matrix, importance, limitations, details) · ErrorCard · status chip
```

## Modules
| File | Role |
|---|---|
| `config.py` | Environment, paths, the seven domain context blocks. |
| `schemas.py` | `AnalysisPlan` and the operation list (the only vocabulary Gemini may use). |
| `gemini_client.py` | Prompts, JSON schema, error mapping, plan + explain calls. |
| `data_profiler.py` | Loading, coercion, role inference (numeric/categorical/datetime/identifier/…), profile. |
| `validator.py`, `filtering.py` | Plan validation against data, filters with real-value matching, sufficiency checks. |
| `preprocessing.py`, `context.py` | Feature selection and encoding; per-session model cache. |
| `*_engine.py`, `executor.py` | The ten operations and their dispatcher (asserted equal to the operation list). |
| `session_manager.py` | In-memory sessions: dataset, conversation, pending clarification, routing log, cache. |
| `reference_models.py`, `train_models.py`, `models/`, `sample_data/` | Reference/validation/demo only; never loaded when answering questions. |

## Result contract
`{success, operation, title, summary, metrics, table, table_title, tables{}, charts[], evaluation, explainability, forecast, scenario, prediction, limitations[], assumptions[], warnings[], provenance}`.
Chart types: `bar` (optionally horizontal), `line` (one or several series), `scatter` (optional highlight / identity line), `forecast` (actual, forecast, 80% band).

## Follow-ups
The last four turns (question, executed plan, digest of the verified result) go to Gemini, which returns a complete plan carrying over target/aggregation/filters and sets `uses_previous_context`. A clarification is stored as `pending_clarification` and shown to Gemini with the next message. "New chat" and re-upload delete the context and cached models.

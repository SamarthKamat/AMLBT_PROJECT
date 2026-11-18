# Phase 2: what changed and why

## Response to the first-review feedback (20.2 / 30)
| Feedback | Action in Phase 2 | Evidence |
|---|---|---|
| 7 pages exceed the 6-page limit | Final paper is 5 pages including references (IEEE conference format) | `report/UDI2_Final_Report.pdf` |
| AI-use statement says tools "may have been used" | Precise declaration naming Gemini (runtime), ChatGPT and Claude Code (development) and what each was used for | paper §VIII-A |
| Evaluate routing, column mapping, invalid requests and numerical correctness on held-out questions; publish pass/fail logs; separate implemented from executed tests | 104-question held-out benchmark over 6 datasets with gold labels and pandas oracles; frozen run 1 and post-fix run 2 reported separately; per-question logs; executed test report (168/168) | `evaluation/`, `TEST_REPORT.md` |
| Test two or three real public datasets with realistic missingness and noise | Four public datasets (Global Superstore, IBM HR, IBM Telco, UCI Bank Marketing) plus a controlled defect-injection study | `evaluation/results/real_datasets.json` |
| Generated data limits generalisation claims | Generated reference results kept separate and explicitly not used as real-world evidence | paper §V-D |

## Requested capabilities
| Requirement | Implementation |
|---|---|
| Column-wise relations clearly defined | `relationships.py`, `semantics.py`; Relationships tab; `describe_relationships` operation |
| Agent answers any question on the data (e.g. "which country has the highest sales") | 18 operations, Gemini or offline local planner, validation gate; Ask tab |
| Outlier detection | `detect_anomalies` (IQR + z + Isolation Forest, multivariate) and quality-engine outlier dimension |
| EDA | `run_eda`; EDA tab |
| Reports for marketing / HR / operations / sales / finance / customer | `agents.py`, `report_engine.py`; Reports tab; PDF / Excel / JSON |
| Visualisations that explain the answer | every result carries charts (bar, pie, line, area, scatter, forecast bands, heatmap, box, waterfall) with a view switcher and CSV export |

## Design-review items (from the external review) and status
| Item | Status |
|---|---|
| Forecasting: frequencies, ARIMA/ETS, intervals, backtesting | done (`forecasting_engine.py`); Prophet not added (heavy dependency) |
| Invalid forecast target silently replaced | verified not to happen in the code base; regression test added (`test_forecast_never_substitutes_an_invalid_target`) |
| Single-split model ranking | CV mean ± sd, one-standard-error rule, paired fold test, tuning, learning curve, temporal validation |
| SHAP explanations | `explainability_engine.py` (global + local, exact additivity) |
| Data-quality score and cleaning with before/after | `data_quality.py`; Data quality tab |
| Business decision intelligence (KPIs, root cause, recommendations with quantified impact) | `business_engine.py`, `business_analyst.py`, `agents.py` |
| Multi-agent architecture with controlled tools | `agents.py` (Data Analyst, Business Analyst, ML, Report agents + trace) |
| Production engineering | SQLite persistence, auth + ownership, rate limits, background jobs, metrics, deletion controls, Docker, waitress |

## Response to the second review (24.0 / 30)
| Comment | Action | Evidence |
|---|---|---|
| Bank result includes call duration (post-outcome leakage) | `exclude_columns` ("predict y without duration"); dominant-feature warning (≥30% of SHAP attribution) now flags duration; the reported Bank result is the pre-call model (ROC-AUC 0.798 [0.772, 0.824]) | `real_datasets.json`, paper Table V |
| Benchmark written by the same tool as the planner | Wording now matches the AI declaration; new set v2 written after all fixes, frozen (SHA-256 + git commit) and run once; 56 of its 99 questions use five schemas never used in development | `evaluation/benchmark_questions_v2.*`, paper §IV |
| Small sample, no confidence intervals; run 2 not held out | Wilson 95% CIs for every rate, bootstrap CIs for every model metric; fresh held-out run v2: 87.9% [80.0, 92.9] | `benchmark_v2_local_heldout.json` |
| No comparison | Keyword-rule router and exact-linking ablation on both sets with exact McNemar tests (UDI vs router p < 1e-8); plain-LLM and Gemini scripts provided (need an API key) | `paired_tests.json`, paper Table IV, Fig. 3 |
| Dark screenshots unreadable in print | Replaced by vector figures drawn from the verified outputs | `report/figures/fig_example.pdf` |
| URL typeset wrongly; stray ", ;" in Eq. 3; Superstore sampling unclear; V1 score quoted | All fixed; extract provenance documented | paper §IV, `sample_data/real/SOURCES.md` |

## Final revision (after the similarity / AI reports)
* Found and fixed a real forecasting defect while preparing figures: AIC-selected ARIMA won the Superstore backtest but diverged when refit on the full history. A sanity guard now rejects implausible refits and falls back to the next-best model (trend + seasonal; RMSE 14,238, MAPE 25.1%, 10.5% better than baseline), with a regression test.
* New figures generated from the engine: a six-panel results gallery (ROC incl. the leakage pair, SHAP drivers, guarded forecast with 80/95% bands, root-cause waterfall, defect-injection robustness, per-dataset held-out accuracy) and per-operation accuracy / latency.
* A recorded, unedited HR case-study session (Table VI). The paper is exactly six full pages.

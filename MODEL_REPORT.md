# UDI Model Report

## Phase 2: models selected at run time on public datasets
Produced by `python evaluation/run_real_datasets.py` (seed 42; full numbers in `evaluation/results/real_datasets.json`).

| Dataset / target | Selected (how) | CV (mean ± sd) | Held-out | Baseline | Top SHAP drivers |
|---|---|---|---|---|---|
| Global Superstore / Profit | Random Forest (one-SE rule over GBM; tuned) | RMSE 395.7 ± 43.3 | R² 0.707, RMSE 290.4 | RMSE 536.2 | Discount, Sales, Shipping Cost |
| IBM HR / Attrition | XGBoost | macro-F1 0.718 ± 0.042 | F1 0.636, ROC-AUC 0.796, PR-AUC 0.504 | acc. 0.840 | OverTime, MonthlyIncome, Age |
| IBM Telco / Churn | Random Forest | macro-F1 0.734 ± 0.012 | F1 0.728, ROC-AUC 0.834, PR-AUC 0.640 | acc. 0.735 | Contract, InternetService, tenure |
| UCI Bank / y | Random Forest | macro-F1 0.782 ± 0.011 | F1 0.793, ROC-AUC 0.944, PR-AUC 0.661 | acc. 0.888 | duration, emp.var.rate, nr.employed |

Local planner intent model: TF-IDF (word 1–3 + char 2–5 grams) + logistic regression, 1,658 templated training questions, 19 classes, in-corpus 5-fold CV accuracy 0.963 ± 0.011 (`backend/models/planner/training_report.json`). Its real evaluation is the held-out benchmark (README §6).

# Phase 1: domain reference models

> **Reference / validation / demonstration only.** These artifacts document the seven bundled demo datasets and serve as a benchmark. UDI never loads them when answering questions about an uploaded dataset: each question trains and selects its own models (see README, section 4). `python backend/reference_models.py --validate` reloads them and checks that they reproduce the stored metrics.

Each domain has explicitly named reference artifacts. This table lists the four core artifacts shipped in the main ZIP; the non-selected candidates (listed with their metrics in each `metadata.json`) are in the optional `UDI_reference_candidates.zip`. Regression and classification candidates are all trained for each domain; the selected model is recorded in metadata. Forecasting and anomaly models are also trained per domain.

| Domain | Selected Regression | Selected Classification | Forecasting | Anomaly |
|---|---|---|---|---|
| college | `college_regression_linear.joblib` | `college_classification_random_forest.joblib` | `college_forecasting_random_forest.joblib` | `college_anomaly_isolation_forest.joblib` |
| community | `community_regression_gradient_boosting.joblib` | `community_classification_xgboost.joblib` | `community_forecasting_random_forest.joblib` | `community_anomaly_isolation_forest.joblib` |
| company | `company_regression_linear.joblib` | `company_classification_logistic.joblib` | `company_forecasting_random_forest.joblib` | `company_anomaly_isolation_forest.joblib` |
| event_organizer | `event_organizer_regression_gradient_boosting.joblib` | `event_organizer_classification_gradient_boosting.joblib` | `event_organizer_forecasting_random_forest.joblib` | `event_organizer_anomaly_isolation_forest.joblib` |
| hospital | `hospital_regression_linear.joblib` | `hospital_classification_logistic.joblib` | `hospital_forecasting_random_forest.joblib` | `hospital_anomaly_isolation_forest.joblib` |
| municipality | `municipality_regression_linear.joblib` | `municipality_classification_xgboost.joblib` | `municipality_forecasting_random_forest.joblib` | `municipality_anomaly_isolation_forest.joblib` |
| ngo | `ngo_regression_gradient_boosting.joblib` | `ngo_classification_logistic.joblib` | `ngo_forecasting_random_forest.joblib` | `ngo_anomaly_isolation_forest.joblib` |

## Candidate algorithms trained for every domain

Regression: Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, XGBoost Regressor.

Classification: Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Gradient Boosting Classifier, XGBoost Classifier.

Forecasting: Random Forest lag-based model with time-ordered backtesting.

Anomaly detection: Isolation Forest.

Uploaded datasets are never routed to these models: the runtime ML engine always trains fresh candidate models from the uploaded data.
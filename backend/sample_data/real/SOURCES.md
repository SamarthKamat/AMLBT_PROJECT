# Real-world datasets used in UDI 2.0

All files are public datasets, redistributed unchanged (CSV) for reproducibility. They were downloaded from public GitHub mirrors on 2026-11-10.

| File | Dataset | Original source | Mirror used | Rows × cols | Used for |
|---|---|---|---|---|---|
| `global_superstore_sample.csv` | Global Superstore (1,000-order extract) | Tableau "Global Superstore" sample data | github.com/yannie28/Global-Superstore | 1,000 × 24 | sales, operations, finance; forecasting; country-level questions |
| `ibm_hr_attrition.csv` | IBM HR Analytics Employee Attrition & Performance | IBM (fictional data created by IBM data scientists) | github.com/IBM/employee-attrition-aif360 | 1,470 × 35 | HR reports, attrition classification |
| `ibm_telco_churn.csv` | IBM Telco Customer Churn | IBM sample data | github.com/IBM/telco-customer-churn-on-icp4d | 7,043 × 21 | marketing / customer reports, churn classification; contains blank `TotalCharges` (real missingness) |
| `uci_bank_marketing.csv` | Bank Marketing (bank-additional-full) | S. Moro, P. Cortez, P. Rita (2014), UCI Machine Learning Repository | github.com/selva86/datasets | 41,188 × 21 | marketing reports, subscription classification; 'unknown' categories |

Note: the IBM HR dataset is itself a simulated dataset published by IBM; it is treated as a public benchmark, not as observations of a real company.

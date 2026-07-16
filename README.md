# Risk Analytics Engine

An end-to-end credit risk modeling and analytics pipeline built with Python, LightGBM, and XGBoost.

This project simulates a real-world financial risk modeling workflow, including:

- feature screening
- supervised variable selection
- tree-based feature filtering
- LightGBM / XGBoost modeling
- model evaluation
- stability analysis
- score enhancement analysis
- automated model reporting

The repository is designed as a reusable modeling framework for:
- credit risk analytics
- fintech modeling
- banking data science
- machine learning model development

---

# Project Structure

```bash
risk-analytics-engine/
│
├── notebooks/
│   ├── 01.无监督筛选.ipynb
│   ├── 02.有监督筛选.ipynb
│   ├── 03.单变量初筛.ipynb
│   ├── 04.决策树初筛_不含评分.ipynb
│   ├── 04.决策树初筛_含评分.ipynb
│   ├── 05.LGBM不含评分.ipynb
│   ├── 05.LGBM含评分.ipynb
│   ├── 06.LGBM模型开发报告整理.ipynb
│   ├── 07.XGB模型优化.ipynb
│   ├── 08.最终模型结论整理.ipynb
│   └── 09.正式模型开发报告.ipynb
│
├── scripts/
│   └── generate_demo_data.py
│
├── src/
│   └── 函数代码包.py
│
├── docs/
│   └── data_schema.md
│
├── requirements.txt
└── README.md
```

---

# Workflow Overview

## 01. 无监督筛选 (Unsupervised Feature Screening)

Perform:
- missing rate analysis
- dominant value analysis
- PSI stability analysis

Goal:
Remove unstable or low-quality variables before modeling.

---

## 02. 有监督筛选 (Supervised Feature Screening)

Evaluate variable predictive power using:
- KS
- AUC
- monthly KS
- overall performance

Goal:
Identify variables with meaningful risk discrimination ability.

---

## 03. 单变量初筛 (Initial Feature Filtering)

Combine:
- PSI thresholds
- KS thresholds
- coverage rules
- business exclusion lists

Goal:
Build the initial candidate variable pool.

---

## 04. 决策树初筛 (Tree-based Feature Filtering)

Use:
- LightGBM feature importance
- product prefix control
- variable redundancy reduction

Two versions:
- without external score variables
- with external score variables

Goal:
Reduce dimensionality and retain strong variables.

---

## 05. LGBM Modeling

Train LightGBM models for:
- y1 target
- y2 target
- score-enhanced versions
- non-score versions

Outputs:
- KS
- AUC
- feature importance
- model stability metrics

---

## 06. LGBM Report Aggregation

Aggregate:
- model metrics
- feature importance
- ranking results
- business conclusions

Generate standardized model summary tables.

---

## 07. XGB Model Optimization

Train and optimize XGBoost models.

Includes:
- hyperparameter tuning
- stability comparison
- score enhancement analysis
- algorithm comparison with LGBM

---

## 08. Final Model Comparison

Compare:
- LGBM vs XGB
- y1 vs y2
- score vs non-score models

Automatically generate:
- model rankings
- recommendation conclusions
- algorithm comparison tables

---

## 09. Formal Modeling Report

Generate a structured modeling report including:
- project background
- feature engineering workflow
- model evaluation
- score analysis
- ranking stability
- final recommendations

Output format:
- Excel-based formal modeling report

---

# Demo Data

For privacy protection, the original business data is NOT included.

Instead, the repository provides:

```bash
scripts/generate_demo_data.py
```

which generates simulated datasets with similar structure and workflow logic.

The demo data preserves:
- feature naming conventions
- variable engineering logic
- pipeline structure
- modeling workflow

while removing all sensitive business information.

---

# Tech Stack

- Python
- Pandas
- NumPy
- LightGBM
- XGBoost
- Scikit-learn
- OpenPyXL
- Jupyter Notebook

---

# Modeling Topics Covered

This project covers many practical industry modeling techniques, including:

- PSI stability analysis
- KS/AUC evaluation
- feature importance filtering
- score enhancement analysis
- model decay analysis
- OOT validation
- decision table monotonicity
- Lift analysis
- model ranking systems
- automated report generation

---

# Notes

This repository is a sanitized educational version of a real-world risk modeling workflow.

All:
- customer information
- business identifiers
- sensitive variables
- production data

have been removed or anonymized.


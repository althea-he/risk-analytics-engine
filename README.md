# Credit Risk Modeling Pipeline Template

This repository is a sanitized, GitHub-friendly version of a credit-risk modeling workflow.

It includes notebooks from unsupervised screening to final model report:

| Notebook | Purpose |
|---|---|
| 01.无监督筛选.ipynb | Basic statistics, match rate, same rate, PSI |
| 02.有监督筛选.ipynb | Single-variable AUC/KS, monthly KS |
| 03.单变量初筛.ipynb | Apply PSI/KS/statistical screening rules |
| 04.决策树初筛_不含评分.ipynb | LGBM-based feature/product screening without score |
| 04.决策树初筛_含评分.ipynb | LGBM-based feature/product screening with score |
| 05.LGBM不含评分.ipynb | Final LGBM model without score |
| 05.LGBM含评分.ipynb | Final LGBM model with score |
| 06.LGBM模型开发报告整理.ipynb | LGBM result summary |
| 07.XGB模型优化.ipynb | XGB comparison / optimization |
| 08.最终模型结论整理.ipynb | Final model comparison and recommendation |
| 09.正式模型开发报告.ipynb | Formal Excel report generation |

## Privacy note

No raw customer data is included.  
All paths, customer names, and product fields are anonymized.

## Expected data location

Place raw data under:

```text
data/raw/
```

Outputs will be saved to:

```text
data/output/
```

## Main dependency examples

```bash
pip install pandas numpy scikit-learn lightgbm xgboost hyperopt openpyxl joblib
```

## Data schema

See `docs/data_schema.md`.

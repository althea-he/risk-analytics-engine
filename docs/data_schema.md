# Data Schema

This project does not include raw data.  
To run the notebooks, prepare CSV files with the following structure.

## 1. Modeling sample file

Example filename:

```text
sample_modeling_y1.csv
sample_modeling_y2.csv
```

Required columns:

| Column | Meaning |
|---|---|
| id | anonymized user id |
| cell | anonymized phone/hash |
| name | anonymized name/hash |
| apply_date | application date |
| month | application month |
| flag | train / valid / oot |
| y | binary target label |

Optional columns:

| Column | Meaning |
|---|---|
| y1 | short-period target |
| y2 | long-period target |

## 2. Product feature files

Example filenames:

```text
product_features_part1.csv
product_features_part2.csv
```

Required join keys:

| Column |
|---|
| id |
| cell |
| name |
| apply_date |

Feature naming convention:

| Example | Meaning |
|---|---|
| ir_m1_cell_x_id_cnt | product prefix = ir |
| alft_m3m12_n_a_ratio | product prefix = alft |
| score_base | score variable |

## 3. Delete-list file

Example filename:

```text
delete_fields.xlsx
```

Required column:

| Column | Meaning |
|---|---|
| enname | feature names to remove |

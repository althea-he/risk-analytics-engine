import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(2024)

root = Path(__file__).resolve().parents[1]
raw = root / "data" / "raw"
raw.mkdir(parents=True, exist_ok=True)

n = 5000

df = pd.DataFrame({
    "id": [f"id_{i:06d}" for i in range(n)],
    "cell": [f"cell_{i:06d}" for i in range(n)],
    "name": [f"name_{i:06d}" for i in range(n)],
    "apply_date": pd.date_range("2023-11-01", periods=n, freq="H").strftime("%Y%m%d"),
})

df["month"] = df["apply_date"].str[:6]
df["flag"] = np.random.choice(["train", "valid", "oot"], size=n, p=[0.6, 0.2, 0.2])

base_risk = np.random.normal(0, 1, n)
df["score_base"] = np.clip(700 - 80 * base_risk + np.random.normal(0, 30, n), 300, 1000)

for prefix in ["ir", "alft", "aps", "rc", "ice"]:
    for j in range(1, 9):
        df[f"{prefix}_var{j}"] = base_risk + np.random.normal(0, 1, n)

logit_y1 = -3.8 + 0.8 * base_risk + 0.3 * df["ir_var1"]
prob_y1 = 1 / (1 + np.exp(-logit_y1))
df["y1"] = (np.random.rand(n) < prob_y1).astype(int)

logit_y2 = -2.5 + 0.7 * base_risk + 0.2 * df["alft_var2"]
prob_y2 = 1 / (1 + np.exp(-logit_y2))
df["y2"] = (np.random.rand(n) < prob_y2).astype(int)

sample_y1 = df[["id", "cell", "name", "apply_date", "month", "flag", "y1", "y2"]].rename(columns={"y1": "y"})
sample_y2 = df[["id", "cell", "name", "apply_date", "month", "flag", "y1", "y2"]].rename(columns={"y2": "y"})

feature_cols = ["id", "cell", "name", "apply_date", "score_base"] + [
    c for c in df.columns if any(c.startswith(p + "_") for p in ["ir", "alft", "aps", "rc", "ice"])
]

product_features = df[feature_cols]

sample_y1.to_csv(raw / "sample_modeling_y1.csv", index=False)
sample_y2.to_csv(raw / "sample_modeling_y2.csv", index=False)
product_features.to_csv(raw / "product_features.csv", index=False)

delete_fields = pd.DataFrame({"enname": ["rc_var8"]})
delete_fields.to_excel(raw / "delete_fields.xlsx", index=False)

print("Demo data generated under:", raw)

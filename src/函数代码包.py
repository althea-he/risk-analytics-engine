# -*- coding: utf-8 -*-
"""
Utility functions for a sanitized credit-risk modeling project.

This file is reconstructed for learning/interview review.
It does not contain company data or private paths.
"""

import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def read_csv_safely(file_path, **kwargs):
    """Read CSV with UTF-8 first, then GBK fallback."""
    try:
        return pd.read_csv(file_path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="gbk", **kwargs)


def get_feature_type(series, var_name=None):
    """Classify feature as score / categorical / numeric."""
    if var_name is not None and str(var_name).startswith("score"):
        return "评分"
    if str(series.dtype) in ["object", "category"]:
        return "类别型"
    return "数值型"


def get_product_prefix(var_name):
    """Extract product/module prefix from a feature name."""
    if str(var_name).startswith("score"):
        return "score"
    return str(var_name).split("_")[0]


def auc_gini_ks(y_score, y_true, example="sample"):
    """Calculate AUC, Gini and KS."""
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    gini = 2 * auc - 1
    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score)
    ks = np.max(tpr - fpr)
    return pd.DataFrame([{
        "auc": auc,
        "gini": gini,
        "ks": ks,
        "example": example
    }])


def var_ks(df_data, col_var, flagy="y"):
    """Calculate single-variable AUC and KS."""
    rows = []
    for col in col_var:
        try:
            auc = roc_auc_score(y_true=df_data[flagy], y_score=df_data[col])
            fpr, tpr, _ = roc_curve(y_true=df_data[flagy], y_score=df_data[col])
            ks = np.max(np.abs(tpr - fpr))
            rows.append({"var": col, "auc": auc, "ks": ks})
        except Exception:
            rows.append({"var": col, "auc": np.nan, "ks": np.nan})
    return pd.DataFrame(rows)


def var_match_rate(df_data, miss_value=-999):
    """Calculate non-missing rate using a filled missing value."""
    rows = []
    for col in df_data.columns:
        match_rate = (df_data[col] != miss_value).mean()
        rows.append({"var": col, "match_rate": match_rate})
    return pd.DataFrame(rows)


def var_same_rate(df_data):
    """Calculate the maximum single-value proportion for each variable."""
    rows = []
    for col in df_data.columns:
        if len(df_data[col]) == 0:
            same_rate = np.nan
        else:
            same_rate = df_data[col].value_counts(dropna=False, normalize=True).iloc[0]
        rows.append({"var": col, "same_rate": same_rate})
    return pd.DataFrame(rows)


def calc_basic_stat(df, var_cols):
    """Calculate dtype, match rate, same rate and unique count."""
    rows = []
    for col in var_cols:
        s = df[col]
        rows.append({
            "var": col,
            "dtype": get_feature_type(s, col),
            "match_rate": s.notna().mean(),
            "miss_rate": s.isna().mean(),
            "same_rate": s.value_counts(dropna=False, normalize=True).iloc[0] if len(s) > 0 else np.nan,
            "nunique": s.nunique(dropna=True)
        })
    return pd.DataFrame(rows)


def calc_psi(expected, actual, bins=10):
    """Calculate PSI between expected and actual distributions."""
    expected = pd.Series(expected)
    actual = pd.Series(actual)

    try:
        if pd.api.types.is_numeric_dtype(expected):
            q = np.linspace(0, 1, bins + 1)
            points = expected.dropna().quantile(q).drop_duplicates().values
            if len(points) <= 2:
                return np.nan
            expected_bins = pd.cut(expected, bins=points, include_lowest=True, duplicates="drop")
            actual_bins = pd.cut(actual, bins=points, include_lowest=True, duplicates="drop")
        else:
            expected_bins = expected.astype(str)
            actual_bins = actual.astype(str)

        expected_dist = expected_bins.value_counts(dropna=False, normalize=True)
        actual_dist = actual_bins.value_counts(dropna=False, normalize=True)

        idx = expected_dist.index.union(actual_dist.index)
        expected_dist = expected_dist.reindex(idx, fill_value=0.0001).replace(0, 0.0001)
        actual_dist = actual_dist.reindex(idx, fill_value=0.0001).replace(0, 0.0001)

        return ((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)).sum()
    except Exception:
        return np.nan


def calc_psi_by_var(train_valid_df, oot_df, var_cols, bins=10):
    """Batch PSI calculation."""
    rows = []
    for i, col in enumerate(var_cols):
        if i % 500 == 0:
            print(f"PSI progress: {i}/{len(var_cols)}")
        psi = calc_psi(train_valid_df[col], oot_df[col], bins=bins)
        rows.append({
            "var": col,
            "dtype": get_feature_type(train_valid_df[col], col),
            "psi_max": psi,
            "psi_avg": psi
        })
    return pd.DataFrame(rows)


def model_dictionary_score(df_data, var, bins=10, flagy=None, risk_direction="low"):
    """
    Build a decision table.

    risk_direction:
    - "low": lower score = higher risk
    - "high": higher score/probability = higher risk
    """
    assert isinstance(bins, int) or isinstance(bins, list)

    if isinstance(bins, int):
        point = [df_data[var].quantile(i / bins) for i in range(bins + 1)]
        point = sorted(list(pd.Series(point).drop_duplicates()))
    else:
        point = bins.copy()

    if len(point) < 2:
        return pd.DataFrame()

    if risk_direction == "low":
        loop_zip = zip(point[:-1], point[1:])
    else:
        loop_zip = zip(point[:-1][::-1], point[1:][::-1])

    total_n = df_data.shape[0]
    pass_n = total_n
    reject_n = 0
    pass_rate = 1

    rows = []

    if flagy:
        total_good = (df_data[flagy] == 0).sum()
        total_bad = (df_data[flagy] == 1).sum()
        total_bad_rate = total_bad / max(total_n, 1)
        pass_bad = total_bad
        pass_good = total_good
        acu_good = 0
        acu_bad = 0

    for i, j in loop_zip:
        right_closed = (j == point[-1])
        if right_closed:
            mask = (df_data[var] >= i) & (df_data[var] <= j)
            range_label = f"[{i},{j}]"
        else:
            mask = (df_data[var] >= i) & (df_data[var] < j)
            range_label = f"[{i},{j})"

        cur = df_data[mask]
        range_n = cur.shape[0]
        range_rate = range_n / max(total_n, 1)

        row = {
            "区间段": range_label,
            "区间人数": range_n,
            "区间占比": "%.2f%%" % (range_rate * 100),
            "通过率": "%.2f%%" % (pass_rate * 100)
        }

        if flagy:
            range_good = (cur[flagy] == 0).sum()
            range_bad = (cur[flagy] == 1).sum()
            range_bad_rate = range_bad / max(range_n, 1)
            range_lift = range_bad_rate / max(total_bad_rate, 1e-9)

            acu_good += range_good
            acu_bad += range_bad

            acu_bad_ratio = acu_bad / max(total_bad, 1)
            acu_good_ratio = acu_good / max(total_good, 1)
            ks = abs(acu_bad_ratio - acu_good_ratio)

            pass_bad_rate = pass_bad / max(pass_n, 1)
            reject_bad = total_bad - pass_bad
            reject_bad_rate = reject_bad / max(reject_n, 1)
            reject_lift = reject_bad_rate / max(total_bad_rate, 1e-9)

            row.update({
                "区间坏客户数": range_bad,
                "区间坏客户率": "%.2f%%" % (range_bad_rate * 100),
                "区间提升度": "%.2f%%" % (range_lift * 100),
                "累计坏客户占比": "%.2f%%" % (acu_bad_ratio * 100),
                "累计好客户占比": "%.2f%%" % (acu_good_ratio * 100),
                "好坏区分程度(ks)": "%.2f%%" % (ks * 100),
                "通过坏客户率": "%.2f%%" % (pass_bad_rate * 100),
                "拒绝坏客户率": "%.2f%%" % (reject_bad_rate * 100),
                "累计提升度": reject_lift
            })

            pass_bad -= range_bad
            pass_good -= range_good

        pass_n -= range_n
        reject_n = total_n - pass_n
        pass_rate = pass_n / max(total_n, 1)

        rows.append(row)

    return pd.DataFrame(rows)


def solve_scorecard_params(min_score, max_score, prob_ary):
    """Map predicted probability to a score range."""
    prob_ary = np.clip(np.asarray(prob_ary), 1e-6, 1 - 1e-6)
    odds_ary = np.log(prob_ary / (1 - prob_ary))
    B = (max_score - min_score) / (odds_ary.max() - odds_ary.min())
    A = max_score + B * odds_ary.min()
    return round(A, 0), round(B, 0)


def score_divide(x, bins):
    """Assign score to a score-bin index."""
    for idx, point in enumerate(bins[:-1]):
        if x < point:
            return idx
    return len(bins) - 1

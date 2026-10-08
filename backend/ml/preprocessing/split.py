"""
Split strategy.

The IBM Telco dataset is a single cross-sectional snapshot -- there are
no event timestamps (no signup date, no churn date), so a genuine
temporal train/validation/test split (as required for time-series-safe
evaluation) is NOT POSSIBLE on this dataset without fabricating dates.
We say so explicitly rather than pretending otherwise.

What we do instead, and disclose in the API (/api/leakage-audit and
/api/models/{id}) and the UI:
  - Stratified random split (train 60% / val 20% / test 20%), stratified
    on the target to preserve class balance in each split.
  - A fixed random_state for reproducibility.
  - The held-out test set is NEVER touched during model selection --
    only the validation set is used to pick the production model; test
    metrics are computed once, at the end, for honest reporting.

If a temporal dataset is plugged in later (e.g. KKBox has real dates),
this module's `temporal_split()` function should be used instead.
"""
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def stratified_split(X: pd.DataFrame, y: pd.Series, ids: pd.Series):
    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        X, y, ids, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test, id_val, id_test = train_test_split(
        X_temp, y_temp, id_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )
    return {
        "train": (X_train, y_train, id_train),
        "val": (X_val, y_val, id_val),
        "test": (X_test, y_test, id_test),
    }


def temporal_split(df: pd.DataFrame, timestamp_col: str, target_col: str):
    """Only usable when the dataset actually has a timestamp column.
    Not used for the IBM Telco dataset (see module docstring)."""
    if timestamp_col not in df.columns:
        raise ValueError(
            f"No '{timestamp_col}' column present -- this dataset does not "
            "support a temporal split. Use stratified_split() instead."
        )
    df_sorted = df.sort_values(timestamp_col)
    n = len(df_sorted)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    return {
        "train": df_sorted.iloc[:train_end],
        "val": df_sorted.iloc[train_end:val_end],
        "test": df_sorted.iloc[val_end:],
    }

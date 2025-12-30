# src/risk_model.py

import pandas as pd
import numpy as np

# -----------------------------
# SIMPLE RISK MAP (EDIT LATER)
# -----------------------------
RISK_MAP = {
    "strawberry": 2,
    "vanilla": 1,
    "chocolate": 1,
    "mixed berry": 3,
    "unknown": 4,
}

# -----------------------------
# TEXT NORMALISATION
# -----------------------------
def norm_text(val):
    if pd.isna(val):
        return ""
    return str(val).strip().lower()

# -----------------------------
# FLAVOUR INFERENCE
# -----------------------------
def infer_flavour_label(row):
    name = norm_text(row.get("product_name", ""))
    flavour = norm_text(row.get("flavour_name", ""))

    for key in RISK_MAP:
        if key in name or key in flavour:
            return key

    return "unknown"

# -----------------------------
# ROW-LEVEL RISK SCORING
# -----------------------------
def _score_row(row):
    flavour = infer_flavour_label(row)
    base_risk = RISK_MAP.get(flavour, 4)

    pack_size = row.get("pack_size_g", np.nan)

    # simple rule: bigger packs = higher risk
    if not pd.isna(pack_size) and pack_size >= 500:
        base_risk += 1

    return min(base_risk, 5)

# -----------------------------
# ✅ REQUIRED FUNCTION (THIS WAS MISSING)
# -----------------------------
def score_risk(df: pd.DataFrame) -> pd.Series:
    """
    Returns a risk score per row (1–5)
    """
    return df.apply(_score_row, axis=1)

# src/risk_model.py
from __future__ import annotations

import re
from typing import Tuple, List

import numpy as np
import pandas as pd


# -----------------------------
# Text helpers
# -----------------------------
def norm_text(x) -> str:
    """Lowercase, trim, remove extra spaces. Safe for None/NA."""
    if x is None or x is pd.NA:
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = norm_text(text)
    return any(k in t for k in keywords)


# -----------------------------
# Risk scoring
# -----------------------------
DEFAULT_PH_MIN = 3.6
DEFAULT_PH_MAX = 4.9


def _to_float(x) -> float:
    """Convert to float safely; return np.nan if missing/invalid."""
    if x is None or x is pd.NA:
        return np.nan
    try:
        # pandas may store numbers as strings
        return float(x)
    except Exception:
        return np.nan


def _score_row(row: pd.Series,
               ph_min: float = DEFAULT_PH_MIN,
               ph_max: float = DEFAULT_PH_MAX) -> Tuple[float, str]:
    """
    Always returns (risk_score, risk_reason).

    Score meaning:
      0.0 - 0.49  -> Low
      0.5 - 0.99  -> Medium
      1.0+        -> High
    """
    score = 0.0
    reasons: List[str] = []

    product_name = row.get("product_name", "")
    flavour_label = row.get("flavour_label", "")
    machine = row.get("machine", "")

    p_name = norm_text(product_name)
    f_lab = norm_text(flavour_label)
    m_txt = norm_text(machine)

    # ---- 1) pH checks (only if pH exists)
    # If pH column missing, we don't penalize hard; if present but missing, small risk.
    ph_val = _to_float(row.get("ph", np.nan))

    if "ph" in row.index:
        if np.isnan(ph_val):
            score += 0.25
            reasons.append("pH missing")
        else:
            if ph_val < ph_min or ph_val > ph_max:
                score += 1.2
                reasons.append(f"pH out of range ({ph_val:.2f}; spec {ph_min}-{ph_max})")
            elif (ph_val < (ph_min + 0.1)) or (ph_val > (ph_max - 0.1)):
                score += 0.35
                reasons.append(f"pH near limit ({ph_val:.2f})")

    # ---- 2) “Complex formulation” heuristic (granola, white choc, tophat, etc.)
    # You can expand these keyword lists anytime.
    complex_keywords = [
        "granola",
        "white choc",
        "white chocolate",
        "tophat",
        "top hat",
        "layered",
        "pieces",
        "bits",
    ]
    if _contains_any(p_name, complex_keywords) or _contains_any(f_lab, complex_keywords):
        score += 0.40
        reasons.append("complex formulation")

    # ---- 3) Pack size missing / unusual (if column exists)
    if "pack_size_g" in row.index:
        pack = _to_float(row.get("pack_size_g", np.nan))
        if np.isnan(pack) or pack <= 0:
            score += 0.20
            reasons.append("pack size missing/invalid")
        else:
            # not a strict rule; just a light flag if very unusual
            if pack < 80 or pack > 12000:
                score += 0.20
                reasons.append(f"unusual pack size ({int(pack)}g)")

    # ---- 4) Machine missing (if you expect machine assignment downstream)
    if "machine" in row.index:
        if norm_text(machine) == "":
            score += 0.15
            reasons.append("machine not assigned")

    # ---- 5) Product name missing (should not happen after normalize, but safe)
    if norm_text(product_name) == "":
        score += 0.60
        reasons.append("product name missing")

    # Build reason text
    reason_text = "; ".join(reasons).strip()

    return float(score), reason_text


def score_risk(df: pd.DataFrame,
               ph_min: float = DEFAULT_PH_MIN,
               ph_max: float = DEFAULT_PH_MAX) -> pd.Series:
    """
    Adds:
      - risk_final (float)
      - risk_reason (string)
      - risk_band (Low/Medium/High)

    Returns:
      df["risk_final"]  (Series)
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("score_risk expected a pandas DataFrame")

    # Apply row scoring
    scored = df.apply(lambda r: _score_row(r, ph_min=ph_min, ph_max=ph_max), axis=1)

    # scored is a Series of tuples -> split safely
    df["risk_final"] = scored.apply(lambda x: float(x[0]) if isinstance(x, (tuple, list)) else _to_float(x))
    df["risk_reason"] = scored.apply(lambda x: str(x[1]) if isinstance(x, (tuple, list)) and len(x) > 1 else "")

    # Banding
    def _band(x):
        x = _to_float(x)
        if np.isnan(x):
            return "Unknown"
        if x >= 1.0:
            return "High"
        if x >= 0.5:
            return "Medium"
        return "Low"

    df["risk_band"] = df["risk_final"].apply(_band)

    return df["risk_final"]

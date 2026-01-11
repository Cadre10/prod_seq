# src/normalize.py
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Text helpers
# -----------------------------
def norm_text(x: object) -> str:
    """Lowercase, strip, collapse whitespace; safe for NaN/None."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Find a column whose normalized name matches any candidate exactly
    or contains the candidate token.
    """
    norm_cols = {norm_text(c): c for c in df.columns}

    # exact match first
    for cand in candidates:
        c_norm = norm_text(cand)
        if c_norm in norm_cols:
            return norm_cols[c_norm]

    # contains match
    for cand in candidates:
        token = norm_text(cand)
        for nc, original in norm_cols.items():
            if token and token in nc:
                return original

    return None


# -----------------------------
# Product parsing
# -----------------------------
def extract_pack_size_g(product_name: str) -> float:
    """
    Returns pack size in grams if found (e.g., 150g, 450 g, 2kg, 10kg)
    else NaN.
    """
    s = norm_text(product_name)

    # grams: "150g" or "150 g"
    m = re.search(r"(\d+(?:\.\d+)?)\s*g\b", s)
    if m:
        return float(m.group(1))

    # kilograms: "2kg" or "2 kg"
    m = re.search(r"(\d+(?:\.\d+)?)\s*kg\b", s)
    if m:
        return float(m.group(1)) * 1000.0

    return float("nan")


def infer_flavour_label(product_name: str, flavour_name: str = "") -> str:
    """
    Heuristic flavour detection from product_name/flavour_name.
    Returns a simple label like: plain, vanilla, strawberry, blueberry, honey, raspberry, mango, etc.
    """
    s = f"{norm_text(product_name)} {norm_text(flavour_name)}"

    buckets: List[Tuple[str, List[str]]] = [
        ("plain", ["plain", "natural", "greek"]),
        ("vanilla", ["vanilla"]),
        ("honey", ["honey"]),
        ("strawberry", ["strawberry"]),
        ("blueberry", ["blueberry"]),
        ("raspberry", ["raspberry"]),
        ("mango", ["mango"]),
        ("mandarin_lime", ["mandarin", "lime"]),
        ("toffee", ["toffee"]),
        ("apple_cinnamon", ["apple", "cinamon", "cinnamon"]),
        ("white_choc", ["white choc", "whitechoc", "white chocolate"]),
        ("granola", ["granola"]),
        ("tophat", ["tophat"]),
    ]

    for label, keys in buckets:
        if all(k in s for k in keys):
            return label

    # fallback: unknown
    return ""


def is_plain_yoghurt(product_name: str, flavour_label: str = "") -> bool:
    """
    Plain yoghurt = natural/greek/plain AND not obviously flavoured.
    """
    s = f"{norm_text(product_name)} {norm_text(flavour_label)}"
    is_plainish = any(k in s for k in ["plain", "natural", "greek"])
    is_flavoured = any(
        k in s
        for k in [
            "vanilla",
            "strawberry",
            "blueberry",
            "raspberry",
            "honey",
            "mango",
            "mandarin",
            "lime",
            "toffee",
            "choc",
            "granola",
            "tophat",
            "apple",
            "cinamon",
            "cinnamon",
        ]
    )
    return bool(is_plainish and not is_flavoured)


# -----------------------------
# Machine assignment rules (your factory rules)
# -----------------------------
def assign_machine_from_product(product_name: str, pack_size_g: float) -> str:
    """
    Your rules:
    - All SS granola products -> M3 (flavour added on line)
    - All granola products -> M3
    - Buckets 2/5/10 kg -> BUCKET_LINE
    - 450g pots -> M2 (and some 150g also run on M2)
    - M1: 150g and 170/175g
    - You have 3 machines + bucket line

    Note: Since plan doesn't specify machines, we apply a best-effort default.
    """
    pn = norm_text(product_name)

    # Buckets (2kg, 5kg, 10kg) => bucket line
    if ("kg" in pn) or (not np.isnan(pack_size_g) and pack_size_g >= 2000):
        return "BUCKET_LINE"

    # Granola lines (including SS granola)
    if "granola" in pn:
        return "M3"

    # 450g pots only on M2
    if not np.isnan(pack_size_g) and 440 <= pack_size_g <= 460:
        return "M2"

    # some 150g on M2; we need a rule. Use "c." or "crown" as default to M1 unless specified.
    # If you later add explicit mapping, replace this section.
    if not np.isnan(pack_size_g) and 145 <= pack_size_g <= 155:
        # Default: M1, but allow specific items to go M2
        # Example heuristics: "c.vanilla 450g" doesn't apply; for 150g, if it's a "C." core cup, keep M1.
        return "M1"

    # 170/175g on M1
    if not np.isnan(pack_size_g) and (165 <= pack_size_g <= 180):
        return "M1"

    # fallback
    return "M1"


# -----------------------------
# Column standardisation
# -----------------------------
def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a consistent set of columns used by the rest of the pipeline.
    Safe if columns are missing.
    """
    df = df.copy()

    # Identify likely input columns
    product_col = _find_column(df, ["product name", "product", "item", "description", "desc"])
    pack_col = _find_column(df, ["pack size (g)", "pack size", "pack_size", "pack", "size (g)", "size"])
    flavour_col = _find_column(df, ["flavour", "flavor", "flavour name", "flavor name"])
    ph_col = _find_column(df, ["ph"])
    batch_vol_col = _find_column(df, ["batch volume", "batch_volume", "total mix", "total mixed", "mix (kg)", "mix kg"])

    # Build normalized columns
    if product_col is not None:
        df["product_name"] = df[product_col].astype(str)
    else:
        # hard fail: agent needs product names
        raise ValueError(f"Missing product column. Available columns: {list(df.columns)}")

    if pack_col is not None:
        # try numeric; if it fails, we'll still parse from product_name later
        df["pack_size_g"] = pd.to_numeric(df[pack_col], errors="coerce")
    else:
        df["pack_size_g"] = np.nan  # will be inferred from product_name

    if flavour_col is not None:
        df["flavour_name"] = df[flavour_col].astype(str)
    else:
        df["flavour_name"] = ""

    # Optional numeric cols (agent can run without them)
    if ph_col is not None:
        df["ph"] = pd.to_numeric(df[ph_col], errors="coerce")
    else:
        df["ph"] = pd.NA

    if batch_vol_col is not None:
        # If mix is in kg, store it as batch_volume_kg; otherwise just keep numeric
        df["batch_volume_kg"] = pd.to_numeric(df[batch_vol_col], errors="coerce")
    else:
        df["batch_volume_kg"] = pd.NA

    return df


# -----------------------------
# Main normalize function
# -----------------------------
def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end normalization:
    - standardise key columns
    - infer pack_size_g if missing
    - infer flavour_label
    - infer is_plain_yoghurt
    - assign machine if missing
    """
    df = standardise_columns(df)

    # Fill pack size from product_name where missing
    # (only overwrite NaNs)
    inferred_pack = df["product_name"].apply(lambda x: extract_pack_size_g(str(x)))
    df["pack_size_g"] = df["pack_size_g"].where(~df["pack_size_g"].isna(), inferred_pack)

    # flavour label
    df["flavour_label"] = df.apply(
        lambda r: infer_flavour_label(str(r.get("product_name", "")), str(r.get("flavour_name", ""))),
        axis=1,
    )

    # plain yoghurt flag
    df["is_plain_yoghurt"] = df.apply(
        lambda r: is_plain_yoghurt(str(r.get("product_name", "")), str(r.get("flavour_label", ""))),
        axis=1,
    )

    # machine assignment (only if machine column doesn't exist or is empty)
    if "machine" not in df.columns:
        df["machine"] = df.apply(
            lambda r: assign_machine_from_product(str(r.get("product_name", "")), float(r.get("pack_size_g", np.nan))),
            axis=1,
        )
    else:
        # fill blanks only
        df["machine"] = df["machine"].astype(str)
        mask_blank = df["machine"].str.strip().eq("") | df["machine"].str.lower().eq("nan")
        df.loc[mask_blank, "machine"] = df.loc[mask_blank].apply(
            lambda r: assign_machine_from_product(str(r.get("product_name", "")), float(r.get("pack_size_g", np.nan))),
            axis=1,
        )

    return df

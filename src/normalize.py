import re
import pandas as pd
import numpy as np

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize real production planning data
    into agent-ready schema
    """

    # ---- CLEAN COLUMN NAMES ----
    df.columns = [c.strip() for c in df.columns]

    # ---- RENAME REAL COLUMNS TO AGENT STANDARD ----
    df = df.rename(columns={
        "Product name": "product_name",
        "Plan Base Volume (L)": "base_volume_l",
        "Flavor (%)": "flavour_pct",
        "Sugar (%)": "sugar_pct",
        "Flavor mass (kg)": "flavour_mass_kg",
        "Sugar (kg)": "sugar_kg",
        "pH": "ph"
    })

    # ---- DERIVED FEATURES ----

    # Convert volume to pack-size proxy (1L ≈ 1000g yoghurt)
    if "base_volume_l" in df.columns:
        df["pack_size_g"] = df["base_volume_l"] * 1000

    # Safe numeric conversion
    numeric_cols = [
        "base_volume_l",
        "flavour_pct",
        "sugar_pct",
        "flavour_mass_kg",
        "sugar_kg",
        "pack_size_g",
        "ph"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- SAFETY CHECK ----
    if "product_name" not in df.columns:
        raise ValueError("Missing required column: product_name")

    return df


def norm_text(val):
    if pd.isna(val):
        return ''
    return str(val).strip().lower()

def extract_pack_size_g(product_name):
    if pd.isna(product_name):
        return np.nan
    s = str(product_name)
    m = re.search(r'(\d+)\s*g\b', s.lower())
    if m:
        return float(m.group(1))
    return np.nan

def infer_flavour_label(product_name, flavour_name=''):
    s = norm_text(product_name) + ' ' + norm_text(flavour_name)
    buckets = [
        ('plain', ['plain', 'natural', 'greek']),
        ('vanilla', ['vanilla']),
        ('honey', ['honey']),
        ('strawberry', ['strawberry']),
        ('blueberry', ['blueberry']),
        ('raspberry', ['raspberry']),
        ('mango', ['mango']),
        ('apple_cinnamon', ['apple', 'cinamon', 'cinnamon']),
        ('nectarine', ['nectarine']),
        ('chocolate', ['choc', 'chocolate']),
        ('granola', ['granola']),
        ('other', [])
    ]
    for lab, kws in buckets:
        if lab == 'other':
            continue
        if any(k in s for k in kws):
            return lab
    return 'other'

RISK_MAP = {
    'plain': 0,
    'vanilla': 1,
    'honey': 2,
    'nectarine': 3,
    'apple_cinnamon': 3,
    'mango': 4,
    'strawberry': 5,
    'blueberry': 6,
    'granola': 6,
    'raspberry': 7,
    'chocolate': 8,
    'other': 5
}
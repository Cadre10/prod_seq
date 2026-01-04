import re
import pandas as pd
import numpy as np
import pandas as pd

def extract_pack_size_g(product_name: str):
    """Returns pack size in grams if found, else None.
    Handles 150g, 450g, 2kg, 5kg, 10kg.
    """
    if product_name is None:
        return None
    s = str(product_name).lower()

    m_g = re.search(r"(\d+)\s*g\b", s)
    if m_g:
        return int(m_g.group(1))

    m_kg = re.search(r"(\d+)\s*kg\b", s)
    if m_kg:
        return int(m_kg.group(1)) * 1000

    return None

def is_granola(product_name: str) -> bool:
    if product_name is None:
        return False
    return "granola" in str(product_name).lower()

def is_plain_yoghurt(product_name: str) -> bool:
    if product_name is None:
        return False
    s = str(product_name).lower()

    plain_keywords = ["plain", "natural", "greek"]
    flav_keywords = ["strawberry","raspberry","mango","blueberry","honey","vanilla","toffee","choc","berry"]

    if any(k in s for k in flav_keywords):
        return False
    return any(k in s for k in plain_keywords)

def norm_text(val) -> str:
    if val is None:
        return ""
    return (
        str(val)
        .lower()
        .replace(".", " ")
        .replace("_", " ")
        .strip()
    )


def infer_flavour_label(product_name: str) -> str:
    """Very simple label. You can expand keywords anytime."""
    if product_name is None:
        return "unknown"
    s = str(product_name).lower()

    if is_plain_yoghurt(product_name):
        return "plain"

    flavours = [
        "strawberry","raspberry","mango","blueberry","honey","vanilla","toffee","choc","berry","mandarin","lime","nectarine"
    ]
    for f in flavours:
        if f in s:
            return f
    return "flavoured"
def assign_machine(row) -> str:
    """Returns: M1, M2, M3, BUCKET, or UNKNOWN."""
    product = row.get("product_name", "")
    pack = row.get("pack_size_g")

    # Bucket line: 2kg/5kg/10kg
    if pack in (2000, 5000, 10000):
        return "BUCKET"

    # Granola always on M3
    if row.get("is_granola", False):
        return "M3"

    # 450g always on M2
    if pack == 450:
        return "M2"

    # M1 handles only 150g, 170g, 175g
    if pack in (150, 170, 175):
        # some 150g also run on M2 -> handle with an override list
        return "M1"

    return "UNKNOWN"
M2_150G_OVERRIDES = [
    # Put exact product name fragments here that run 150g on M2
    # e.g. "C. Vanilla 150g", "SS Strawberry 150g"
]

def apply_m2_overrides(df: pd.DataFrame) -> pd.DataFrame:
    if not M2_150G_OVERRIDES:
        return df
    mask_150 = df["pack_size_g"].eq(150)
    mask_match = df["product_name"].astype(str).apply(
        lambda x: any(k.lower() in x.lower() for k in M2_150G_OVERRIDES)
    )
    df.loc[mask_150 & mask_match, "machine"] = "M2"
    return df

def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    # strip spaces + lower for matching
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)

    # possible names in your real files
    aliases = {
        "product_name": ["product_name", "product", "product name", "Product name_Plan", "Product name", "Product"],
        "batch_volume_l": ["batch_volume_l", "batch_volume", "base volume (l)", "Plan Base Volume (L)", "Base Volume (L)"],
        "ph": ["ph", "pH", "PH"],
    }

    # build reverse lookup based on lowercase
    lower_map = {c.lower(): c for c in df.columns}

    for standard, options in aliases.items():
        if standard in df.columns:
            continue
        for opt in options:
            key = opt.lower()
            if key in lower_map:
                df = df.rename(columns={lower_map[key]: standard})
                break
    return df
    df = standardise_columns(df)
    required = ["product_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")
    # Clean headers
    df.columns = [c.strip() for c in df.columns]
    df["pack_size_g"] = df["product_name"].apply(extract_pack_size_g)
    df["is_plain_yoghurt"] = df["product_name"].apply(is_plain_yoghurt)
    df["flavour_label"] = df["product_name"].apply(infer_flavour_label)
    df["is_granola"] = df["product_name"].apply(is_granola)
    df["machine"] = df.apply(assign_machine, axis=1)
    df = apply_m2_overrides(df)



    # Helper: find a column by possible names
    def pick_col(options):
        opts = [o.strip().lower() for o in options]
        for c in df.columns:
            if c.strip().lower() in opts:
                return c
        return None

    # --- Find real columns in your plan ---
    product_col = pick_col(["product name", "product_name", "product", "product name " , "product name"])
    vol_col = pick_col([
        "plan base volume (l)", "base volume (l)", "base volume l",
        "base volume", "volume (l)", "volume l", "plan volume (l)"
    ])
    ph_col = pick_col(["ph", "pH", "target ph", "target pH"])

    # --- Rename to agent standard ---
    rename_map = {}
    if product_col: rename_map[product_col] = "product_name"
    if vol_col: rename_map[vol_col] = "batch_volume_l"
    if ph_col: rename_map[ph_col] = "ph"

    df = df.rename(columns=rename_map)

    # --- Required columns ---
    if "product_name" not in df.columns:
        raise ValueError(f"Missing product column. Available columns: {list(df.columns)}")

    if "batch_volume_l" not in df.columns:
        raise ValueError(f"Missing volume column. Available columns: {list(df.columns)}")

    # --- Optional columns (create if missing) ---
    if "ph" not in df.columns:
        df["ph"] = pd.NA  # allow agent to run even if plan doesn't include pH

    # Numeric conversions
    df["batch_volume_l"] = pd.to_numeric(df["batch_volume_l"], errors="coerce")
    df["ph"] = pd.to_numeric(df["ph"], errors="coerce")

    return df
def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
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
def is_plain_yoghurt(product_name: str) -> bool:
    if product_name is None:
        return False
    s = str(product_name).lower().replace(".", " ").replace("_", " ")

    # plain styles (add more if your naming changes)
    plain_keywords = ["plain", "natural", "greek"]

    # anything that clearly indicates flavour
    flav_keywords = ["strawberry", "raspberry", "mango", "blueberry", "honey", "vanilla", "toffee", "choc", "granola", "berry"]

    if any(k in s for k in flav_keywords):
        return False
    return any(k in s for k in plain_keywords)
def infer_flavour_label(product_name: str) -> str:
    if product_name is None:
        return "unknown"
    s = str(product_name).lower().replace(".", " ").replace("_", " ")

    if is_plain_yoghurt(product_name):
        return "plain"

    flavours = [
        "vanilla", "honey", "toffee", "choc",
        "strawberry", "raspberry", "mango", "blueberry", "berry",
        "mandarin", "lime", "nectarine"
    ]
    for f in flavours:
        if f in s:
            return f

    # fallback
    return "flavoured"

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
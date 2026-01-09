import re
import pandas as pd
import numpy as np

#from risk_model import norm_text#
import pandas as pd

def norm_text(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip().lower().replace(".", " ").replace("_", " ")

def detect_product_column(df: pd.DataFrame) -> str | None:
    """
    Try to find the column that contains product description/name.
    Strong heuristic: column name contains 'product' and/or 'name' and/or 'description'.
    """
    cols = list(df.columns)
    low = {c: str(c).strip().lower() for c in cols}

    # best matches first
    priority = [
        lambda s: ("product" in s and "name" in s),
        lambda s: ("product" in s and "desc" in s),
        lambda s: ("description" in s),
        lambda s: ("product" in s),
        lambda s: ("item" in s and "name" in s),
        lambda s: ("item" in s),
    ]

    for rule in priority:
        for c in cols:
            if rule(low[c]):
                return c
    return None

def extract_pack_size_g(product_name: str):
    """Returns pack size in grams if found, else None.
    Handles 150g, 450g, 2kg, 5kg, 10kg.
    """
    mask_blank = df["product_name"].fillna("").astype(str).str.strip().eq("")
    fallback_cols = []
    for c in df.columns:
      lc = c.lower()
    if any(k in lc for k in ["code", "sku", "item", "desc", "description", "pack", "size", "variant", "flavour", "flavor"]):
         fallback_cols.append(c)
    if fallback_cols and mask_blank.any():
    # Build a row-wise joined string safely
        rebuilt = (
    df.loc[mask_blank, fallback_cols]
        .fillna("")
        .astype(str)
        .apply(lambda r: " ".join([x.strip() for x in r.tolist() if str(x).strip() != ""]), axis=1)
    )
    df.loc[mask_blank, "product_name"] = rebuilt
    # Final cleanup and drop still-blank rows
    df["product_name"] = df["product_name"].fillna("").astype(str).str.strip()
    df = df[df["product_name"] != ""].copy()


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
    df["is_plain_yoghurt"] = df.apply(
    lambda r: is_plain_yoghurt(r.get("product_name", ""), r.get("flavour_label", "")),
    axis=1
)

    df["machine"] = df.apply(
    lambda r: assign_machine(r.get("product_name", ""), r.get("pack_size_g", None)),
    axis=1
)

    # Clean headers
    df.columns = [str(c).strip() for c in df.columns]

    # --- Detect product column and create product_name ---
    prod_col = detect_product_column(df)

    if prod_col is not None:
        df["product_name"] = df[prod_col].fillna("").astype(str).str.strip()
    else:
        # If no obvious product column, try to build from multiple likely columns
        # (common in production plans)
        possible_parts = []
        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in ["product", "name", "desc", "flavour", "flavor", "variant", "size", "pack"]):
                possible_parts.append(c)

        if possible_parts:
            df["product_name"] = (
                df[possible_parts].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
            )
        else:
            df["product_name"] = ""

    # Make sure it's not empty spaces
    df["product_name"] = df["product_name"].fillna("").astype(str).str.strip()
# Drop rows with empty product_name (prevents UNKNOWN PRODUCT spam)
    df["product_name"] = df["product_name"].fillna("").astype(str).str.strip()
    df = df[df["product_name"] != ""].copy()

    # DEBUG (keep for now)
    print("DEBUG product column detected:", prod_col)
    print("DEBUG product_name sample:", df["product_name"].head(5).tolist())

    # ... keep the rest of your normalize logic below ...


    def norm_text(val):
        if pd.isna(val):
            return ''
        return str(val).strip().lower()
    return df
    # Guarantee product_name exists (even if the input file uses other headings) if "product_name" not in df.columns:
    # try common alternatives (case-insensitive)
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in ["product", "product name", "product_name_plan", "product name_plan", "product name plan"]:
        if candidate in lower_map:
         df = df.rename(columns={lower_map[candidate]: "product_name"})
        break
# If still missing, create empty so downstream doesn't crash
        if "product_name" not in df.columns:
            df["product_name"] = ""

    df["product_name"] = ""

# Ensure it's filled and clean
    df["product_name"] = df["product_name"].fillna("").astype(str).str.strip()

def extract_pack_size_g(product_name):
    if pd.isna(product_name):
        return np.nan
    s = str(product_name)
    m = re.search(r'(\d+)\s*g\b', s.lower())
    if m:
        return float(m.group(1))
    return np.nan 

def infer_flavour_label(product_name, flavour_name=''):

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

def is_plain_yoghurt(product_name: str, flavour_label: str = "") -> bool:
    s = norm_text(product_name)
    f = norm_text(flavour_label)

    # If flavour label is explicitly non-plain, it is not plain
    if f and f not in ["plain", "natural", "greek"]:
        return False

    plain_keywords = ["plain", "natural", "greek"]
    flav_keywords = [
        "strawberry", "raspberry", "mango", "blueberry",
        "honey", "vanilla", "toffee", "choc",
        "granola", "berry"
    ]

    if any(k in s for k in flav_keywords):
        return False

    return any(k in s for k in plain_keywords)

def assign_machine(product_name: str, pack_size_g: float):
    s = norm_text(product_name)

    # Buckets line (2kg, 5kg, 10kg)
    if pack_size_g in [2000, 5000, 10000] or "kg" in s:
        return "BUCKET_LINE"

    # Granola always on M3
    if "granola" in s:
        return "M3"

    # 450g pots on M2
    if pack_size_g == 450:
        return "M2"

    # M1 does 150g / 170g / 175g
    if pack_size_g in [150, 170, 175]:
        return "M1"

    return "UNKNOWN_LINE"
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
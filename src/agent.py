# src/agent.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd


# ---- Imports from your project ----
from src.io_data import load_input_csv
from src.normalize import normalize_data
from src.risk_model import score_risk
from src.sequencer import propose_sequence, sequence_actions
# Optional: if you have src/sequencer.py with sequence_actions(df) -> list[str] or Series[str]
try:
    from src.sequencer import sequence_actions  # type: ignore
    _HAS_SEQUENCER = True
except Exception:
    sequence_actions = None
    _HAS_SEQUENCER = False


# =========================
# CONFIG
# =========================
DEFAULT_INPUT = Path("data/input/Prod_Plan_Today.csv")
OUTPUT_DIR = Path("Outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# MACHINE ASSIGNMENT RULES
# =========================
M2_150_OVERRIDES = {
    # Add exact product names that are 150g but run on M2
    # Example:
    # "C.SomeProduct 150g",
}

def assign_machine(row: pd.Series) -> str:
    pn = str(row.get("product_name", "") or "")
    pn_l = pn.lower()
    pack = row.get("pack_size_g", None)

    if pn in M2_150_OVERRIDES:
        return "M2"

    # Buckets line
    if any(x in pn_l for x in ["2kg", "5kg", "10kg"]) or pack in [2000, 5000, 10000]:
        return "BUCKET_LINE"

    # Granola line is M3 (all granola + all SS granola)
    if "granola" in pn_l or pn_l.startswith("ss "):
        return "M3"

    # 450g pots are M2
    if pack == 450 or "450g" in pn_l:
        return "M2"

    # M1 default for 150g, 170g, 175g and anything else unknown
    if pack in [150, 170, 175] or any(x in pn_l for x in ["150g", "170g", "175g"]):
        return "M1"

    return "M1"


# =========================
# CHANGEOVER + WASHDOWN LOGIC
# =========================
def add_changeover_and_washdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rules based on what you told me:
    - Changeover: if product changes on same machine
    - Washdown:
        A) Plain ↔ flavoured switch -> washdown required
        B) Flavour label change -> washdown required
        C) Granola line (M3): washdown required when changing products
    """
    df = df.copy()

    # Ensure columns exist
    if "changeover_required" not in df.columns:
        df["changeover_required"] = False
    if "changeover_reason" not in df.columns:
        df["changeover_reason"] = ""

    if "washdown_required" not in df.columns:
        df["washdown_required"] = False
    if "washdown_reason" not in df.columns:
        df["washdown_reason"] = ""

    # Sort by machine to do per-machine sequencing
    if "machine" in df.columns:
        df = df.sort_values(["machine"], na_position="last").reset_index(drop=True)

    prev = {}  # machine -> (product, flavour, is_plain)

    for i, row in df.iterrows():
        m = str(row.get("machine", "") or "")
        prod = str(row.get("product_name", "") or "")
        flav = str(row.get("flavour_label", "") or "")
        is_plain = bool(row.get("is_plain_yoghurt", False))

        if not m:
            continue

        if m in prev:
            pprod, pflav, pplain = prev[m]

            # Changeover if product changed
            if prod and pprod and prod != pprod:
                df.at[i, "changeover_required"] = True
                df.at[i, "changeover_reason"] = f"Product change on {m}: {pprod} → {prod}"

            # Washdown rules
            reasons = []

            # Granola machine rule
            if m == "M3" and prod and pprod and prod != pprod:
                reasons.append("M3 granola line: washdown required for product change")

            # Plain ↔ flavoured switch
            if pplain != is_plain:
                reasons.append("Plain ↔ flavoured switch")

            # Flavour label change
            if pflav and flav and pflav != flav:
                reasons.append(f"Flavour change: {pflav} → {flav}")

            if reasons:
                df.at[i, "washdown_required"] = True
                existing = str(df.at[i, "washdown_reason"] or "").strip()
                joined = " | ".join(reasons)
                df.at[i, "washdown_reason"] = (existing + " | " + joined).strip(" | ")

        prev[m] = (prod, flav, is_plain)

    return df


# =========================
# PIPELINE
# =========================
def ensure_product_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Your normalize_data SHOULD create product_name.
    This function just does a hard guard + helpful debug.
    """
    if "product_name" in df.columns:
        return df

    # Try to detect a likely product column and copy it
    candidates = [c for c in df.columns if str(c).strip().lower() in ["product name", "product", "item", "sku", "description", "desc"]]
    if candidates:
        df = df.copy()
        df["product_name"] = df[candidates[0]].astype(str)
        print(f"DEBUG: product_name missing; created from '{candidates[0]}'")
        return df

    raise ValueError(f"normalize_data did not create product_name and no fallback found. Columns: {list(df.columns)}")


def ensure_machine(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "machine" not in df.columns:
        df["machine"] = ""

    df["machine"] = df["machine"].fillna("").astype(str)
    mask_blank = df["machine"].str.strip().eq("")
    if mask_blank.any():
        df.loc[mask_blank, "machine"] = df.loc[mask_blank].apply(assign_machine, axis=1)
    return df


def ensure_action(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if _HAS_SEQUENCER and callable(sequence_actions):
        try:
            actions = sequence_actions(df)
            df["action"] = actions
            return df
        except Exception as e:
            print(f"⚠️ sequence_actions failed, using fallback actions. Reason: {e}")

    # Fallback action
    if "risk_final" in df.columns:
        df["action"] = df["risk_final"].astype(str).apply(
            lambda x: "HOLD / CHECK" if "HIGH" in x.upper() else "RELEASE"
        )
    else:
        df["action"] = "RELEASE"
    return df

def main() -> None:
    print("🧠 Yoghurt AI Agent starting...")

    # 1) Load data
    df = load_input_csv()
    print("✅ Data loaded")
    print(f"📥 Loaded {len(df)} rows")

    # 2) Normalize
    df = normalize_data(df)
    df = ensure_product_name(df)
    print("✅ Data normalised")

    # 3) Assign machines
    df = ensure_machine(df)

    # 4) Risk scoring
    df["risk_final"] = score_risk(df)
    print("⚠️ Risk scored")

# Proposed optimal sequence per machine
    seq = propose_sequence(df)

# Create actions on the sequenced table
    seq["action"] = sequence_actions(seq)

# Save/print using seq instead of df
    df_out = seq.copy()

    # 5) Changeover + washdown logic
    df = add_changeover_and_washdown(df)

    # 6) Actions (use sequencer if available, else fallback)
    df = ensure_action(df)

    # 7) Select output columns safely
    preferred_cols = [
        "machine",
        "product_name",
        "pack_size_g",
        "flavour_label",
        "is_plain_yoghurt",
        "risk_final",
        "risk_reason",
        "changeover_required",
        "changeover_reason",
        "washdown_required",
        "washdown_reason",
        "action",
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    df_out = df[cols].copy()

    # 8) Console summary (clean, readable)
    print("\n📋 AGENT OUTPUT:")
    for _, row in df_out.iterrows():
        prod = str(row.get("product_name", "") or "").strip() or "UNKNOWN PRODUCT"
        risk = str(row.get("risk_final", "") or "").strip()

        flags = []
        if bool(row.get("changeover_required", False)):
            flags.append("CHANGEOVER")
        if bool(row.get("washdown_required", False)):
            flags.append("WASHDOWN")

        flag_txt = f" [{' + '.join(flags)}]" if flags else ""
        print(f"✅ RELEASE: {prod}{flag_txt} | {risk}")

    # 9) Save output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"agent_decisions_{ts}.csv"
    df_out.to_csv(out_path, index=False)

    print(f"\n📁 Decisions saved to: {out_path}")



if __name__ == "__main__":
    main()



# dashboard.py
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# --- Import your pipeline code ---
# These must exist in your project:
# src/normalize.py -> normalize_data(df)
# src/risk_model.py -> score_risk(df)
try:
    from src.normalize import normalize_data
    from src.risk_model import score_risk
except Exception as e:
    st.error(f"Failed to import your src modules: {e}")
    st.stop()


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
    # Put exact product names here if some 150g runs on M2
    # Example:
    # "C.SomeProduct 150g",
}

def assign_machine(row: pd.Series) -> str:
    pn = str(row.get("product_name", "") or "")
    pn_l = pn.lower()
    pack = row.get("pack_size_g", None)

    # override
    if pn in M2_150_OVERRIDES:
        return "M2"

    # Buckets line
    if any(x in pn_l for x in ["2kg", "5kg", "10kg"]) or pack in [2000, 5000, 10000]:
        return "BUCKET_LINE"

    # Granola line (M3)
    # - all granola products
    # - all SS granola
    if "granola" in pn_l or pn_l.startswith("ss "):
        return "M3"

    # 450g pots -> M2
    if pack == 450 or "450g" in pn_l:
        return "M2"

    # M1: 150g, 170g, 175g (default)
    if pack in [150, 170, 175] or any(x in pn_l for x in ["150g", "170g", "175g"]):
        return "M1"

    return "M1"


# =========================
# CHANGEOVER + WASHDOWN
# =========================
def add_changeover_and_washdown(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in ["changeover_required", "washdown_required"]:
        if c not in df.columns:
            df[c] = False
    for c in ["changeover_reason", "washdown_reason"]:
        if c not in df.columns:
            df[c] = ""

    # Sort by machine first (sequence logic is per-machine).
    # If you later add a "sequence" column, include it here.
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

            # Washdown rules:
            # A) M3 (granola line): washdown required when changing products
            if m == "M3" and prod and pprod and prod != pprod:
                df.at[i, "washdown_required"] = True
                df.at[i, "washdown_reason"] = "M3 granola line: washdown required for product change"

            # B) Plain ↔ flavoured switch
            if pplain != is_plain:
                df.at[i, "washdown_required"] = True
                reason = "Plain ↔ flavoured switch"
                df.at[i, "washdown_reason"] = (str(df.at[i, "washdown_reason"]) + " | " + reason).strip(" |")

            # C) Flavour label change (if both are non-empty)
            if pflav and flav and pflav != flav:
                df.at[i, "washdown_required"] = True
                reason = f"Flavour change: {pflav} → {flav}"
                df.at[i, "washdown_reason"] = (str(df.at[i, "washdown_reason"]) + " | " + reason).strip(" |")

        prev[m] = (prod, flav, is_plain)

    return df


# =========================
# HELPERS
# =========================
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)

def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_data(df)

    # Ensure product_name exists
    if "product_name" not in df.columns:
        raise ValueError(f"normalize_data did not create product_name. Columns: {list(df.columns)}")

    # Auto-assign machine if missing or blank
    if "machine" not in df.columns:
        df["machine"] = ""
    df["machine"] = df["machine"].fillna("").astype(str)
    mask_blank_machine = df["machine"].str.strip().eq("")
    if mask_blank_machine.any():
        df.loc[mask_blank_machine, "machine"] = df.loc[mask_blank_machine].apply(assign_machine, axis=1)

    # Risk score
    df["risk_final"] = score_risk(df)

    # Washdown/changeover
    df = add_changeover_and_washdown(df)

    # Basic action column (if you already have src.sequencer, you can replace this)
    if "action" not in df.columns:
        # Example simple action from risk:
        df["action"] = df["risk_final"].astype(str).apply(lambda x: "HOLD / CHECK" if "HIGH" in x.upper() else "RELEASE")

    return df


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Yoghurt QA Agent Dashboard", layout="wide")

st.title("🧠 Yoghurt QA Agent Dashboard")
st.caption("Upload / load plan → normalize → risk score → machine assignment → washdown/changeover decisions")

# Sidebar inputs
st.sidebar.header("Inputs")

use_upload = st.sidebar.checkbox("Upload plan CSV instead of using default")
uploaded = None
if use_upload:
    uploaded = st.sidebar.file_uploader("Upload production plan CSV", type=["csv"])

default_path = st.sidebar.text_input("Default input path", str(DEFAULT_INPUT))
default_path = Path(default_path)

run_btn = st.sidebar.button("▶ Run Agent")

# Filters
st.sidebar.divider()
st.sidebar.header("Filters")

only_washdown = st.sidebar.checkbox("Show only washdown required")
only_changeover = st.sidebar.checkbox("Show only changeover required")
search_text = st.sidebar.text_input("Search product name contains", "")

# Main
if run_btn:
    try:
        if uploaded is not None:
            df_in = pd.read_csv(uploaded)
            st.success("✅ Loaded uploaded CSV")
        else:
            df_in = load_csv(default_path)
            st.success(f"✅ Loaded default CSV: {default_path}")

        df = run_pipeline(df_in)

        # Apply filters
        view = df.copy()

        if "machine" in view.columns:
            machines = sorted([m for m in view["machine"].dropna().unique().tolist() if str(m).strip() != ""])
            chosen_machines = st.sidebar.multiselect("Machines", machines, default=machines)
            if chosen_machines:
                view = view[view["machine"].isin(chosen_machines)]

        if only_washdown and "washdown_required" in view.columns:
            view = view[view["washdown_required"] == True]

        if only_changeover and "changeover_required" in view.columns:
            view = view[view["changeover_required"] == True]

        if search_text.strip():
            s = search_text.strip().lower()
            view = view[view["product_name"].astype(str).str.lower().str.contains(s, na=False)]

        # Summary
        st.subheader("📌 Summary by machine")
        if "machine" in view.columns:
            summary = view.groupby("machine", dropna=False).agg(
                runs=("product_name", "count"),
                washdowns=("washdown_required", "sum"),
                changeovers=("changeover_required", "sum"),
            ).reset_index()
            st.dataframe(summary, use_container_width=True)
        else:
            st.info("No machine column found.")

        # Table
        st.subheader("📋 Decisions table")

        SHOW_COLS = [
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
        cols = [c for c in SHOW_COLS if c in view.columns]
        st.dataframe(view[cols], use_container_width=True)

        # Save + Download
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"agent_decisions_{ts}.csv"
        view[cols].to_csv(out_path, index=False)

        st.success(f"✅ Saved decisions to: {out_path}")

        st.download_button(
            "⬇ Download decisions CSV",
            data=view[cols].to_csv(index=False).encode("utf-8"),
            file_name=out_path.name,
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("Click **Run Agent** in the sidebar to generate today's decisions.")
    st.write("Expected default input file:")
    st.code(str(DEFAULT_INPUT))

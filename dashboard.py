import os
import glob
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Yoghurt AI Agent Dashboard", layout="wide")

st.title("🧠 Yoghurt AI Agent Dashboard")
st.caption("Shows sequenced plan + washdown/changeover + risks (auto machine assignment supported).")

# ----------------------------
# Helpers
# ----------------------------
def find_latest_output(pattern="Outputs/agent_decisions_*.csv"):
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def safe_bool_series(s):
    # converts True/False, "TRUE"/"FALSE", 1/0, blanks → False
    if s is None:
        return None
    if s.dtype == bool:
        return s.fillna(False)
    return s.fillna(False).astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])

def ensure_cols(df):
    # Create missing columns so dashboard never breaks
    defaults = {
        "machine": "",
        "sequence": None,
        "product_name": "",
        "pack_size_g": None,
        "flavour_label": "",
        "is_plain_yoghurt": False,
        "risk_final": "",
        "risk_reason": "",
        "changeover_required": False,
        "changeover_reason": "",
        "washdown_required": False,
        "washdown_reason": "",
        "action": "",
    }
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v

    # normalize booleans
    df["washdown_required"] = safe_bool_series(df["washdown_required"])
    df["changeover_required"] = safe_bool_series(df["changeover_required"])
    df["is_plain_yoghurt"] = safe_bool_series(df["is_plain_yoghurt"])

    # clean strings
    for c in ["machine", "product_name", "flavour_label", "risk_final", "risk_reason",
              "changeover_reason", "washdown_reason", "action"]:
        df[c] = df[c].fillna("").astype(str)

    # numeric
    for c in ["pack_size_g", "sequence"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def compute_summary(df):
    total = len(df)
    wash = int(df["washdown_required"].sum())
    chg = int(df["changeover_required"].sum())
    high_risk = int((df["risk_final"].str.lower().isin(["high", "critical"])).sum())
    unknown_machine = int((df["machine"].str.strip() == "").sum())

    return {
        "Total rows": total,
        "Washdowns": wash,
        "Changeovers": chg,
        "High/Critical risk": high_risk,
        "Missing machine": unknown_machine
    }

# ----------------------------
# Load Data
# ----------------------------
with st.sidebar:
    st.header("📥 Data source")

    latest = find_latest_output()
    st.write("**Latest output found:**")
    st.code(latest if latest else "None found in Outputs/")

    uploaded = st.file_uploader("Upload a CSV (optional)", type=["csv"])

    use_latest = st.checkbox("Use latest Outputs/agent_decisions_*.csv", value=True)

    if uploaded is None and not use_latest:
        st.warning("Choose latest output or upload a CSV.")
        st.stop()

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    if latest is None:
        st.error("No Outputs/agent_decisions_*.csv found. Run: python -m src.agent")
        st.stop()
    df = pd.read_csv(latest)

df = ensure_cols(df)

# ----------------------------
# Filters
# ----------------------------
with st.sidebar:
    st.header("🔎 Filters")

    machines = sorted([m for m in df["machine"].unique() if str(m).strip() != ""])
    machine_sel = st.multiselect("Machine", machines, default=machines)

    show_only_wash = st.checkbox("Only rows requiring washdown", value=False)
    show_only_chg = st.checkbox("Only rows requiring changeover", value=False)

    risk_vals = sorted([r for r in df["risk_final"].unique() if r.strip() != ""])
    risk_sel = st.multiselect("Risk level", risk_vals, default=risk_vals)

view = df.copy()

if machine_sel:
    view = view[view["machine"].isin(machine_sel)]

if risk_sel:
    view = view[view["risk_final"].isin(risk_sel)]

if show_only_wash:
    view = view[view["washdown_required"] == True]

if show_only_chg:
    view = view[view["changeover_required"] == True]

# Sort nicely
sort_cols = []
if "machine" in view.columns: sort_cols.append("machine")
if "sequence" in view.columns: sort_cols.append("sequence")
if sort_cols:
    view = view.sort_values(sort_cols, na_position="last")

# ----------------------------
# Summary KPIs
# ----------------------------
st.subheader("📊 Summary")
summary = compute_summary(view)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total", summary["Total rows"])
c2.metric("Washdowns", summary["Washdowns"])
c3.metric("Changeovers", summary["Changeovers"])
c4.metric("High/Critical risk", summary["High/Critical risk"])
c5.metric("Missing machine", summary["Missing machine"])

# ----------------------------
# Highlight tables
# ----------------------------
st.subheader("🚨 Attention list (Washdown / Changeover / High Risk)")

attention = view.copy()
attention["risk_flag"] = attention["risk_final"].str.lower().isin(["high", "critical"])
attention = attention[
    (attention["washdown_required"]) |
    (attention["changeover_required"]) |
    (attention["risk_flag"])
].copy()

attention_cols = [
    "machine", "sequence", "product_name", "pack_size_g", "flavour_label",
    "risk_final", "risk_reason",
    "changeover_required", "changeover_reason",
    "washdown_required", "washdown_reason",
    "action"
]
attention = attention[[c for c in attention_cols if c in attention.columns]]

st.dataframe(attention, use_container_width=True, height=320)

# ----------------------------
# Full plan
# ----------------------------
st.subheader("📋 Full sequenced plan (filtered)")

plan_cols = [
    "machine", "sequence", "product_name", "pack_size_g", "flavour_label",
    "is_plain_yoghurt", "risk_final",
    "changeover_required", "washdown_required",
    "action"
]
plan = view[[c for c in plan_cols if c in view.columns]].copy()
st.dataframe(plan, use_container_width=True, height=420)

# ----------------------------
# Download
# ----------------------------
st.subheader("⬇️ Download")
csv_bytes = view.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered CSV",
    data=csv_bytes,
    file_name="agent_dashboard_view.csv",
    mime="text/csv"
)


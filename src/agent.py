# src/agent.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime

import pandas as pd

from src.io_data import load_input_csv
from src.normalize import normalize_data
from src.risk_model import score_risk

# NEW sequencer functions you pasted
from src.sequencer import assign_machine, sequence_machine


DEFAULT_INPUT = Path("data/input/Prod_Plan_Today.csv")
OUTPUT_DIR = Path("Outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def _ensure_plan_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataframe has the minimum columns required by the sequencer:
      - Product name
      - Pack size (g)

    Your normalize_data() creates product_name/pack_size_g, but sequencer.py
    expects the original plan headers.
    To keep everything consistent, we create/align both.
    """
    df = df.copy()

    # Ensure original plan headers exist (for sequencer.py)
    if "Product name" not in df.columns and "product_name" in df.columns:
        df["Product name"] = df["product_name"].astype(str)

    if "Pack size (g)" not in df.columns:
        # try derive from pack_size_g
        if "pack_size_g" in df.columns:
            df["Pack size (g)"] = pd.to_numeric(df["pack_size_g"], errors="coerce")
        else:
            df["Pack size (g)"] = pd.NA

    # Ensure normalized columns exist too (for risk model / dashboard friendliness)
    if "product_name" not in df.columns and "Product name" in df.columns:
        df["product_name"] = df["Product name"].astype(str)

    if "pack_size_g" not in df.columns and "Pack size (g)" in df.columns:
        df["pack_size_g"] = pd.to_numeric(df["Pack size (g)"], errors="coerce")

    return df


def _sequence_all_machines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Machines run in parallel, so we sequence within each machine separately.
    """
    df = df.copy()

    # Assign machines from your hard rules
    df["machine"] = df.apply(assign_machine, axis=1)

    # Sequence per machine
    sequenced_parts = []
    for m in ["M1", "M2", "M3", "BUCKET"]:
        part = df[df["machine"] == m].copy()
        if len(part) == 0:
            continue
        part_seq = sequence_machine(part)  # adds sequence, washdown_required, changeover_required, downtime_min
        part_seq["machine"] = m
        sequenced_parts.append(part_seq)

    # Any unexpected machines go last (rare)
    unknown = df[~df["machine"].isin(["M1", "M2", "M3", "BUCKET"])].copy()
    if len(unknown) > 0:
        unknown["sequence"] = range(1, len(unknown) + 1)
        unknown["washdown_required"] = False
        unknown["changeover_required"] = False
        unknown["downtime_min"] = 0
        sequenced_parts.append(unknown)

    if not sequenced_parts:
        return df

    out = pd.concat(sequenced_parts, ignore_index=True)

    # Sort nicely for output
    if "sequence" in out.columns:
        out = out.sort_values(["machine", "sequence"], na_position="last").reset_index(drop=True)
    else:
        out = out.sort_values(["machine"], na_position="last").reset_index(drop=True)

    return out


def _add_reasons_and_actions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add human-readable reasons and an 'action' column for operators.
    """
    df = df.copy()

    # Create reason fields if not present
    if "washdown_reason" not in df.columns:
        df["washdown_reason"] = ""
    if "changeover_reason" not in df.columns:
        df["changeover_reason"] = ""

    # Simple reasons based on flags (you can make this more detailed later)
    df.loc[df["washdown_required"] == True, "washdown_reason"] = "Washdown required (max 30 min rule)"
    df.loc[df["changeover_required"] == True, "changeover_reason"] = "Changeover required"

    # Action column
    df["action"] = "RUN"
    df.loc[df["changeover_required"] == True, "action"] = "CHANGEOVER + RUN"
    df.loc[df["washdown_required"] == True, "action"] = "WASHDOWN (≤30min) + RUN"

    return df


def main() -> None:
    print("🧠 Yoghurt AI Agent starting...")

    # 1) Load data
    # If your load_input_csv supports default_path, this will work.
    # If not, it should still load internally from its own default.
    df = load_input_csv()
    print("✅ Data loaded")
    print(f"📥 Loaded {len(df)} rows")

    # 2) Normalize (creates product_name, pack_size_g, flavour_label, is_plain_yoghurt, etc.)
    df = normalize_data(df)
    print("✅ Data normalised")

    # 3) Align columns for sequencer rules
    df = _ensure_plan_columns(df)

    # 4) Risk scoring (adds risk_final/risk_reason/risk_band if your risk_model does)
    score_risk(df)
    print("⚠️ Risk scored")

    # 5) Sequence per machine (parallel lines)
    df_final = _sequence_all_machines(df)

    # 6) Add actions/reasons
    df_final = _add_reasons_and_actions(df_final)

    # 7) Output selection (keeps both plan + normalized columns where helpful)
    preferred_cols = [
        "Date",
        "machine",
        "sequence",
        "Product name",
        "Pack size (g)",
        "product_name",
        "pack_size_g",
        "flavour_label",
        "is_plain_yoghurt",
        "risk_final",
        "risk_reason",
        "risk_band",
        "washdown_required",
        "washdown_reason",
        "changeover_required",
        "changeover_reason",
        "downtime_min",
        "action",
        # keep useful plan columns if present:
        "Plan Base Volume (L)",
        "Flavor (%)",
        "Sugar (%)",
        "Flavor mix (kg)",
        "Sugar (kg)",
        "Plan Total mixd yogh (kg)",
        "Packed no/trays",
        "Packed yogh (kg)",
        "Plan Yield",
        "Recipe check",
    ]
    cols = [c for c in preferred_cols if c in df_final.columns]
    out = df_final[cols].copy()

    # 8) Print quick console preview (first 25 rows)
    print("\n📋 AGENT OUTPUT (preview):")
    preview = out.head(25)
    for _, r in preview.iterrows():
        m = str(r.get("machine", ""))
        seq = r.get("sequence", "")
        pn = str(r.get("Product name", r.get("product_name", "")))
        act = str(r.get("action", ""))
        risk = str(r.get("risk_band", r.get("risk_final", "")))
        wd = "WASH" if bool(r.get("washdown_required", False)) else ""
        ch = "CHG" if bool(r.get("changeover_required", False)) else ""
        flags = " ".join([x for x in [wd, ch] if x]).strip()
        if flags:
            flags = f" [{flags}]"
        print(f"#{seq} {m}: {pn}{flags} -> {act} | {risk}")

    # 9) Save CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"agent_decisions_{ts}.csv"
    out.to_csv(out_path, index=False)
    print(f"\n📁 Decisions saved to: {out_path}")


if __name__ == "__main__":
    main()



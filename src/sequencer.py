# src/sequencer.py
import pandas as pd
from typing import Tuple, List
import pandas as pd

def compress_runs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Use original order if no explicit order column exists
    if "plan_order" not in df.columns:
        df["plan_order"] = range(len(df))

    keys = ["machine", "product_name", "pack_size_g", "flavour_label"]

    df = df.sort_values(["machine", "plan_order"], na_position="last").copy()

    # New run whenever any key changes
    change = (df[keys] != df[keys].shift()).any(axis=1)
    df["_run_id"] = change.cumsum()

    # Aggregate quantities if present
    qty_cols = [c for c in ["Packed no/trays", "Packed yogh (kg)", "Plan Total mixd yog (kg)"] if c in df.columns]

    agg = {
        "machine": "first",
        "product_name": "first",
        "pack_size_g": "first",
        "flavour_label": "first",
        "is_plain_yoghurt": "first",

        # worst-case flags inside the run
        "risk_final": "max",
        "risk_reason": "first",
        "changeover_required": "max",
        "changeover_reason": "first",
        "washdown_required": "max",
        "washdown_reason": "first",
        "action": "first",
        "plan_order": "first",
    }

    for c in qty_cols:
        agg[c] = "sum"

    out = df.groupby("_run_id", as_index=False).agg(agg)
    out = out.sort_values(["machine", "plan_order"], na_position="last")
    return out.drop(columns=["plan_order"])

print("✅ sequencer.py LOADED")
def needs_washdown(prev_row, curr_row) -> Tuple[bool, str]:
    """A + B rules + SS granola nuance."""
    prev_flav = str(prev_row.get("flavour_label", "unknown"))
    curr_flav = str(curr_row.get("flavour_label", "unknown"))

    prev_plain = bool(prev_row.get("is_plain_yoghurt", False))
    curr_plain = bool(curr_row.get("is_plain_yoghurt", False))

    # A) flavoured -> plain
    if (prev_plain is False) and (curr_plain is True):
        return True, f"Washdown: flavoured → plain ({prev_flav} → plain)"

    # B) any flavour change (including plain->flavoured and flavoured->flavoured)
    if prev_flav != curr_flav:
        # SS granola nuance: flavour is dosed on-line
        if bool(curr_row.get("is_granola", False)) or bool(prev_row.get("is_granola", False)):
            return True, f"Washdown/flush + verify doser: granola flavour change ({prev_flav} → {curr_flav})"
        return True, f"Washdown: flavour change ({prev_flav} → {curr_flav})"

    return False, ""
def sequence_actions(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Returns list of action strings and adds changeover/washdown columns to df."""
    actions = []
    changeover_required = []
    changeover_reason = []
    washdown_required = []
    washdown_reason = []

    # Sort by machine first (since they run separately), then risk
    df_sorted = df.copy()
    if "risk_final" in df_sorted.columns:
        df_sorted = df_sorted.sort_values(by=["machine", "risk_final"], ascending=[True, False])
    else:
        df_sorted = df_sorted.sort_values(by=["machine"])

    prev_by_machine = {}

    for idx, row in df_sorted.iterrows():
        product = row.get("product_name", "UNKNOWN PRODUCT")
        machine = row.get("machine", "UNKNOWN")
        risk = row.get("risk_final", 0)

        # ---- Your action rules (keep simple) ----
        if risk >= 5:
            action = f"🚨 STOP LINE ({machine}): {product}"
        elif risk >= 4:
            action = f"⚠️ HOLD BATCH ({machine}): {product}"
        elif risk >= 3:
            action = f"🔍 INCREASE MONITORING ({machine}): {product}"
        else:
            action = f"✅ RELEASE ({machine}): {product}"

        # ---- Changeover/Washdown logic per machine ----
        prev = prev_by_machine.get(machine)

        co = False
        co_reason = ""
        wd = False
        wd_reason = ""

        if prev is not None:
            prev_product = prev.get("product_name", "")
            if str(prev_product) != str(product):
                co = True
                co_reason = f"Changeover on {machine}: '{prev_product}' → '{product}'"

                wd, wd_reason = needs_washdown(prev, row)

        changeover_required.append(co)
        changeover_reason.append(co_reason)
        washdown_required.append(wd)
        washdown_reason.append(wd_reason)

        actions.append(action)
        prev_by_machine[machine] = row

    # Write back to df in the original row order
    # (We created lists in df_sorted order, so we assign using df_sorted index)
    df.loc[df_sorted.index, "changeover_required"] = changeover_required
    df.loc[df_sorted.index, "changeover_reason"] = changeover_reason
    df.loc[df_sorted.index, "washdown_required"] = washdown_required
    df.loc[df_sorted.index, "washdown_reason"] = washdown_reason

    return actions

def sequence_actions(df: pd.DataFrame):
    """
    Convert risk scores into ordered operator actions
    """
    actions = []

    for _, row in df.iterrows():
        risk = row.get("risk_final", 0)
        product = (row.get("product_" \
    "name") or "").strip() or "UNKNOWN PRODUCT"

        if risk >= 5:
            actions.append(
                f"🚨 STOP LINE: {product} – Critical risk detected"
            )
        elif risk == 4:
            actions.append(
                f"⚠️ HOLD BATCH: {product} – QA review required"
            )
        elif risk == 3:
            actions.append(
                f"🔍 INCREASE MONITORING: {product}"
            )
        else:
            actions.append(
                f"✅ RELEASE: {product}"
            )

    return actions

def add_changeover_and_washdown(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["changeover_required"] = False
    df["changeover_reason"] = ""
    df["washdown_required"] = False
    df["washdown_reason"] = ""

    # sort by machine, keep plan order if you have a "seq" column; else keep as-is
    if "machine" in df.columns:
        df = df.sort_values(["machine"], na_position="last")

    prev = {}  # machine -> previous row info

    for i, row in df.iterrows():
        m = str(row.get("machine", "") or "")
        prod = str(row.get("product_name", "") or "")
        flav = str(row.get("flavour_label", "") or "")
        is_plain = bool(row.get("is_plain_yoghurt", False))

        if not m:
            continue

        if m in prev:
            pprod, pflav, pplain = prev[m]

            # Changeover when product changes
            if prod and pprod and prod != pprod:
                df.at[i, "changeover_required"] = True
                df.at[i, "changeover_reason"] = f"Product change on {m}: {pprod} → {prod}"

            # Washdown rules
            # Rule A: Granola line (M3): washdown when changing products
            if m == "M3" and prod and pprod and prod != pprod:
                df.at[i, "washdown_required"] = True
                df.at[i, "washdown_reason"] = f"M3 granola line: washdown required for product change"

            # Rule B: plain ↔ flavoured change
            if pplain != is_plain:
                df.at[i, "washdown_required"] = True
                reason = "Plain ↔ flavoured switch"
                df.at[i, "washdown_reason"] = (df.at[i, "washdown_reason"] + " | " + reason).strip(" |")

            # Rule C: flavour label changed (non-empty)
            if pflav and flav and pflav != flav:
                df.at[i, "washdown_required"] = True
                reason = f"Flavour change: {pflav} → {flav}"
                df.at[i, "washdown_reason"] = (df.at[i, "washdown_reason"] + " | " + reason).strip(" |")

        prev[m] = (prod, flav, is_plain)

    return df

# src/sequencer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# CONFIG (tune these)
# -----------------------------
@dataclass
class ChangeoverConfig:
    # time weights (minutes) – tune to your factory
    BASE_PRODUCT_CHANGE_MIN: float = 10.0
    FLAVOUR_CHANGE_EXTRA_MIN: float = 12.0
    PLAIN_TO_FLAVOURED_EXTRA_MIN: float = 8.0
    GRANOLA_CHANGE_EXTRA_MIN: float = 15.0

    # washdown “hard” time adders (minutes)
    WASHDOWN_MIN: float = 25.0
    FULL_WASHDOWN_MIN: float = 45.0

    # policy: if true, any granola product change on M3 requires washdown
    M3_ALWAYS_WASHDOWN_ON_PRODUCT_CHANGE: bool = True


CFG = ChangeoverConfig()


# -----------------------------
# Helpers (features)
# -----------------------------
def _s(x) -> str:
    return "" if x is None else str(x)

def is_granola(product_name: str) -> bool:
    return "granola" in _s(product_name).lower() or _s(product_name).lower().startswith("ss ")

def has_chocolate(product_name: str, flavour_label: str = "") -> bool:
    t = (_s(product_name) + " " + _s(flavour_label)).lower()
    return ("choc" in t) or ("chocolate" in t)

def has_nuts_granola_risk(product_name: str, flavour_label: str = "") -> bool:
    # Adjust to your allergen reality (granola often implies cereals/nuts)
    t = (_s(product_name) + " " + _s(flavour_label)).lower()
    return ("granola" in t) or ("nut" in t) or ("almond" in t) or ("hazelnut" in t)

def flavour_bucket(flavour_label: str) -> str:
    # light normalization so “mango granola” and “mango” align
    f = _s(flavour_label).strip().lower()
    return f

def make_features(row: pd.Series) -> Dict[str, object]:
    pn = _s(row.get("product_name", ""))
    fl = _s(row.get("flavour_label", ""))
    return {
        "product_name": pn,
        "flavour": flavour_bucket(fl),
        "is_plain": bool(row.get("is_plain_yoghurt", False)),
        "is_granola": is_granola(pn),
        "has_choc": has_chocolate(pn, fl),
        "has_nut_risk": has_nuts_granola_risk(pn, fl),
        "pack": row.get("pack_size_g", np.nan),
        "machine": _s(row.get("machine", "")),
    }


# -----------------------------
# Transition rules
# -----------------------------
def transition_cost(a: Dict[str, object], b: Dict[str, object], cfg: ChangeoverConfig) -> Tuple[float, bool, str]:
    """
    Returns (cost_minutes, washdown_required, reason)
    """
    if a["machine"] != b["machine"]:
        # we never transition across machines in sequencing; treat as huge cost
        return 1e9, True, "Different machine"

    cost = 0.0
    reasons: List[str] = []
    washdown = False

    # Product change base
    if a["product_name"] != b["product_name"]:
        cost += cfg.BASE_PRODUCT_CHANGE_MIN
        reasons.append("product change")

    # Granola rule (M3)
    if a["machine"] == "M3" and cfg.M3_ALWAYS_WASHDOWN_ON_PRODUCT_CHANGE:
        if a["product_name"] != b["product_name"]:
            washdown = True
            cost += cfg.WASHDOWN_MIN
            reasons.append("M3 granola washdown")

    # Flavour change
    if a["flavour"] and b["flavour"] and a["flavour"] != b["flavour"]:
        cost += cfg.FLAVOUR_CHANGE_EXTRA_MIN
        washdown = True
        reasons.append(f"flavour change {a['flavour']}→{b['flavour']}")

    # Plain ↔ flavoured
    if bool(a["is_plain"]) != bool(b["is_plain"]):
        cost += cfg.PLAIN_TO_FLAVOURED_EXTRA_MIN
        washdown = True
        reasons.append("plain↔flavoured switch")

    # Granola ↔ non-granola
    if bool(a["is_granola"]) != bool(b["is_granola"]):
        cost += cfg.GRANOLA_CHANGE_EXTRA_MIN
        washdown = True
        reasons.append("granola↔non-granola")

    # “Harder” washdown triggers (tune to your HACCP/allergen reality)
    # Example: nut/choc risk transitions → full washdown
    if bool(a["has_nut_risk"]) != bool(b["has_nut_risk"]):
        washdown = True
        cost += (cfg.FULL_WASHDOWN_MIN - cfg.WASHDOWN_MIN)
        reasons.append("allergen risk shift (full washdown)")

    if bool(a["has_choc"]) != bool(b["has_choc"]):
        washdown = True
        cost += (cfg.FULL_WASHDOWN_MIN - cfg.WASHDOWN_MIN)
        reasons.append("choc/non-choc shift (full washdown)")

    return cost, washdown, "; ".join(reasons) if reasons else "no change"


# -----------------------------
# Sequencing algorithm (greedy)
# -----------------------------
def build_sequence_for_machine(df_m: pd.DataFrame, cfg: ChangeoverConfig) -> pd.DataFrame:
    """
    Returns df_m with a proposed order (sequence_rank) and transition notes.
    Greedy approach:
      - choose a smart start (plain/low-risk)
      - repeatedly pick the next product with minimal transition cost
    """
    df_m = df_m.copy().reset_index(drop=True)

    # Create one “job” per distinct product run candidate
    # If your plan has duplicates, collapse to unique SKUs for sequencing:
    keys = ["product_name", "flavour_label", "pack_size_g", "machine"]
    df_jobs = df_m.drop_duplicates(subset=[k for k in keys if k in df_m.columns]).copy().reset_index(drop=True)

    if len(df_jobs) <= 1:
        df_jobs["sequence_rank"] = range(1, len(df_jobs) + 1)
        df_jobs["washdown_required"] = False
        df_jobs["washdown_reason"] = ""
        df_jobs["changeover_required"] = False
        df_jobs["changeover_reason"] = ""
        return df_jobs

    # Precompute features
    feats = [make_features(df_jobs.loc[i]) for i in range(len(df_jobs))]

    # Choose start:
    # Prefer plain + non-granola + no nut risk + no choc (lowest cleaning burden)
    def start_score(f: Dict[str, object]) -> float:
        s = 0.0
        if f["is_plain"]: s -= 5.0
        if not f["is_granola"]: s -= 2.0
        if not f["has_nut_risk"]: s -= 2.0
        if not f["has_choc"]: s -= 1.0
        return s

    start_idx = min(range(len(feats)), key=lambda i: start_score(feats[i]))

    remaining = set(range(len(feats)))
    order: List[int] = []
    order.append(start_idx)
    remaining.remove(start_idx)

    # Greedy next
    while remaining:
        last = order[-1]
        best_i = None
        best_cost = 1e18
        best_wash = False
        best_reason = ""

        for j in remaining:
            c, w, r = transition_cost(feats[last], feats[j], cfg)
            if c < best_cost:
                best_cost = c
                best_i = j
                best_wash = w
                best_reason = r

        order.append(best_i)  # type: ignore
        remaining.remove(best_i)  # type: ignore

    # Build transition columns
    seq_rows = df_jobs.loc[order].copy().reset_index(drop=True)
    seq_rows["sequence_rank"] = range(1, len(seq_rows) + 1)

    wash_flags = [False]
    wash_reasons = [""]
    chg_flags = [False]
    chg_reasons = ["START"]

    for i in range(1, len(seq_rows)):
        a = make_features(seq_rows.loc[i - 1])
        b = make_features(seq_rows.loc[i])
        c, w, r = transition_cost(a, b, cfg)

        prod_change = a["product_name"] != b["product_name"]
        chg_flags.append(bool(prod_change))
        chg_reasons.append(r if prod_change else "no product change")

        wash_flags.append(bool(w))
        wash_reasons.append(r if w else "")

    seq_rows["changeover_required"] = chg_flags
    seq_rows["changeover_reason"] = chg_reasons
    seq_rows["washdown_required"] = wash_flags
    seq_rows["washdown_reason"] = wash_reasons

    return seq_rows


def propose_sequence(df: pd.DataFrame, cfg: ChangeoverConfig = CFG) -> pd.DataFrame:
    """
    Returns a proposed sequence per machine.
    Output includes:
      - sequence_rank
      - washdown_required / reason
      - changeover_required / reason
    """
    if "machine" not in df.columns:
        raise ValueError("propose_sequence requires a 'machine' column (auto-assign it in normalize.py)")

    out_parts = []
    for m, df_m in df.groupby("machine", dropna=False):
        df_m = df_m.copy()
        df_m["machine"] = str(m)
        out_parts.append(build_sequence_for_machine(df_m, cfg))

    return pd.concat(out_parts, ignore_index=True)


# -----------------------------
# Actions for the agent
# -----------------------------
def sequence_actions(df: pd.DataFrame) -> List[str]:
    """
    Creates a simple action label for each row (in the proposed sequence table).
    """
    actions: List[str] = []
    for _, r in df.iterrows():
        if bool(r.get("washdown_required", False)):
            actions.append("WASHDOWN + RUN")
        elif bool(r.get("changeover_required", False)):
            actions.append("CHANGEOVER + RUN")
        else:
            actions.append("RUN")
    return actions


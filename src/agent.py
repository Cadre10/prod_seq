# src/agent.py

import pandas as pd

from src.normalize import normalize_data
from src.risk_model import score_risk
from src.sequencer import sequence_actions


def main():
    print("🔍 Yoghurt AI Agent starting...")

    # ---- LOAD DATA ----
    df = pd.read_csv("data/input/Prod_Plan.csv")
    print(f"✅ Loaded {len(df)} rows")

    # ---- NORMALISE ----
    df= normalize_data(df)
    print("✅ Data normalised")

    # ---- SCORE RISK ----
    df["risk_final"] = score_risk(df)
    print("✅ Risk scored")

    # ---- SEQUENCE ACTIONS ----
    actions = sequence_actions(df)

    print("\n📋 AGENT OUTPUT:")
    for a in actions:
        print("•", a)


if __name__ == "__main__":
    main()

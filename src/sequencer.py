# src/sequencer.py

import pandas as pd

print("✅ sequencer.py LOADED")


def sequence_actions(df: pd.DataFrame):
    """
    Convert risk scores into ordered operator actions
    """
    actions = []

    for _, row in df.iterrows():
        risk = row.get("risk_final", 0)
        product = row.get("product_name", "UNKNOWN PRODUCT")

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

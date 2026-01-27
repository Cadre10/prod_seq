import pandas as pd

WASHDOWN_MIN = 30
CHANGEOVER_MIN = 10


def classify_product(name: str):
    n = name.lower()

    return {
        "is_plain": any(k in n for k in ["natural", "greek"]) and "flavour" not in n,
        "is_granola": "granola" in n,
        "is_ss": n.startswith("ss"),
        "is_allergen": any(k in n for k in ["choc", "chocolate"]),
    }


import pandas as pd

def assign_machine(row):
    """
    Assign machine based on product name and pack size.
    Safe against NaN / missing values.
    """

    # --- SAFETY FIRST ---
    # Get product name safely
    product = row.get("product_name") or row.get("Product name")
    if pd.isna(product):
        return "UNKNOWN"

    name = str(product).lower()

    # Get pack size safely
    pack = row.get("pack_size_g", None)
    try:
        pack = int(pack)
    except Exception:
        pack = None

    # -------------------
    # BUCKET LINE RULES
    # -------------------
    if pack in (2000, 5000, 10000):
        return "BUCKET_LINE"

    # -------------------
    # GRANOLA RULES
    # -------------------
    if "granola" in name:
        return "M3"

    # -------------------
    # 450g RULES
    # -------------------
    if pack == 450:
        return "M2"

    # -------------------
    # SMALL POTS RULES
    # -------------------
    if pack in (150, 170, 175):
        return "M1"

    # -------------------
    # FALLBACK
    # -------------------
    return "UNKNOWN"



def sequence_machine(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # extract attributes
    attrs = df["Product name"].apply(classify_product)
    df = pd.concat([df, attrs.apply(pd.Series)], axis=1)

    # sort key: safest first
    df["_sort"] = (
        df["is_plain"].astype(int) * -10
        + df["is_granola"].astype(int) * 5
        + df["is_allergen"].astype(int) * 5
    )

    df = df.sort_values("_sort").drop(columns="_sort")

    # compute transitions
    prev = None
    washdowns = []
    changeovers = []
    downtime = []

    for _, row in df.iterrows():
        if prev is None:
            washdowns.append(False)
            changeovers.append(False)
            downtime.append(0)
        else:
            wd = (
                row["is_plain"] != prev["is_plain"]
                or row["is_granola"]
                or row["is_allergen"] != prev["is_allergen"]
                or row["is_ss"] != prev["is_ss"]
            )

            if wd:
                washdowns.append(True)
                changeovers.append(False)
                downtime.append(WASHDOWN_MIN)
            else:
                washdowns.append(False)
                changeovers.append(True)
                downtime.append(CHANGEOVER_MIN)

        prev = row

    df["washdown_required"] = washdowns
    df["changeover_required"] = changeovers
    df["downtime_min"] = downtime
    df["sequence"] = range(1, len(df) + 1)

    return df


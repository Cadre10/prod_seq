from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/input")
DEFAULT_FILE = DATA_DIR / "Prod_Plan_Today.csv"

def load_input_csv(path: str | None = None) -> pd.DataFrame:
    """
    Load production plan CSV.
    If path is None, loads default daily plan.
    """
    csv_path = Path(path) if path else DEFAULT_FILE

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path.resolve()}")

    return pd.read_csv(csv_path)

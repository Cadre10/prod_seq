import pandas as pd

def load_daily_plan(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if ' Date' in df.columns and 'Date' not in df.columns:
        df = df.rename(columns={' Date': 'Date'})
    return df

def load_form_responses(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if 'Total Volume (kg)' not in df.columns:
        for c in list(df.columns):
            if c.replace(' ', '') == 'TotalVolume(kg)':
                df = df.rename(columns={c: 'Total Volume (kg)'})
    return df
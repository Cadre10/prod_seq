import pandas as pd

def load_daily_plan(path):
    df_daily = pd.read_csv(path)
    df_daily.columns = [str(c).strip() for c in df_daily.columns]
    return df_daily

def load_form_responses(path):
    df_form = pd.read_csv(path)
    df_form.columns = [str(c).strip() for c in df_form.columns]
    return df_form

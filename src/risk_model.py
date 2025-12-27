import pandas as pd
from .normalize import norm_text, infer_flavour_label, RISK_MAP

def build_product_risk_profile(df_form):
    form = df_form.copy()
    form['Date_parsed'] = pd.to_datetime(form.get('Date'), errors='coerce')
    form['Product_name'] = form.get('Product name').astype(str)
    form['Product_norm'] = form['Product_name'].map(norm_text)

    flavour_col = 'Flavour name' if 'Flavour name' in form.columns else ''
    form['Flavour_label'] = [
        infer_flavour_label(p, f)
        for p, f in zip(form['Product_name'], form.get(flavour_col, '').astype(str))
    ]
    form['Risk_score'] = form['Flavour_label'].map(RISK_MAP).fillna(5).astype(float)

    prof = (
        form.groupby('Product_norm', as_index=False)
        .agg(
            hist_entries=('Product_norm', 'size'),
            hist_risk_median=('Risk_score', 'median'),
            hist_risk_mean=('Risk_score', 'mean'),
            hist_common_flavour=('Flavour_label', lambda s: s.value_counts().index[0] if len(s.dropna()) else 'other')
        )
    )
    return prof
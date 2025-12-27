import pandas as pd
import numpy as np
from .normalize import norm_text, infer_flavour_label, extract_pack_size_g, RISK_MAP

def build_plan_jobs(df_daily, prod_profile=None):
    plan = df_daily.copy()
    plan.columns = [str(c).strip() for c in plan.columns]
    if ' Date' in plan.columns and 'Date' not in plan.columns:
        plan = plan.rename(columns={' Date': 'Date'})

    plan['Date_parsed'] = pd.to_datetime(plan.get('Date'), errors='coerce')
    plan['Product_name'] = plan.get('Product name').astype(str)
    plan['Product_norm'] = plan['Product_name'].map(norm_text)

    plan['Plan_total_mix_kg'] = pd.to_numeric(plan.get('Plan Total mixd yogh(kg)', np.nan), errors='coerce')
    plan['Plan_base_L'] = pd.to_numeric(plan.get('Plan Base Volume (L)', np.nan), errors='coerce')
    plan['Qty_kg_proxy'] = plan['Plan_total_mix_kg']
    plan.loc[plan['Qty_kg_proxy'].isna(), 'Qty_kg_proxy'] = plan.loc[plan['Qty_kg_proxy'].isna(), 'Plan_base_L']

    plan['Pack_size_g'] = pd.to_numeric(plan.get('Pack size (g)', np.nan), errors='coerce')
    plan['Pack_size_g'] = plan['Pack_size_g'].fillna(plan['Product_name'].map(extract_pack_size_g))

    if prod_profile is not None:
        plan = plan.merge(prod_profile, on='Product_norm', how='left')
    else:
        plan['hist_entries'] = np.nan
        plan['hist_risk_median'] = np.nan
        plan['hist_common_flavour'] = np.nan

    plan['infer_flavour_from_name'] = [infer_flavour_label(p, '') for p in plan['Product_name'].tolist()]
    plan['risk_final'] = plan['hist_risk_median']
    plan.loc[plan['risk_final'].isna(), 'risk_final'] = (
        plan.loc[plan['risk_final'].isna(), 'infer_flavour_from_name'].map(RISK_MAP).fillna(5)
    )

    return plan

def sequence_jobs(df_jobs):
    jobs = df_jobs.copy()
    jobs['risk_final'] = pd.to_numeric(jobs['risk_final'], errors='coerce').fillna(5)
    jobs['Pack_size_g'] = pd.to_numeric(jobs['Pack_size_g'], errors='coerce')
    jobs['Qty_kg_proxy'] = pd.to_numeric(jobs['Qty_kg_proxy'], errors='coerce')

    jobs = jobs.sort_values(['risk_final', 'Pack_size_g', 'Qty_kg_proxy'], ascending=[True, True, False]).reset_index(drop=True)
    jobs['Sequence'] = np.arange(1, len(jobs) + 1)

    def explain_row(r):
        lab = r.get('hist_common_flavour')
        if pd.isna(lab) or str(lab) == '':
            lab = r.get('infer_flavour_from_name')
        return 'risk=' + str(int(r.get('risk_final'))) + ' flavour=' + str(lab)

    jobs['Rationale'] = jobs.apply(explain_row, axis=1)
    return jobs
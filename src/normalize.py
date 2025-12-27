import re
import pandas as pd
import numpy as np

def norm_text(val):
    if pd.isna(val):
        return ''
    return str(val).strip().lower()

def extract_pack_size_g(product_name):
    if pd.isna(product_name):
        return np.nan
    s = str(product_name)
    m = re.search(r'(\d+)\s*g\b', s.lower())
    if m:
        return float(m.group(1))
    return np.nan

def infer_flavour_label(product_name, flavour_name=''):
    s = norm_text(product_name) + ' ' + norm_text(flavour_name)
    buckets = [
        ('plain', ['plain', 'natural', 'greek']),
        ('vanilla', ['vanilla']),
        ('honey', ['honey']),
        ('strawberry', ['strawberry']),
        ('blueberry', ['blueberry']),
        ('raspberry', ['raspberry']),
        ('mango', ['mango']),
        ('apple_cinnamon', ['apple', 'cinamon', 'cinnamon']),
        ('nectarine', ['nectarine']),
        ('chocolate', ['choc', 'chocolate']),
        ('granola', ['granola']),
        ('other', [])
    ]
    for lab, kws in buckets:
        if lab == 'other':
            continue
        if any(k in s for k in kws):
            return lab
    return 'other'

RISK_MAP = {
    'plain': 0,
    'vanilla': 1,
    'honey': 2,
    'nectarine': 3,
    'apple_cinnamon': 3,
    'mango': 4,
    'strawberry': 5,
    'blueberry': 6,
    'granola': 6,
    'raspberry': 7,
    'chocolate': 8,
    'other': 5
}
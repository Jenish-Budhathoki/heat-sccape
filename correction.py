"""
Bias correction for SDGSAT-1 TIS LST retrieval.
Best score used: DEFAULT_CORRECTION = {'__global__': (1.0, -1.65)}
"""
import os
import numpy as np

DEFAULT_CORRECTION = {
    '__global__': (1.0, -1.65),
}

def _fit_linear(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return 1.0, 0.0
    xv, yv = x[valid], y[valid]
    A = np.vstack([xv, np.ones(len(xv))]).T
    a, b = np.linalg.lstsq(A, yv, rcond=None)[0]
    return float(a), float(b)

def load_training_labels(csv_path):
    import csv as _csv
    if not os.path.exists(csv_path):
        return []
    records = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = _csv.DictReader(f)
        for row in reader:
            try:
                lst_val = row.get('LST(K)', '').strip()
                if not lst_val:
                    continue
                records.append({
                    'file_id':  row['File'].strip(),
                    'lat':      float(row['Lat']),
                    'lon':      float(row['Lon']),
                    'lst_true': float(lst_val),
                })
            except (ValueError, KeyError):
                continue
    return records

def train_correction(physics_lst_arr, true_lst_arr):
    a, b = _fit_linear(
        np.asarray(physics_lst_arr, dtype=np.float64),
        np.asarray(true_lst_arr,    dtype=np.float64),
    )
    return {'__global__': (a, b)}

def apply_correction(lst_val, coeffs=None):
    if coeffs is None:
        coeffs = DEFAULT_CORRECTION
    a, b = coeffs.get('__global__', (1.0, 0.0))
    return float(a * lst_val + b)

def update_default_correction(coeffs):
    DEFAULT_CORRECTION.update(coeffs)

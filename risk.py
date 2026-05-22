"""
Thermal risk classification and summary for SDGSAT-1 LST output.
"""
import numpy as np
import json
import os

RISK_THRESHOLDS = {
    'Low':      (-np.inf, 1.0),
    'Moderate': (1.0,     2.0),
    'High':     (2.0,     3.0),
    'Extreme':  (3.0,     np.inf),
}

def classify_risk(lst_val, scene_mean, scene_std):
    if not np.isfinite(lst_val) or scene_std == 0:
        return 'Unknown'
    z = (lst_val - scene_mean) / scene_std
    for label, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= z < hi:
            return label
    return 'Extreme'

def build_risk_summary(results: dict) -> dict:
    values = []
    for v in results.values():
        if isinstance(v, list) and len(v) > 0:
            val = v[0]
            if np.isfinite(val):
                values.append(val)

    if not values:
        return {'error': 'No valid LST values'}

    arr = np.array(values)
    mean, std = float(np.mean(arr)), float(np.std(arr))

    risk_counts = {k: 0 for k in RISK_THRESHOLDS}
    risk_counts['Unknown'] = 0
    for v in values:
        label = classify_risk(v, mean, std)
        risk_counts[label] = risk_counts.get(label, 0) + 1

    return {
        'n_points':   len(values),
        'mean_lst_K': round(mean, 4),
        'std_lst_K':  round(std,  4),
        'min_lst_K':  round(float(np.min(arr)), 4),
        'max_lst_K':  round(float(np.max(arr)), 4),
        'risk_distribution': risk_counts,
    }

def write_risk_summary(results: dict, output_dir: str):
    summary = build_risk_summary(results)
    path = os.path.join(output_dir, 'risk_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    return summary

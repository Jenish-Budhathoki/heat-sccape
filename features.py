"""
Feature extraction for SDGSAT-1 TIS LST prediction.
26 features — version that produced best competition score.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lst import split_window_lst, DEFAULT_EPS_B2, DEFAULT_EPS_B3

FEATURE_NAMES = [
    'bt1', 'bt2', 'bt3',
    'dT23', 'dT12', 'dT13',
    'swa_lst',
    'bt2_mean5', 'bt2_std5',
    'bt2_mean11', 'bt2_std11',
    'bt3_mean5', 'bt3_std5',
    'scene_bt2_mean', 'scene_bt2_std',
    'scene_bt3_mean', 'scene_bt3_std',
    'bt2_anom', 'bt3_anom', 'swa_anom',
    'month_sin', 'month_cos',
    'point_lat', 'point_lon',
    'scene_lat', 'scene_lon',
]

def parse_scene_meta(file_id):
    parts = file_id.split('_')
    date_str = parts[2]
    month = int(date_str[4:6])
    scene_lon, scene_lat = 0.0, 0.0
    for p in parts:
        if p.startswith('E'):
            try: scene_lon = float(p[1:])
            except ValueError: pass
        elif p.startswith('N'):
            try: scene_lat = float(p[1:])
            except ValueError: pass
    return month, scene_lat, scene_lon

def compute_scene_stats(bt2_arr, bt3_arr):
    swa = split_window_lst(bt2_arr, bt3_arr, DEFAULT_EPS_B2, DEFAULT_EPS_B3)
    return {
        'scene_bt2_mean': float(np.nanmean(bt2_arr)),
        'scene_bt2_std':  float(np.nanstd(bt2_arr)),
        'scene_bt3_mean': float(np.nanmean(bt3_arr)),
        'scene_bt3_std':  float(np.nanstd(bt3_arr)),
        'scene_swa_mean': float(np.nanmean(swa)),
    }

def _window_stats(arr, row, col, radius):
    h, w = arr.shape
    r0 = max(0, row - radius); r1 = min(h, row + radius + 1)
    c0 = max(0, col - radius); c1 = min(w, col + radius + 1)
    patch = arr[r0:r1, c0:c1]
    valid = patch[np.isfinite(patch)]
    if len(valid) == 0:
        return np.nan, np.nan
    return float(np.mean(valid)), float(np.std(valid))

def extract_features(bt1_arr, bt2_arr, bt3_arr,
                     row, col, file_id, lat, lon,
                     scene_stats=None):
    h, w = bt2_arr.shape

    def _sample(arr, r, c, win=5):
        for radius in range(win + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        v = arr[rr, cc]
                        if np.isfinite(v):
                            return float(v)
        return float(np.nanmedian(arr))

    bt1 = _sample(bt1_arr, row, col)
    bt2 = _sample(bt2_arr, row, col)
    bt3 = _sample(bt3_arr, row, col)

    if not (np.isfinite(bt2) or np.isfinite(bt3)):
        return None

    dT23 = bt2 - bt3 if (np.isfinite(bt2) and np.isfinite(bt3)) else np.nan
    dT12 = bt1 - bt2 if (np.isfinite(bt1) and np.isfinite(bt2)) else np.nan
    dT13 = bt1 - bt3 if (np.isfinite(bt1) and np.isfinite(bt3)) else np.nan

    if np.isfinite(bt2) and np.isfinite(bt3):
        swa_lst = float(split_window_lst(bt2, bt3, DEFAULT_EPS_B2, DEFAULT_EPS_B3))
    elif np.isfinite(bt2):
        swa_lst = bt2
    else:
        swa_lst = np.nan

    bt2_m5,  bt2_s5  = _window_stats(bt2_arr, row, col, radius=5)
    bt2_m11, bt2_s11 = _window_stats(bt2_arr, row, col, radius=11)
    bt3_m5,  bt3_s5  = _window_stats(bt3_arr, row, col, radius=5)

    if scene_stats is None:
        scene_stats = compute_scene_stats(bt2_arr, bt3_arr)

    s_bt2_mean = scene_stats['scene_bt2_mean']
    s_bt2_std  = scene_stats['scene_bt2_std']
    s_bt3_mean = scene_stats['scene_bt3_mean']
    s_bt3_std  = scene_stats['scene_bt3_std']
    s_swa_mean = scene_stats['scene_swa_mean']

    bt2_anom = bt2 - s_bt2_mean
    bt3_anom = bt3 - s_bt3_mean
    swa_anom = swa_lst - s_swa_mean if np.isfinite(swa_lst) else np.nan

    month, scene_lat, scene_lon = parse_scene_meta(file_id)
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)

    vec = np.array([
        bt1, bt2, bt3,
        dT23, dT12, dT13,
        swa_lst,
        bt2_m5, bt2_s5,
        bt2_m11, bt2_s11,
        bt3_m5, bt3_s5,
        s_bt2_mean, s_bt2_std,
        s_bt3_mean, s_bt3_std,
        bt2_anom, bt3_anom, swa_anom,
        month_sin, month_cos,
        lat, lon,
        scene_lat, scene_lon,
    ], dtype=np.float64)

    return vec

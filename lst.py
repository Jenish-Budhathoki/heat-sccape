"""
LST retrieval for SDGSAT-1 TIS.
OLS TCWV-aware three-channel split-window algorithm.
"""
import numpy as np

DEFAULT_EPS_B2 = 0.970
DEFAULT_EPS_B3 = 0.965
RHO = 14387.69

TCWV_COEFFS = (
     535.02056273,
      -2.74658257,
      -0.90381818,
      -0.11613069,
      -0.88174931,
      17.38816829,
       2.67510281,
      -2.19708178,
      -0.00259474,
       0.00654154,
       0.00758955,
)

COLD   = (-22.395417, 1.085547, -1.556580, 1.451483, -0.428544, -0.727851, -0.111977)
MILD   = (-4.253524,  1.011510, -0.659701, 0.170676, -0.835996, -0.138240, -0.021268)
WARM   = (-152.746475, 1.544852, -0.420073, -0.819235, -0.225125, -4.964260, -0.763732)
GLOBAL = (-4.384680,  1.017611, -0.334948, -0.087668, -0.648772, -0.142502, -0.021923)

def tcwv_lst(bt1, bt2, bt3, tcwv, eps2=DEFAULT_EPS_B2, eps3=DEFAULT_EPS_B3):
    eps_mean = (eps2 + eps3) / 2.0
    d_eps    = eps2 - eps3
    dT23     = bt2 - bt3
    dT12     = bt1 - bt2
    B = TCWV_COEFFS
    return (B[0] + B[1]*bt2 + B[2]*dT23 + B[3]*dT23**2
            + B[4]*dT12 + B[5]*(1-eps_mean) + B[6]*d_eps
            + B[7]*tcwv + B[8]*tcwv*dT23
            + B[9]*bt2**2 + B[10]*bt2*tcwv)

def _apply_coeffs(coeffs, bt1, bt2, bt3, eps2, eps3):
    eps_mean = (eps2 + eps3) / 2.0
    d_eps    = eps2 - eps3
    dT23     = bt2 - bt3
    dT12     = bt1 - bt2
    B0,B1,B2,B3,B4,B5,B6 = coeffs
    return B0 + B1*bt2 + B2*dT23 + B3*dT23**2 + B4*dT12 + B5*(1-eps_mean) + B6*d_eps

def three_channel_lst(bt1, bt2, bt3, eps2=DEFAULT_EPS_B2, eps3=DEFAULT_EPS_B3,
                      scene_bt2_mean=None):
    bt1 = np.asarray(bt1, dtype=np.float64)
    bt2 = np.asarray(bt2, dtype=np.float64)
    bt3 = np.asarray(bt3, dtype=np.float64)
    if scene_bt2_mean is not None:
        if scene_bt2_mean < 278:
            coeffs = COLD
        elif scene_bt2_mean < 290:
            coeffs = MILD
        else:
            coeffs = WARM
    else:
        coeffs = GLOBAL
    return _apply_coeffs(coeffs, bt1, bt2, bt3, eps2, eps3)

def split_window_lst(bt2, bt3, eps2=DEFAULT_EPS_B2, eps3=DEFAULT_EPS_B3):
    bt2 = np.asarray(bt2, dtype=np.float64)
    bt3 = np.asarray(bt3, dtype=np.float64)
    eps_mean  = (eps2 + eps3) / 2.0
    delta_eps = eps2 - eps3
    dT        = bt2 - bt3
    return (bt2 + 1.378*dT + 0.183*dT**2 - 0.268
            + (54.30 - 2.238*dT)*(1.0-eps_mean)
            + (-129.20 + 16.40*dT)*delta_eps)

def single_channel_lst(bt, wavelength_um=10.73, emissivity=DEFAULT_EPS_B2):
    bt  = np.asarray(bt,  dtype=np.float64)
    eps = np.asarray(emissivity, dtype=np.float64)
    return bt / (1.0 + (wavelength_um * bt / RHO) * np.log(eps))

def lst_for_point(bt2_val, bt3_val, eps2=DEFAULT_EPS_B2, eps3=DEFAULT_EPS_B3,
                  bt1_val=None, scene_bt2_mean=None, tcwv=None):
    if (bt1_val is not None and tcwv is not None
            and np.isfinite(bt1_val) and np.isfinite(bt2_val) and np.isfinite(bt3_val)):
        return float(tcwv_lst(bt1_val, bt2_val, bt3_val, tcwv, eps2, eps3))
    if (bt1_val is not None
            and np.isfinite(bt1_val) and np.isfinite(bt2_val) and np.isfinite(bt3_val)):
        return float(three_channel_lst(bt1_val, bt2_val, bt3_val, eps2, eps3, scene_bt2_mean))
    if np.isfinite(bt2_val) and np.isfinite(bt3_val):
        return float(split_window_lst(bt2_val, bt3_val, eps2, eps3))
    if np.isfinite(bt2_val):
        return float(single_channel_lst(bt2_val, 10.73, eps2))
    return float('nan')

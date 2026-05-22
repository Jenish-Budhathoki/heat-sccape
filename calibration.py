"""
Radiometric calibration for SDGSAT-1 TIS.
DN → Spectral Radiance → Brightness Temperature.
"""
import numpy as np
import re
import os

C1 = 1.19104e-16   # W·m²/sr
C2 = 0.0143878     # m·K

BAND_WAVELENGTHS = {
    'B1': 9.35e-6,
    'B2': 10.73e-6,
    'B3': 11.72e-6,
}

def parse_calib_xml(xml_path):
    with open(xml_path, 'rb') as f:
        raw = f.read()
    try:
        content = raw.decode('utf-8')
    except UnicodeDecodeError:
        content = raw.decode('latin-1')

    gains, biases = {}, {}
    for band in ['B1', 'B2', 'B3']:
        g = re.search(rf'<CalibrationGain[^>]*band["\s=]*{band}[^>]*>([\d.eE+\-]+)', content)
        b = re.search(rf'<CalibrationBias[^>]*band["\s=]*{band}[^>]*>([\d.eE+\-]+)', content)
        if not g:
            g = re.search(rf'{band}.*?gain.*?([\d.eE+\-]+)', content, re.IGNORECASE)
        if not b:
            b = re.search(rf'{band}.*?bias.*?([\d.eE+\-]+)', content, re.IGNORECASE)
        gains[band]  = float(g.group(1)) if g else 1.0
        biases[band] = float(b.group(1)) if b else 0.0
    return gains, biases

def dn_to_radiance(dn_array, gain, bias):
    dn = dn_array.astype(np.float64)
    dn[dn == 0] = np.nan
    return gain * dn + bias

def radiance_to_bt(radiance, wavelength):
    L = np.asarray(radiance, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        bt = C2 / (wavelength * np.log(C1 / (wavelength**5 * L) + 1))
    bt[~np.isfinite(bt)] = np.nan
    return bt

def calibrate_scene(dn_b1, dn_b2, dn_b3, gains, biases):
    rad1 = dn_to_radiance(dn_b1, gains['B1'], biases['B1'])
    rad2 = dn_to_radiance(dn_b2, gains['B2'], biases['B2'])
    rad3 = dn_to_radiance(dn_b3, gains['B3'], biases['B3'])
    bt1 = radiance_to_bt(rad1, BAND_WAVELENGTHS['B1'])
    bt2 = radiance_to_bt(rad2, BAND_WAVELENGTHS['B2'])
    bt3 = radiance_to_bt(rad3, BAND_WAVELENGTHS['B3'])
    return bt1, bt2, bt3

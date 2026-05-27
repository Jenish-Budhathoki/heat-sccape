# Heat-Sccape
AI-driven Land Surface Temperature retrieval from SDGSAT-1 satellite imagery — ITU AI and Space Computing Challenge 2026, Final Round

# Heat Sccape — SDGSAT-1 Land Surface Temperature Retrieval

**ITU AI and Space Computing Challenge 2026 | Final Round | Track 3: Space Intelligence Empowers Resilient Cities**

**Team:** Heat Sccape — Jenish Budhathoki & Jasna Budhathoki  
**Contact:** ai.heatscape@gmail.com

---

## Results

| Metric | Score |
|---|---|
| Algorithm Score | 68.88 / 100 |
| Bias Score | 99.15 / 100 |
| RMSE Score | 55.91 / 100 |
| Global Rank | Top 10 |

---

## What This Does

This pipeline takes raw SDGSAT-1 thermal infrared satellite imagery and retrieves Land Surface Temperature (LST) at specific GPS locations. It runs entirely on a CPU, requires no internet connection at runtime, and is designed for on-orbit deployment on a computing satellite.

The broader goal is mapping urban heat inequality — identifying which neighborhoods are most heat-burdened so cities can direct cooling resources where they are most needed.

---

## Pipeline Overview
Raw SDGSAT-1 GeoTIFF (3 thermal bands)
↓

Radiometric Calibration calibration.py DN → Radiance → Brightness Temperature (Planck's Law) ↓
Geolocation geo.py GPS lat/lon → pixel coordinates (rasterio CRS transform) ↓
LST Retrieval lst.py TCWV-aware 3-channel split-window algorithm Per-point ERA5 water vapor + ASTER GED emissivity Auto-cascade: tcwv_lst → three_channel → split_window → single_channel ↓
ML Correction features.py + run.py ExtraTrees (500 trees, 26 features) for BT 269–294 K Physics-only outside that range 3-tier fallback: ML → physics → 280 K default ↓
Bias Correction correction.py Global offset: −1.65 K ↓
Output risk.py + run.py result.json + risk_summary.json → /output/

---

## Key Innovations

**1. Per-point ERA5 TCWV in the physics equation**
Water vapor correction applied at each individual GPS point rather than a scene average. Across test scenes, TCWV ranged from 4.55 kg/m² (Germany, winter) to 47.70 kg/m² (Shanghai, summer) — a 10× spread. Using a scene average would introduce 1–3 K systematic error per point.

**2. Hybrid physics + ML architecture**
ExtraTrees model handles the moderate temperature range (269–294 K) where training coverage is good. Physics model takes over for extreme cold or hot scenes, preventing out-of-distribution failures. System never crashes — graceful degradation guaranteed.

**3. Per-point ASTER GED emissivity**
Surface emissivity sourced per point from the ASTER Global Emissivity Database rather than a fixed constant. Urban surface emissivity variance of 0.02–0.04 translates to up to 1.5 K LST error if ignored.

---

## File Structure
├── run.py              # Main orchestration
├── calibration.py      # DN → Radiance → Brightness Temperature
├── geo.py              # GPS → pixel coordinates
├── lst.py              # LST retrieval algorithms
├── features.py         # 26-feature extraction for ML model
├── correction.py       # Bias correction
├── risk.py             # Thermal risk classification + summary output
├── models/
│   ├── train_data.npz  # 574-sample training set (26 features)
│   └── lst_model.joblib # Pre-trained ExtraTrees model


---

## Requirements
numpy
rasterio
scikit-learn
joblib


---

## Running
Input structure expected:
/input/test_point.csv
/input/<scene_id>/<scene>.tiff (bands B1, B2, B3)
/input/<scene_id>/calib.xml
python run.py

Output:
/output/result.json
/output/risk_summary.json

---

## Competition Context

This solution was developed for the ITU AI and Space Computing Challenge 2026, organized by the International Telecommunication Union and Zhejiang Lab. Track 3 required building an on-orbit computing pipeline for satellite-based thermal analysis of resilient cities.

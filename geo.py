"""
Geolocation for SDGSAT-1 TIS.
GPS coordinates → pixel coordinates.
"""
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

def latlon_to_pixel(src, lat, lon):
    try:
        xs, ys = warp_transform('EPSG:4326', src.crs, [lon], [lat])
        row, col = src.index(xs[0], ys[0])
        h, w = src.height, src.width
        row = max(0, min(h - 1, row))
        col = max(0, min(w - 1, col))
        return int(row), int(col)
    except Exception:
        return None, None

def sample_array(arr, row, col, search_radius=5):
    if row is None or col is None:
        return float('nan')
    h, w = arr.shape
    for radius in range(search_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = row + dr, col + dc
                if 0 <= rr < h and 0 <= cc < w:
                    v = arr[rr, cc]
                    if np.isfinite(v):
                        return float(v)
    return float(np.nanmedian(arr))

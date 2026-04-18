"""
spatial_analysis.py
-------------------
Reusable wrappers for spatial autocorrelation analysis.
Full implementation: notebooks/03_geospatial_analysis.ipynb
"""
import numpy as np
import geopandas as gpd
import libpysal
from esda.moran import Moran, Moran_Local


def build_queen_weights(gdf: gpd.GeoDataFrame):
    """Build row-standardised Queen contiguity spatial weights."""
    w = libpysal.weights.Queen.from_dataframe(gdf)
    w.transform = 'r'
    return w


def compute_global_moran(y: np.ndarray, w) -> dict:
    m = Moran(y, w)
    return {
        'I':     float(m.I),
        'EI':    float(m.EI),
        'p_sim': float(m.p_sim),
        'z_sim': float(m.z_sim),
    }


def classify_lisa(y: np.ndarray, w, p_threshold: float = 0.05) -> np.ndarray:
    lisa   = Moran_Local(y, w, seed=42)
    sig    = lisa.p_sim < p_threshold
    labels = np.full(len(y), 'Not Significant', dtype=object)
    labels[(lisa.q == 1) & sig] = 'High-High'
    labels[(lisa.q == 2) & sig] = 'Low-High'
    labels[(lisa.q == 3) & sig] = 'Low-Low'
    labels[(lisa.q == 4) & sig] = 'High-Low'
    return labels

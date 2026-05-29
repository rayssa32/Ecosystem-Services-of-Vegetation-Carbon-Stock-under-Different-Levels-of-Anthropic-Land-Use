"""Biomass classification by quantiles for Sankey and other categorical analyses."""

from typing import List, Tuple

import numpy as np


def quantile_edges(values: np.ndarray, n_quantiles: int) -> np.ndarray:
    """Compute quantile bin edges from valid (non-NaN) values.

    Args:
        values: 1D array of biomass (or any continuous) values; NaNs are ignored.
        n_quantiles: Number of quantile classes (e.g. 3 → Low, Medium, High).

    Returns:
        Array of shape (n_quantiles - 1,) with edges between classes.
        Example: n_quantiles=3 → edges at 33.3% and 66.7%.
    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    if np.ma.isMaskedArray(flat):
        flat = np.ma.filled(flat, np.nan)
    valid = flat[~np.isnan(flat)]
    if valid.size == 0:
        return np.array([])
    valid = np.asarray(valid, dtype=np.float64)  # plain ndarray for nanquantile (avoids partition mask warning)
    q = np.linspace(0, 1, n_quantiles + 1)[1:-1]
    return np.nanquantile(valid, q)


def classify_by_quantiles(
    values: np.ndarray,
    n_quantiles: int = 3,
    edges: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each pixel to a biomass class (0 .. n_quantiles-1) using quantiles.

    Args:
        values: 2D or 1D array of biomass values (same as used in violin).
        n_quantiles: Number of classes (e.g. 3 → Low, Medium, High).
        edges: Optional precomputed quantile edges; if None, computed from values.

    Returns:
        - class_ids: Array same shape as values; 0..n_quantiles-1, NaN where values are NaN.
        - edges: Quantile edges used (shape (n_quantiles-1,)).
    """
    if edges is None:
        edges = quantile_edges(values, n_quantiles)
    flat = np.asarray(values, dtype=np.float64).ravel()
    if np.ma.isMaskedArray(flat):
        flat = np.ma.filled(flat, np.nan)
    out = np.full_like(flat, np.nan, dtype=np.float64)
    valid = ~np.isnan(flat)
    if not np.any(valid) or len(edges) == 0:
        return (out.reshape(values.shape), edges)
    # digitize returns 1..n_quantiles for n_quantiles-1 edges; convert to 0..n_quantiles-1
    out[valid] = np.digitize(flat[valid], edges, right=False) - 1
    out[valid] = np.clip(out[valid], 0, n_quantiles - 1)
    return (out.reshape(values.shape), edges)


def default_biomass_labels(n_quantiles: int) -> List[str]:
    """Rótulos padrão em português para classes de biomassa por quantis."""
    if n_quantiles <= 3:
        return ["Baixa", "Média", "Alta"][:n_quantiles]
    return [f"Q{i + 1}" for i in range(n_quantiles)]

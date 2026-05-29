"""
Moran's I global spatial autocorrelation for raster biomass (per pixel).

Computes Global Moran's I and permutation-based p-value for biomass values
clipped to a single city, using rook or queen contiguity on the pixel grid.
"""

import warnings
from typing import Literal, Optional, Tuple

import numpy as np
import libpysal.weights as lw
from esda.moran import Moran


def _neighbors_rook() -> Tuple[Tuple[int, int], ...]:
    """Offsets for rook contiguity (4 neighbors)."""
    return ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors_queen() -> Tuple[Tuple[int, int], ...]:
    """Offsets for queen contiguity (8 neighbors)."""
    return (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )


def build_weights_from_grid(
    valid_mask: np.ndarray,
    contiguity: Literal["rook", "queen"] = "rook",
) -> lw.W:
    """Build spatial weights from a 2D valid-pixel mask (rook or queen contiguity).

    Only pixels where valid_mask is True are included. Neighbors are defined
    on the grid (row/col adjacency).

    Args:
        valid_mask: 2D boolean array, True where pixel has valid data.
        contiguity: "rook" (4 neighbors) or "queen" (8 neighbors).

    Returns:
        libpysal W object with one id per valid pixel (row-major order).
    """
    rows, cols = np.where(valid_mask)
    n = len(rows)
    if n == 0:
        raise ValueError("No valid pixels in mask.")

    rc_to_id = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    neighbors = {i: [] for i in range(n)}

    offsets = _neighbors_rook() if contiguity == "rook" else _neighbors_queen()
    for i in range(n):
        r, c = rows[i], cols[i]
        for dr, dc in offsets:
            j = rc_to_id.get((r + dr, c + dc))
            if j is not None:
                neighbors[i].append(j)

    # Recortes por cidade podem ter várias componentes (ilhas); aviso esperado
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*not fully connected|.*disconnected components",
            category=UserWarning,
        )
        return lw.W(neighbors)


def moran_global(
    biomass_2d: np.ndarray,
    contiguity: Literal["rook", "queen"] = "rook",
    permutations: int = 999,
    two_tailed: bool = True,
) -> Tuple[Moran, np.ndarray, lw.W]:
    """Compute Global Moran's I for biomass (2D raster) using valid pixels only.

    NaN and non-finite values are treated as missing and excluded. Spatial
    weights are built from the pixel grid (one city clip).

    Args:
        biomass_2d: 2D array of biomass per pixel (e.g. clipped to one city).
        contiguity: "rook" or "queen" contiguity for weights.
        permutations: Number of permutations for p-value (e.g. 999).
        two_tailed: If True, p-value is two-tailed.

    Returns:
        Tuple of (Moran result object, 1D array of values used, weights W).
    """
    valid = np.isfinite(biomass_2d)
    if not np.any(valid):
        raise ValueError("No finite values in biomass array.")

    y = biomass_2d[valid].astype(np.float64)
    w = build_weights_from_grid(valid, contiguity=contiguity)

    # Remove islands (pixels with no neighbors) so Moran's I is well-defined
    non_island_ids = [i for i in range(len(y)) if len(w.neighbors.get(i, [])) > 0]
    if len(non_island_ids) < 2:
        raise ValueError(
            "Too few connected pixels (need at least 2 with neighbors) for Moran's I."
        )
    y = y[non_island_ids]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*not fully connected|.*disconnected components",
            category=UserWarning,
        )
        w = lw.w_subset(w, non_island_ids)

    # Garantir alinhamento: libpysal pode retornar id_order ordenado
    id_order = list(w.id_order)
    if id_order != non_island_ids:
        idx_in_y = [non_island_ids.index(k) for k in id_order]
        y = y[idx_in_y]

    mi = Moran(y, w, permutations=permutations, two_tailed=two_tailed)
    return mi, y, w


def moran_scatter_plot(
    moran: Moran,
    title: str = "Diagrama de dispersão de Moran",
    xlabel: str = "Biomassa (padronizada)",
    ylabel: str = "Lag espacial (W·z)",
    save_path: Optional[str] = None,
    scatter_kwds: Optional[dict] = None,
    show_i_in_title: bool = True,
) -> "matplotlib.axes.Axes":
    """Plot Moran scatter plot (z vs W·z) and optionally save.

    Eixo x = z (variável padronizada), eixo y = lag espacial (W·z). A reta tem
    inclinação = Moran's I. Pontos alinhados à reta indicam autocorrelação
    espacial forte (esperado para biomassa em alta resolução).

    Args:
        moran: Fitted Moran result from esda.moran.Moran.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: If set, save figure to this path.
        scatter_kwds: Opções para o scatter (ex.: dict(s=1, alpha=0.5)).
        show_i_in_title: Se True, acrescenta \"(I = valor)\" ao título.

    Returns:
        Matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    kwds = dict(scatter_kwds) if scatter_kwds else {}
    moran.plot_scatter(ax=ax, scatter_kwds=kwds)
    plot_title = f"{title} (I = {moran.I:.3f})" if show_i_in_title else title
    ax.set_title(plot_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return ax

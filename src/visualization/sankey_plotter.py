"""Sankey diagram: land use → biomass class (proportional flows, thickness = count or %)."""

import os
from typing import Dict, List, Optional

import pandas as pd

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def build_flow_df(
    land_use_flat: "np.ndarray",
    biomass_class_flat: "np.ndarray",
    class_map: Dict[int, str],
    biomass_labels: List[str],
    use_percentage: bool = True,
) -> pd.DataFrame:
    """Build flow DataFrame (source, target, value) from pixel-level land use and biomass class.

    Args:
        land_use_flat: 1D array of land use class codes (int or float).
        biomass_class_flat: 1D array of biomass class indices 0..n_quantiles-1.
        class_map: Mapping class code → label (e.g. {0: "Água", 1: "Urbano", ...}).
        biomass_labels: Labels for biomass classes (e.g. ["Low", "Medium", "High"]).
        use_percentage: If True, value = % of pixels; else count.

    Returns:
        DataFrame with columns source, target, value (and optionally count).
    """
    import numpy as np

    mask = ~(np.isnan(land_use_flat) | np.isnan(biomass_class_flat))
    lu = land_use_flat[mask].astype(int)
    bc = biomass_class_flat[mask].astype(int)

    # Map to labels
    source_labels = [class_map.get(c, str(c)) for c in lu]
    target_labels = [
        biomass_labels[i] if 0 <= i < len(biomass_labels) else str(i)
        for i in bc
    ]

    df = pd.DataFrame({"source": source_labels, "target": target_labels})
    flow = df.groupby(["source", "target"], as_index=False).size().rename(columns={"size": "count"})
    if use_percentage:
        total = flow["count"].sum()
        flow["value"] = flow["count"] / total * 100.0
    else:
        flow["value"] = flow["count"].astype(float)
    return flow


def plot_sankey(
    flow_df: pd.DataFrame,
    outpath: str,
    title: str = "Land use → Biomass class",
    left_title: str = "Land use",
    right_title: str = "Biomass class",
    value_label: str = "% pixels",
    width: int = 900,
    height: int = 500,
) -> None:
    """Draw Sankey diagram and save to HTML (and PNG if kaleido available).

    Args:
        flow_df: DataFrame with columns source, target, value (and optionally count).
        outpath: Base path for output (e.g. dados_gerados/sankey/city_sankey).
        title: Plot title.
        left_title: Label for left node column.
        right_title: Label for right node column.
        value_label: Label for link value (e.g. "% pixels" or "pixels").
        width: Figure width in pixels.
        height: Figure height in pixels.
    """
    if not HAS_PLOTLY:
        raise RuntimeError("plotly is required for Sankey. Install with: pip install plotly")

    # Unique nodes: left = sources, right = targets; preserve order for consistent layout
    sources = flow_df["source"].unique().tolist()
    targets = flow_df["target"].unique().tolist()
    all_nodes = sources + [t for t in targets if t not in sources]
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    n_left = len(sources)

    source_idx = flow_df["source"].map(node_to_idx).values
    target_idx = flow_df["target"].map(node_to_idx).values
    value = flow_df["value"].values

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="gray", width=0.5),
                    label=all_nodes,
                    customdata=[left_title] * n_left + [right_title] * (len(all_nodes) - n_left),
                    hovertemplate="%{label}<br>%{customdata}<extra></extra>",
                ),
                link=dict(
                    source=source_idx,
                    target=target_idx,
                    value=value,
                    hovertemplate="%{source.label} → %{target.label}<br>%{value:.2f} " + value_label + "<extra></extra>",
                ),
            )
        ],
        layout=dict(
            title=dict(text=title, x=0.5, xanchor="center"),
            font=dict(size=12),
            width=width,
            height=height,
            margin=dict(t=50, b=20, l=20, r=20),
        ),
    )

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    html_path = outpath if outpath.endswith(".html") else outpath + ".html"
    fig.write_html(html_path)

    try:
        import kaleido
        png_path = html_path.replace(".html", ".png")
        fig.write_image(png_path, scale=2)
    except Exception:
        pass  # PNG optional

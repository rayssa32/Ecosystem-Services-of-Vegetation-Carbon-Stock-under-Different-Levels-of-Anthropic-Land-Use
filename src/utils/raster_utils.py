"""Utility functions for raster operations."""

from affine import Affine
import rasterio
from rasterio.warp import Resampling


def pixel_area_from_transform(transform: Affine) -> float:
    """Calculate pixel area in square meters from transform (for projected CRS).

    Args:
        transform: Affine transformation matrix

    Returns:
        Pixel area in square meters
    """
    return abs(transform.a * transform.e)


def resampling_mode_from_name(name: str) -> Resampling:
    """Map textual name to rasterio Resampling enum.

    Args:
        name: Resampling method name (e.g., "nearest", "bilinear")

    Returns:
        Resampling enum value, defaults to nearest if name not found
    """
    name = (name or "nearest").lower()
    mapping = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "cubicspline": Resampling.cubic_spline,
        "lanczos": Resampling.lanczos,
        "average": Resampling.average,
        "mode": Resampling.mode,
        "max": Resampling.max,
        "min": Resampling.min,
        "med": Resampling.med,
        "q1": Resampling.q1,
        "q3": Resampling.q3,
    }
    return mapping.get(name, Resampling.nearest)

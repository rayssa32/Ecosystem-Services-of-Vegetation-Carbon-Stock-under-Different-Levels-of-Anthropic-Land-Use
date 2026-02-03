"""Raster data loading and clipping operations."""

import os
from typing import Dict, List, Tuple
from contextlib import ExitStack

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.vrt import WarpedVRT
from shapely.geometry import mapping

from ..utils.raster_utils import resampling_mode_from_name


class RasterLoader:
    """Handles loading and clipping of raster data."""

    def __init__(self, resample_mode: str = "nearest"):
        """Initialize raster loader.

        Args:
            resample_mode: Resampling method for metric rasters
        """
        self.resample_mode = resample_mode
        self._resampling = resampling_mode_from_name(resample_mode)

    def validate_paths(self, paths: List[str]) -> None:
        """Validate that all file paths exist.

        Args:
            paths: List of file paths to validate

        Raises:
            FileNotFoundError: If any path does not exist
        """
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path not found: {path}")

    def load_classification_raster(self, path: str) -> rasterio.DatasetReader:
        """Load and validate classification raster.

        Args:
            path: Path to classification raster file

        Returns:
            Open rasterio dataset

        Raises:
            ValueError: If raster is not in a projected CRS
        """
        src = rasterio.open(path)
        if not (src.crs and src.crs.is_projected):
            current_crs = src.crs if src.crs else "Not defined"
            raise ValueError(
                f"Classification raster must be in a projected CRS (meters), "
                f"but found: {current_crs}. "
                f"Please reproject your raster to a projected CRS (e.g., UTM). "
                f"For Brazil, consider EPSG:32723 (UTM 23S) or EPSG:5880 (SIRGAS 2000)."
            )
        return src

    def clip_classification(
        self, src: rasterio.DatasetReader, geometry: List[dict]
    ) -> Tuple[np.ndarray, rasterio.Affine]:
        """Clip classification raster to geometry.

        Args:
            src: Open rasterio dataset for classification
            geometry: Geometry in GeoJSON format

        Returns:
            Tuple of (clipped array, transform)

        Raises:
            ValueError: If geometry is outside raster bounds
        """
        try:
            class_ma, class_transform = rio_mask(src, geometry, crop=True, filled=False)
        except ValueError:
            raise ValueError("Geometry outside raster bounds")

        class_clip = class_ma[0].astype("float32", copy=False)

        # Handle masked arrays
        if np.ma.isMaskedArray(class_ma):
            class_clip[class_ma.mask[0]] = np.nan

        # Handle nodata values
        if src.nodata is not None:
            class_clip[class_clip == float(src.nodata)] = np.nan

        return class_clip, class_transform

    def clip_raster_native(
        self, src: rasterio.DatasetReader, geometry: List[dict]
    ) -> Tuple[np.ndarray, rasterio.Affine]:
        """Clip raster to geometry at its native resolution (no resampling).

        Use para Moran's I "realista": biomassa na resolução nativa (ex. 500 m)
        sem reamostrar para a grade do LULC. A geometria deve estar no CRS do raster.

        Args:
            src: Open rasterio dataset (e.g. biomass).
            geometry: Geometry in GeoJSON format (same CRS as src).

        Returns:
            Tuple of (clipped 2D array, transform).
        """
        try:
            ma, transform = rio_mask(src, geometry, crop=True, filled=False)
        except ValueError:
            raise ValueError("Geometry outside raster bounds")

        out = ma[0].astype("float32", copy=False)
        if np.ma.isMaskedArray(ma):
            out[ma.mask[0]] = np.nan
        if src.nodata is not None:
            out[out == float(src.nodata)] = np.nan
        out[out <= -1e10] = np.nan
        return out, transform

    def clip_metric_raster(
        self,
        src_metric: rasterio.DatasetReader,
        src_class: rasterio.DatasetReader,
        geometry: List[dict],
        class_transform: rasterio.Affine,
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """Clip metric raster and align to classification grid.
        
        This method automatically reprojects and resamples the metric raster to match
        the classification grid. For example, if biomass is 500m/pixel and LULC is 10m/pixel,
        the biomass will be resampled to 10m/pixel using the specified resampling method.

        Args:
            src_metric: Open rasterio dataset for metric (e.g., biomass at 500m/pixel)
            src_class: Open rasterio dataset for classification (e.g., LULC at 10m/pixel)
            geometry: Geometry in GeoJSON format
            class_transform: Transform from classification clip (defines target resolution)
            shape: Shape (height, width) from classification clip (defines target dimensions)

        Returns:
            Clipped and resampled metric array at classification grid resolution
        """
        SENTINEL = np.float32(-3.4e38)

        # Use WarpedVRT to reproject and resample metric raster to classification grid
        # This automatically converts from source resolution (e.g., 500m/pixel) 
        # to target resolution (e.g., 10m/pixel) based on class_transform
        with WarpedVRT(
            src_metric,
            crs=src_class.crs,
            transform=class_transform,  # Target transform (10m/pixel from LULC)
            width=shape[1],  # Target width from LULC
            height=shape[0],  # Target height from LULC
            resampling=self._resampling,  # Resampling method (e.g., bilinear for upsampling)
            src_nodata=src_metric.nodata,
            dst_nodata=float(SENTINEL),
        ) as vrt:
            metr_ma, _ = rio_mask(vrt, geometry, crop=False, filled=False)
            metr = metr_ma[0].astype("float32", copy=False)

        # Handle masked arrays
        if np.ma.isMaskedArray(metr_ma):
            metr[metr_ma.mask[0]] = np.nan

        # Handle nodata values
        metr[metr == SENTINEL] = np.nan
        if src_metric.nodata is not None:
            metr[metr == float(src_metric.nodata)] = np.nan

        # Aggressive cleaning of invalid values
        metr[metr <= -1e10] = np.nan

        return metr

    def open_metric_rasters(
        self, paths: Dict[str, str], stack: ExitStack
    ) -> Dict[str, rasterio.DatasetReader]:
        """Open multiple metric raster files using ExitStack.

        Args:
            paths: Dictionary mapping metric names to file paths
            stack: ExitStack context manager for proper cleanup

        Returns:
            Dictionary mapping metric names to open rasterio datasets
        """
        return {name: stack.enter_context(rasterio.open(path)) for name, path in paths.items()}

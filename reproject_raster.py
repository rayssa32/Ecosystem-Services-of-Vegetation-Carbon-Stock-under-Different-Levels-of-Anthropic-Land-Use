"""
Utility script to reproject a raster from geographic (lat/lon) to projected CRS.

This script reprojects the classification raster to a projected coordinate system
that the analysis pipeline requires for calculating pixel areas.

Usage:
    python reproject_raster.py
"""

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


def reproject_raster(
    input_path: str,
    output_path: str,
    target_crs: str = "EPSG:32728",  # WGS 84 / UTM zone 28S
) -> None:
    """Reproject a raster to a target CRS.

    Args:
        input_path: Path to input raster file
        output_path: Path for output reprojected raster
        target_crs: Target CRS (default: EPSG:5880 for Brazil)
    """
    with rasterio.open(input_path) as src:
        print(f"Input CRS: {src.crs}")
        print(f"Input bounds: {src.bounds}")

        # Calculate transform for the reprojected raster
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        # Update metadata for output
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        print(f"Output CRS: {target_crs}")
        print(f"Output dimensions: {width} x {height}")

        # Reproject and save
        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest,  # Use nearest for categorical data
                )

        print(f"Reprojected raster saved to: {output_path}")


if __name__ == "__main__":
    # Configuration
    input_file = "classificacao/LULC_7_cidades_2025-07-10_2025-07-30.tif"
    output_file = "classificacao/LULC_7_cidades_2025-07-10_2025-07-30_projected.tif"

    # Using WGS 84 / UTM zone 28S (EPSG:32728)
    target_crs = "EPSG:32728"

    print(f"Reprojecting {input_file} to {target_crs}...")
    reproject_raster(input_file, output_file, target_crs)
    print("Done!")

"""
Utility script to reproject rasters from geographic (lat/lon) to projected CRS.

This script can reproject both classification and metric rasters to a projected
coordinate system (EPSG:32728) that the analysis pipeline requires.

Usage:
    python reproject_raster.py
    # Or edit the input/output paths in the script
"""

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


def reproject_raster(
    input_path: str,
    output_path: str,
    target_crs: str = "EPSG:32728",  # WGS 84 / UTM zone 28S
    resampling_method: Resampling = Resampling.nearest,
) -> None:
    """Reproject a raster to a target CRS.

    Args:
        input_path: Path to input raster file
        output_path: Path for output reprojected raster
        target_crs: Target CRS (default: EPSG:32728 - UTM zone 28S)
        resampling_method: Resampling method (default: nearest for categorical, 
                          use bilinear/cubic for continuous data)
    """
    with rasterio.open(input_path) as src:
        print(f"Input CRS: {src.crs}")
        print(f"Input bounds: {src.bounds}")
        print(f"Input dimensions: {src.width} x {src.height}")
        print(f"Input pixel size: {abs(src.transform.a)}m x {abs(src.transform.e)}m (if projected)")

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
        print(f"Output pixel size: {abs(transform.a)}m x {abs(transform.e)}m")

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
                    resampling=resampling_method,
                )

        print(f"Reprojected raster saved to: {output_path}")


if __name__ == "__main__":
    # Configuration for biomass raster reprojection
    # Change these paths as needed
    
    # Reproject biomass raster from EPSG:4326 to EPSG:32728
    biomass_input = "metricas/Biomass_sete_cidades.tif"
    biomass_output = "metricas/Biomass_sete_cidades_projected.tif"
    
    # Using WGS 84 / UTM zone 28S (EPSG:32728) - same as LULC raster
    target_crs = "EPSG:32728"
    
    # Use bilinear resampling for continuous data (biomass values)
    # This provides better quality when upsampling from 500m to 10m
    print(f"Reprojecting {biomass_input} from EPSG:4326 to {target_crs}...")
    print("This will convert biomass from geographic coordinates to projected coordinates.")
    print("The raster will be resampled to match the target CRS grid.\n")
    
    reproject_raster(
        biomass_input, 
        biomass_output, 
        target_crs,
        resampling_method=Resampling.bilinear  # Better for continuous data like biomass
    )
    print("\nDone! Biomass raster is now in EPSG:32728")
    print(f"Update main.py to use: '{biomass_output}'")

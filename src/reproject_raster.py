"""
Reprojetar rasters de coordenadas geográficas para CRS projetado.

Uso: python -m src.reproject_raster
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
        resampling_method: Resampling method (nearest for categorical,
                          bilinear/cubic for continuous data)
    """
    with rasterio.open(input_path) as src:
        print(f"Input CRS: {src.crs}")
        print(f"Input bounds: {src.bounds}")
        print(f"Input dimensions: {src.width} x {src.height}")
        print(f"Input pixel size: {abs(src.transform.a)}m x {abs(src.transform.e)}m (if projected)")

        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

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
    biomass_input = "metricas/Biomass_sete_cidades.tif"
    biomass_output = "metricas/Biomass_sete_cidades_projected.tif"
    target_crs = "EPSG:32728"

    print(f"Reprojecting {biomass_input} from EPSG:4326 to {target_crs}...")
    reproject_raster(
        biomass_input,
        biomass_output,
        target_crs,
        resampling_method=Resampling.bilinear,
    )
    print("\nDone! Biomass raster is now in EPSG:32728")

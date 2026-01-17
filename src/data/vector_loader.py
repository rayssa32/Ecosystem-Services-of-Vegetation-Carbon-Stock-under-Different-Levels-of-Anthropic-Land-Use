"""Vector data loading operations."""

import geopandas as gpd
import rasterio


class VectorLoader:
    """Handles loading and processing of vector data."""

    @staticmethod
    def load_cities(
        path: str, city_field: str, target_crs: rasterio.crs.CRS
    ) -> gpd.GeoDataFrame:
        """Load cities shapefile and reproject to target CRS.

        Args:
            path: Path to shapefile
            city_field: Name of field containing city names
            target_crs: Target CRS for reprojection

        Returns:
            GeoDataFrame with cities reprojected to target CRS

        Raises:
            ValueError: If city_field is not found in the shapefile
        """
        gdf = gpd.read_file(path)
        if city_field not in gdf.columns:
            raise ValueError(f"Field '{city_field}' not found in vector file.")
        return gdf.to_crs(target_crs)

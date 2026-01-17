"""Main analysis pipeline orchestration."""

import os
from contextlib import ExitStack
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import mapping

from ..config import AnalysisConfig
from ..data.raster_loader import RasterLoader
from ..data.vector_loader import VectorLoader
from ..processing.aggregator import DataAggregator
from ..processing.statistics import StatisticsAnalyzer
from ..visualization.graphics_factory import GraphicsFactory
from ..utils.raster_utils import pixel_area_from_transform


class AnalysisPipeline:
    """Main pipeline for ecosystem services analysis."""

    def __init__(self, config: AnalysisConfig):
        """Initialize analysis pipeline.

        Args:
            config: Analysis configuration object
        """
        self.config = config
        self.raster_loader = RasterLoader(config.resample_metrics)
        self.vector_loader = VectorLoader()
        self.aggregator = DataAggregator()
        self.statistics = StatisticsAnalyzer(config)
        self.graphics = GraphicsFactory(config)

    def run(
        self,
        class_raster_path: str,
        metrics_rasters: Dict[str, str],
        vector_cities_path: str,
        city_field: str = "municipio",
        class_map: Optional[Dict[int, str]] = None,
    ) -> pd.DataFrame:
        """Execute complete analysis pipeline.

        Args:
            class_raster_path: Path to classification raster
            metrics_rasters: Dictionary mapping metric names to file paths
            vector_cities_path: Path to cities shapefile
            city_field: Field name containing city names
            class_map: Optional mapping from class codes to names

        Returns:
            Combined DataFrame with statistics from all cities
        """
        # Validate paths
        all_paths = [class_raster_path, *metrics_rasters.values(), vector_cities_path]
        self.raster_loader.validate_paths(all_paths)

        # Create output directories
        os.makedirs(self.config.outdir, exist_ok=True)
        stats_dir = os.path.join(self.config.outdir, "stats")
        os.makedirs(stats_dir, exist_ok=True)

        metrics_order = list(metrics_rasters.keys())

        # Main processing loop
        with self.raster_loader.load_classification_raster(
            class_raster_path
        ) as src_class, ExitStack() as stack:
            # Open metric rasters
            src_metrics = self.raster_loader.open_metric_rasters(metrics_rasters, stack)

            # Load and reproject cities
            gdf = self.vector_loader.load_cities(
                vector_cities_path, city_field, src_class.crs
            )

            combined_rows: List[pd.DataFrame] = []
            infer_rows: List[Dict] = []

            # Process each city
            for _, row in gdf.iterrows():
                city = str(row[city_field]).strip()

                if row.geometry is None or row.geometry.is_empty:
                    continue

                geom = [mapping(row.geometry)]

                # Clip classification
                try:
                    class_clip, class_transform = self.raster_loader.clip_classification(
                        src_class, geom
                    )
                except ValueError:
                    continue  # Skip cities outside raster bounds

                pixel_area_m2 = pixel_area_from_transform(class_transform)

                # Process each metric
                metric_stats: List[pd.DataFrame] = []
                raw_arrays: Dict[str, np.ndarray] = {}

                for metric_name, src_metric in src_metrics.items():
                    # Clip metric raster
                    metr = self.raster_loader.clip_metric_raster(
                        src_metric,
                        src_class,
                        geom,
                        class_transform,
                        class_clip.shape,
                    )

                    # Aggregate statistics
                    stats = self.aggregator.summarize_by_classes(metr, class_clip)

                    if not stats.empty:
                        stats = self.aggregator.add_total_kg(
                            stats, metric_name, class_transform, pixel_area_m2
                        )

                    metric_stats.append(stats)
                    raw_arrays[metric_name] = metr

                if not metric_stats:
                    continue

                # Merge metrics
                df_city = self.aggregator.merge_metric_stats(metric_stats)

                if df_city.empty:
                    continue

                # Add metadata
                df_city = self.aggregator.add_metadata(df_city, city, class_map)

                # Save city-specific CSV
                city_csv = os.path.join(self.config.outdir, f"{city}_stats_por_classe.csv")
                df_city.to_csv(city_csv, index=False)
                combined_rows.append(df_city)

                # Run inferential tests
                stats_annots: Dict[str, Dict] = {}
                if self.config.run_inferential_tests:
                    city_stats_dir = os.path.join(stats_dir, city.replace(" ", "_"))
                    os.makedirs(city_stats_dir, exist_ok=True)

                    for metric_name in metrics_order:
                        result = self.statistics.run_inferential_tests(
                            city,
                            metric_name,
                            raw_arrays[metric_name],
                            class_clip,
                            class_map,
                            city_stats_dir,
                        )
                        infer_rows.append(result)
                        stats_annots[metric_name] = result

                # Generate plots
                if self.config.make_plots:
                    label_col = (
                        "classe_nome" if "classe_nome" in df_city.columns else "classe"
                    )
                    self.graphics.generate_all_plots(
                        df_city,
                        metrics_order,
                        city,
                        label_col,
                        self.config.outdir,
                        stats_annots,
                        plot_types=self.config.plot_types,
                    )

            # Save combined outputs
            combined = self._save_combined_outputs(combined_rows, infer_rows)

            return combined

    def _save_combined_outputs(
        self, combined_rows: List[pd.DataFrame], infer_rows: List[Dict]
    ) -> pd.DataFrame:
        """Save combined statistics and inference results.

        Args:
            combined_rows: List of DataFrames with statistics per city
            infer_rows: List of dictionaries with inference results

        Returns:
            Combined statistics DataFrame
        """
        if combined_rows:
            combined = pd.concat(combined_rows, ignore_index=True)
            combined_path = os.path.join(
                self.config.outdir, "todas_cidades_stats_por_classe.csv"
            )
            combined.to_csv(combined_path, index=False)
            print(f"[OK] Combined output: {combined_path}")
        else:
            combined = pd.DataFrame()
            print("[Warning] No statistics generated.")

        if self.config.run_inferential_tests and infer_rows:
            infer_df = pd.DataFrame(infer_rows)
            infer_path = os.path.join(
                self.config.outdir, "stats", "resumo_inferencial_por_cidade.csv"
            )
            infer_df.to_csv(infer_path, index=False)
            print(f"[OK] Inferential summary: {infer_path}")

        return combined

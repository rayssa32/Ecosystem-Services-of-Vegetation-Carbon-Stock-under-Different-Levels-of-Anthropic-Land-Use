"""Main analysis pipeline orchestration."""

import os
from contextlib import ExitStack
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import mapping

from ..config import AnalysisConfig, SankeyConfig
from ..data.raster_loader import RasterLoader
from ..data.vector_loader import VectorLoader
from ..processing.aggregator import DataAggregator
from ..processing.biomass_classes import classify_by_quantiles, default_biomass_labels
from ..processing.statistics import StatisticsAnalyzer
from ..visualization.graphics_factory import GraphicsFactory
from ..visualization.plotter import ViolinPlotter
from ..visualization.sankey_plotter import build_flow_df, plot_sankey
from ..utils.raster_utils import pixel_area_from_transform
from ..utils.constants import CLASS_COLORS, NULL_LULC_CLASS


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
        metrics_rasters: Optional[Dict[str, str]] = None,
        vector_cities_path: str = None,
        city_field: str = "municipio",
        class_map: Optional[Dict[int, str]] = None,
        cities_filter: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Execute complete analysis pipeline.

        Args:
            class_raster_path: Path to classification raster
            metrics_rasters: Optional dictionary mapping metric names to file paths
            vector_cities_path: Path to cities shapefile
            city_field: Field name containing city names
            class_map: Optional mapping from class codes to names
            cities_filter: If set, only these municipalities are processed (None = all)

        Returns:
            Combined DataFrame with statistics from all cities
        """
        # Validate paths
        all_paths = [class_raster_path]
        if metrics_rasters:
            all_paths.extend(metrics_rasters.values())
        if vector_cities_path:
            all_paths.append(vector_cities_path)
        self.raster_loader.validate_paths(all_paths)

        # Create output directories
        os.makedirs(self.config.outdir, exist_ok=True)
        stats_dir = os.path.join(self.config.outdir, "stats")
        os.makedirs(stats_dir, exist_ok=True)

        metrics_order = list(metrics_rasters.keys()) if metrics_rasters else []

        # Main processing loop
        with self.raster_loader.load_classification_raster(
            class_raster_path
        ) as src_class, ExitStack() as stack:
            # Open metric rasters if provided
            src_metrics = {}
            if metrics_rasters:
                src_metrics = self.raster_loader.open_metric_rasters(metrics_rasters, stack)

            # Load and reproject cities
            if not vector_cities_path:
                raise ValueError("vector_cities_path is required")
            gdf = self.vector_loader.load_cities(
                vector_cities_path, city_field, src_class.crs
            )

            combined_rows: List[pd.DataFrame] = []
            infer_rows: List[Dict] = []

            # Process each city
            for _, row in gdf.iterrows():
                city = str(row[city_field]).strip()
                if cities_filter is not None and city not in cities_filter:
                    continue

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

                # If no metrics provided, calculate class area percentages only
                if not metrics_rasters:
                    # Calculate class area percentages
                    df_city = self.aggregator.calculate_class_area_percentages(class_clip)
                    if df_city.empty:
                        continue
                    # Add metadata
                    df_city = self.aggregator.add_metadata(df_city, city, class_map)
                    _excl = set(self.config.exclude_classes or [])
                    _excl.add(NULL_LULC_CLASS)
                    df_city = df_city[~df_city["classe"].isin(_excl)].copy()
                else:
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
                    _excl = set(self.config.exclude_classes or [])
                    _excl.add(NULL_LULC_CLASS)
                    df_city = df_city[~df_city["classe"].isin(_excl)].copy()

                # Save city-specific CSV if enabled
                if self.config.save_csv_files:
                    city_csv = os.path.join(self.config.outdir, f"{city}_stats_por_classe.csv")
                    df_city.to_csv(city_csv, index=False)
                combined_rows.append(df_city)

                # Run inferential tests (only if metrics are provided)
                stats_annots: Dict[str, Dict] = {}
                if self.config.run_inferential_tests and metrics_rasters:
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

                # Generate plots (only if metrics are provided)
                if self.config.make_plots and metrics_rasters:
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

            # Generate stacked bar charts if enabled
            if self.config.make_stacked_bar_charts and not combined.empty:
                # Use color constants from utils.constants
                class_colors = CLASS_COLORS.copy()

                # If no metrics, generate percentage-based stacked bar chart
                if not metrics_rasters:
                    plotter = self.graphics.create_stacked_bar_plotter(class_colors)
                    plotter.plot(
                        combined,
                        "Area",
                        self.config.outdir,
                        value_type="percentage",
                        normalize=False,  # Already normalized to 0-100%
                    )
                    print("[OK] Stacked bar chart (percentage area) generated")
                else:
                    self.graphics.generate_stacked_bar_charts(
                        combined,
                        metrics_order,
                        self.config.outdir,
                        value_type=self.config.stacked_bar_value_type,
                        normalize=self.config.stacked_bar_normalize,
                        class_colors=class_colors,
                    )
                    print("[OK] Stacked bar charts generated")

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
            if self.config.save_csv_files:
                combined_path = os.path.join(
                    self.config.outdir, "todas_cidades_stats_por_classe.csv"
                )
                combined.to_csv(combined_path, index=False)
                print(f"[OK] Combined output: {combined_path}")
        else:
            combined = pd.DataFrame()
            print("[Warning] No statistics generated.")

        if self.config.run_inferential_tests and infer_rows and self.config.save_csv_files:
            infer_df = pd.DataFrame(infer_rows)
            infer_path = os.path.join(
                self.config.outdir, "stats", "resumo_inferencial_por_cidade.csv"
            )
            infer_df.to_csv(infer_path, index=False)
            print(f"[OK] Inferential summary: {infer_path}")

        return combined

    def run_violin_plots_analysis(
        self,
        class_raster_path: str,
        biomass_raster_path: str,
        vector_cities_path: str,
        city_field: str = "municipio",
        class_map: Optional[Dict[int, str]] = None,
        cities_filter: Optional[List[str]] = None,
    ) -> None:
        """Run analysis to generate violin plots for biomass by land use class per city.

        Args:
            class_raster_path: Path to classification raster (LULC)
            biomass_raster_path: Path to biomass raster
            vector_cities_path: Path to cities shapefile
            city_field: Field name containing city names
            class_map: Optional mapping from class codes to names
            cities_filter: If set, only these cities are processed (e.g. ["Lavras"]). None = all.
        """
        # Validate paths
        all_paths = [class_raster_path, biomass_raster_path, vector_cities_path]
        self.raster_loader.validate_paths(all_paths)

        # Create output directory
        os.makedirs(self.config.outdir, exist_ok=True)

        # Collect all city data for combined plot
        city_data_list = []

        # Main processing loop
        with self.raster_loader.load_classification_raster(
            class_raster_path
        ) as src_class, rasterio.open(biomass_raster_path) as src_biomass:
            # Load and reproject cities
            gdf = self.vector_loader.load_cities(
                vector_cities_path, city_field, src_class.crs
            )

            # Process each city and collect data
            for _, row in gdf.iterrows():
                city = str(row[city_field]).strip()
                if cities_filter is not None and city not in cities_filter:
                    continue
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

                # Clip and reproject biomass to match classification grid
                biomass_clip = self.raster_loader.clip_metric_raster(
                    src_biomass,
                    src_class,
                    geom,
                    class_transform,
                    class_clip.shape,
                )

                # Run Kruskal-Wallis test
                test_results = self.statistics.run_kruskal_wallis_test(
                    biomass_clip, class_clip, class_map
                )

                # Collect data for combined plot
                city_data_list.append({
                    "city": city,
                    "values": biomass_clip,
                    "classes": class_clip,
                    "annotation": test_results,
                })

                print(f"[OK] Processed data for {city}")

            # Generate combined violin plot with all cities
            if city_data_list:
                plotter = self.graphics.create_plotter("violin")
                violin_plotter = plotter.plotter
                if isinstance(violin_plotter, ViolinPlotter):
                    violin_plotter.plot_combined_cities(
                        city_data_list,
                        "Biomassa",
                        class_map,
                        self.config.outdir,
                    )
                else:
                    print(f"[Warning] Expected ViolinPlotter, got {type(violin_plotter).__name__}")
            else:
                print("[Warning] No valid city data collected. No plots generated.")

    def run_sankey_analysis(
        self,
        class_raster_path: str,
        biomass_raster_path: str,
        vector_cities_path: str,
        city_field: str,
        class_map: Dict[int, str],
        cities_filter: Optional[List[str]] = None,
        sankey_config: Optional[SankeyConfig] = None,
    ) -> None:
        """Run Sankey diagram: land use → biomass class (quantile-based). Order: after violin/Kruskal.

        Uses same data as violin: read rasters, resample MODIS→Sentinel, pixel-level,
        then build flows (land use → biomass class) with thickness = count or % of pixels.
        """
        if sankey_config is None:
            sankey_config = SankeyConfig()

        all_paths = [class_raster_path, biomass_raster_path, vector_cities_path]
        self.raster_loader.validate_paths(all_paths)
        sankey_dir = os.path.join(self.config.outdir, "sankey")
        os.makedirs(sankey_dir, exist_ok=True)

        n_q = sankey_config.n_quantiles
        biomass_labels = default_biomass_labels(n_q)

        # Collect per-city data and pool biomass for global quantile edges
        city_data_list: List[tuple] = []

        with self.raster_loader.load_classification_raster(
            class_raster_path
        ) as src_class, rasterio.open(biomass_raster_path) as src_biomass:
            gdf = self.vector_loader.load_cities(
                vector_cities_path, city_field, src_class.crs
            )
            for _, row in gdf.iterrows():
                city = str(row[city_field]).strip()
                if cities_filter is not None and city not in cities_filter:
                    continue
                if row.geometry is None or row.geometry.is_empty:
                    continue
                geom = [mapping(row.geometry)]
                try:
                    class_clip, class_transform = self.raster_loader.clip_classification(
                        src_class, geom
                    )
                except ValueError:
                    continue
                biomass_clip = self.raster_loader.clip_metric_raster(
                    src_biomass,
                    src_class,
                    geom,
                    class_transform,
                    class_clip.shape,
                )
                valid = ~(np.isnan(class_clip) | np.isnan(biomass_clip))
                if not np.any(valid):
                    continue
                city_data_list.append((city, class_clip, biomass_clip))

        if not city_data_list:
            print("[Warning] No valid city data for Sankey. Skipping.")
            return

        # Pool biomass only over pixels whose LULC is not excluded (NULL, water, etc.) for quantile edges
        excl_q = {NULL_LULC_CLASS}
        if self.config.exclude_classes:
            excl_q.update(int(x) for x in self.config.exclude_classes)
        pooled_parts: List[np.ndarray] = []
        for _, class_clip, biomass_clip in city_data_list:
            c = np.asarray(class_clip).ravel().astype(int)
            b = np.asarray(biomass_clip, dtype=np.float64).ravel()
            m = ~(np.isnan(np.asarray(class_clip, dtype=np.float64).ravel()) | np.isnan(b))
            for code in excl_q:
                m &= c != code
            pooled_parts.append(b[m])
        pooled = np.concatenate(pooled_parts) if pooled_parts else np.array([])
        if pooled.size == 0:
            print("[Warning] No valid biomass for Sankey after LULC exclusions. Skipping.")
            return
        pooled = np.asarray(pooled, dtype=np.float64)
        edges = np.nanquantile(
            pooled,
            np.linspace(0, 1, n_q + 1)[1:-1],
        )

        def _make_flow_and_plot(city_label: str, class_arr: np.ndarray, biomass_arr: np.ndarray, base_name: str) -> None:
            biomass_class, _ = classify_by_quantiles(biomass_arr, n_quantiles=n_q, edges=edges)
            lu_flat = class_arr.ravel()
            bc_flat = biomass_class.ravel()
            flow_df = build_flow_df(
                lu_flat,
                bc_flat,
                class_map,
                biomass_labels,
                use_percentage=sankey_config.use_percentage,
                exclude_land_use_classes=self.config.exclude_classes,
            )
            if flow_df.empty:
                return
            value_label = "% de pixels" if sankey_config.use_percentage else "pixels"
            outpath = os.path.join(sankey_dir, base_name)
            plot_sankey(
                flow_df,
                outpath,
                title=f"Uso do solo → classe de biomassa{f' — {city_label}' if city_label else ''}",
                value_label=value_label,
            )
            print(f"[OK] Sankey saved: {outpath}.html")

        if sankey_config.per_city:
            for city, class_clip, biomass_clip in city_data_list:
                safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in city)
                _make_flow_and_plot(city, class_clip, biomass_clip, f"sankey_{safe_name}")
        else:
            # One combined Sankey (all cities)
            class_all = np.concatenate([c.ravel() for _, c, _ in city_data_list])
            biomass_all = np.concatenate([b.ravel() for _, _, b in city_data_list])
            _make_flow_and_plot("", class_all, biomass_all, "sankey_all_cities")

    def run_shannon_index_analysis(
        self,
        class_raster_path: str,
        vector_cities_path: str,
        city_field: str = "municipio",
        cities_filter: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compute Shannon (H') and Pielou equitability (J') per municipality; save CSV.

        H' = -Σ p_i ln(p_i); J' = H' / ln(S) with S = number of LULC classes (richness).
        NULL is always excluded; additional exclusions follow ``config.exclude_classes``.
        """
        all_paths = [class_raster_path, vector_cities_path]
        self.raster_loader.validate_paths(all_paths)
        os.makedirs(self.config.outdir, exist_ok=True)

        excl: Set[int] = set()
        if self.config.exclude_classes:
            excl.update(int(x) for x in self.config.exclude_classes)

        rows: List[Dict[str, object]] = []

        with self.raster_loader.load_classification_raster(
            class_raster_path
        ) as src_class:
            gdf = self.vector_loader.load_cities(
                vector_cities_path, city_field, src_class.crs
            )
            for _, row in gdf.iterrows():
                city = str(row[city_field]).strip()
                if cities_filter is not None and city not in cities_filter:
                    continue
                if row.geometry is None or row.geometry.is_empty:
                    continue
                geom = [mapping(row.geometry)]
                try:
                    class_clip, _ = self.raster_loader.clip_classification(
                        src_class, geom
                    )
                except ValueError:
                    continue

                stats = self.aggregator.shannon_entropy_land_cover(
                    class_clip, exclude_classes=excl
                )
                rows.append(
                    {
                        "cidade": city,
                        "shannon_H": stats["shannon_H"],
                        "equitability_J": stats["equitability_J"],
                        "n_classes": stats["n_classes"],
                        "n_pixels": stats["n_pixels"],
                    }
                )

        df = pd.DataFrame(rows)
        if not df.empty and self.config.save_csv_files:
            out_csv = os.path.join(
                self.config.outdir, "shannon_index_por_cidade.csv"
            )
            df.to_csv(out_csv, index=False)
            print(f"[OK] Shannon / equitability: {out_csv}")
        elif df.empty:
            print("[Warning] No Shannon index rows (no cities or no valid pixels).")
        return df

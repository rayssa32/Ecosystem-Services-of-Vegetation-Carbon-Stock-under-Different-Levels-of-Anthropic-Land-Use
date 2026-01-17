"""
Analysis pipeline for ecosystem services: city-based analysis with inferential statistics.

Refactored using clean architecture principles with modular design focused on extensible
graphics generation capabilities.

Usage:
    python main.py
"""

from src import AnalysisConfig, AnalysisPipeline


def main() -> None:
    """Main entry point for the analysis pipeline."""
    # >>>>>>>>>>>> CONFIGURE YOUR PATHS HERE <<<<<<<<<<<<
    class_raster_path = "classificacao/LULC_7_cidades_2025-07-10_2025-07-30_projected.tif"
    # No metric rasters - using only classification to calculate area percentages
    metrics_rasters = None
    vector_cities_path = "shapefile/sete_cidades.shp"
    city_field = "NM_MUN"
    # New classification: 0-Água | 1-Urbano | 2-Solo | 3-Vegetação | 4-Agro/Pasto
    # Colors: ['#3b83bd', '#8c8c8c', '#c8a165', '#2ca25f', '#a1d99b']
    class_map = {
        0: "Água",
        1: "Urbano",
        2: "Solo",
        3: "Vegetação",
        4: "Agro/Pasto",
    }

    # Configure analysis parameters
    config = AnalysisConfig(
        resample_metrics="nearest",
        make_plots=False,  # Disable regular plots (not using metrics)
        outdir="./dados_gerados",
        plot_types=["bar"],  # Not used when make_plots=False
        # Stacked bar chart configuration
        make_stacked_bar_charts=True,  # Enable stacked bar charts with percentage area
        stacked_bar_value_type="percentage",  # Use percentage values (0-100%)
        stacked_bar_normalize=False,  # Already normalized to 0-100%
        save_csv_files=False,  # Disable CSV generation (only generate graph images)
        run_inferential_tests=False,  # No metrics to test
        exclude_classes=[],  # No classes to exclude with new classification scheme
        sample_per_class=5000,
        min_n_for_tests=10,
        alpha=0.05,
        rng_seed=42,
    )

    # Initialize and run pipeline
    pipeline = AnalysisPipeline(config)
    pipeline.run(
        class_raster_path=class_raster_path,
        metrics_rasters=metrics_rasters,
        vector_cities_path=vector_cities_path,
        city_field=city_field,
        class_map=class_map,
    )


if __name__ == "__main__":
    main()

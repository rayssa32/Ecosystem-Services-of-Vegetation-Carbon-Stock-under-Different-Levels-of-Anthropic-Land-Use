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
    biomass_raster_path = "metricas/Biomass_sete_cidades_projected.tif"  # Reprojected to EPSG:32728
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
        resample_metrics="nearest",  # Nearest neighbor resampling
        make_plots=False,  # Not using regular plots
        outdir="./dados_gerados",
        plot_types=["violin"],  # Using violin plots
        save_csv_files=False,  # Disable CSV generation (only generate graph images)
        run_inferential_tests=False,  # Tests run within violin plot generation
        exclude_classes=[0],  # Exclude water class (Água) from violin plots
        sample_per_class=5000,  # Sample size for statistical tests
        min_n_for_tests=10,  # Minimum observations per class
        alpha=0.05,  # Significance level
        rng_seed=42,  # Reproducibility
    )

    # Initialize and run pipeline for violin plots
    pipeline = AnalysisPipeline(config)
    pipeline.run_violin_plots_analysis(
        class_raster_path=class_raster_path,
        biomass_raster_path=biomass_raster_path,
        vector_cities_path=vector_cities_path,
        city_field=city_field,
        class_map=class_map,
    )


if __name__ == "__main__":
    main()

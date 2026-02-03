"""Configuration settings for the analysis pipeline."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PathsConfig:
    """Paths comuns a main.py e run_moran.py."""

    class_raster_path: str = "classificacao/LULC_7_cidades_2025-07-10_2025-07-30_projected.tif"
    biomass_raster_path: str = "metricas/Biomass_sete_cidades_projected.tif"
    vector_cities_path: str = "shapefile/sete_cidades.shp"
    city_field: str = "NM_MUN"
    outdir: str = "./dados_gerados"


@dataclass
class MoranConfig:
    """Configuração do Moran's I (run_moran.py)."""

    use_native_resolution: bool = True
    cities_filter: Optional[List[str]] = None  # None = todas
    permutations: int = 999
    contiguity: str = "rook"  # rook | queen
    save_scatter_plots: bool = True


@dataclass
class AnalysisConfig:
    """Configuration parameters for statistical analysis and visualization."""

    # Resampling configuration
    resample_metrics: str = "nearest"  # nearest | bilinear | cubic | average | ...

    # Output configuration
    make_plots: bool = True
    outdir: str = "./dados_gerados"
    plot_types: List[str] = None  # List of plot types: "bar", "box", "violin"
    make_stacked_bar_charts: bool = True  # Generate stacked bar charts comparing cities
    stacked_bar_value_type: str = "mean"  # Value type for stacked bars: "mean", "sum", "count", "total_kg"
    stacked_bar_normalize: bool = False  # Normalize stacked bars to percentages
    save_csv_files: bool = True  # Save CSV files with statistics (set False to generate only images)

    # Statistical tests configuration
    run_inferential_tests: bool = True
    exclude_classes: List[int] = None
    sample_per_class: int = 5000
    min_n_for_tests: int = 10
    alpha: float = 0.05
    rng_seed: int = 42

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.exclude_classes is None:
            self.exclude_classes = [5]
        if self.plot_types is None:
            self.plot_types = ["bar"]


DEFAULT_CONFIG = AnalysisConfig()
DEFAULT_PATHS = PathsConfig()
DEFAULT_MORAN_CONFIG = MoranConfig()

"""Configuration settings for the analysis pipeline."""

from dataclasses import dataclass
from typing import List


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


# Global default configuration instance
DEFAULT_CONFIG = AnalysisConfig()

"""Factory for creating different types of graphics."""

from typing import Dict, List, Optional

import pandas as pd

from ..config import AnalysisConfig
from .plotter import (
    Plotter,
    BasePlotter,
    BarPlotter,
    BoxPlotter,
    ViolinPlotter,
    StackedBarPlotter,
)


class GraphicsFactory:
    """Factory for generating multiple types of graphics from data."""

    def __init__(self, config: AnalysisConfig):
        """Initialize graphics factory.

        Args:
            config: Analysis configuration object
        """
        self.config = config

    def create_plotter(self, plot_type: str = "bar") -> Plotter:
        """Create a plotter instance for a specific plot type.

        Args:
            plot_type: Type of plot ("bar", "box", "violin")

        Returns:
            Plotter instance
        """
        return Plotter(self.config, plot_type=plot_type)

    def generate_all_plots(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        city: str,
        label_col: str,
        outdir: str,
        annotations: Optional[Dict[str, Dict]] = None,
        plot_types: Optional[List[str]] = None,
    ) -> None:
        """Generate all requested plot types for all metrics.

        Args:
            df: DataFrame with statistics by class
            metrics: List of metric names to plot
            city: City name
            label_col: Column name for class labels
            outdir: Output directory
            annotations: Optional dictionary mapping metric names to annotation dicts
            plot_types: List of plot types to generate (default: ["bar"])
        """
        if plot_types is None:
            plot_types = ["bar"]

        if annotations is None:
            annotations = {}

        for plot_type in plot_types:
            plotter = self.create_plotter(plot_type)
            for metric in metrics:
                annotation = annotations.get(metric)
                plotter.plot(df, metric, city, label_col, outdir, annotation)

    def create_comparison_plot(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        city: str,
        outdir: str,
        plot_type: str = "bar",
    ) -> None:
        """Create a multi-metric comparison plot.

        Args:
            df: DataFrame with statistics for multiple metrics
            metrics: List of metric names to include
            city: City name
            outdir: Output directory
            plot_type: Type of plot to generate
        """
        # TODO: Implement multi-metric comparison visualization
        # This would show multiple metrics side-by-side for comparison
        pass

    def create_stacked_bar_plotter(
        self, class_colors: Optional[Dict[int, str]] = None
    ) -> StackedBarPlotter:
        """Create a stacked bar plotter instance.

        Args:
            class_colors: Optional dictionary mapping class codes to hex colors

        Returns:
            StackedBarPlotter instance
        """
        return StackedBarPlotter(self.config, class_colors)

    def generate_stacked_bar_charts(
        self,
        combined_df: pd.DataFrame,
        metrics: List[str],
        outdir: str,
        value_type: str = "mean",
        normalize: bool = False,
        class_colors: Optional[Dict[int, str]] = None,
    ) -> None:
        """Generate stacked bar charts for all metrics.

        Args:
            combined_df: Combined DataFrame with statistics from all cities
            metrics: List of metric names to plot
            outdir: Output directory
            value_type: Type of value to plot ("mean", "sum", "count", "total_kg")
            normalize: If True, normalize to percentages (0-100), otherwise use absolute values
            class_colors: Optional dictionary mapping class codes to hex colors
        """
        plotter = self.create_stacked_bar_plotter(class_colors)
        for metric in metrics:
            plotter.plot(combined_df, metric, outdir, value_type, normalize)

    def register_custom_plotter(self, plot_type: str, plotter: BasePlotter) -> None:
        """Register a custom plotter type for extensibility.

        Args:
            plot_type: Name identifier for the plot type
            plotter: Custom plotter instance

        Note:
            This allows users to extend the system with custom plot types
        """
        # TODO: Implement custom plotter registration
        pass

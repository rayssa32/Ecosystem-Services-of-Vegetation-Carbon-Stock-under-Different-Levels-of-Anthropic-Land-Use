"""Base plotting interface and concrete implementations."""

import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..config import AnalysisConfig
from ..utils.constants import CLASS_COLORS, DEFAULT_CLASS_COLORS


class BasePlotter(ABC):
    """Abstract base class for plot generators."""

    @abstractmethod
    def plot(
        self,
        df: pd.DataFrame,
        metric: str,
        city: str,
        label_col: str,
        outdir: str,
        annotation: Optional[Dict] = None,
    ) -> None:
        """Generate and save a plot.

        Args:
            df: DataFrame with data to plot
            metric: Name of the metric being plotted
            city: City name for title and filename
            label_col: Column name for x-axis labels
            outdir: Output directory for saved plot
            annotation: Optional statistical annotation dictionary
        """
        pass


class BarPlotter(BasePlotter):
    """Generates bar plots with statistical annotations."""

    def __init__(self, config: AnalysisConfig):
        """Initialize bar plotter.

        Args:
            config: Analysis configuration object
        """
        self.config = config

    def plot(
        self,
        df: pd.DataFrame,
        metric: str,
        city: str,
        label_col: str,
        outdir: str,
        annotation: Optional[Dict] = None,
    ) -> None:
        """Generate bar plot with mean values and error bars.

        Args:
            df: DataFrame with statistics by class
            metric: Name of the metric
            city: City name
            label_col: Column name for class labels
            outdir: Output directory
            annotation: Optional statistical test results
        """
        labels = (
            df[label_col].astype(str).values
            if label_col in df.columns
            else df["classe"].astype(str).values
        )

        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        if mean_col not in df.columns:
            return

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(
            x,
            df[mean_col].values,
            yerr=df.get(std_col, None),
            capsize=4,
            alpha=0.8,
            color="steelblue",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f"{city}: Média de {metric} por classe", fontsize=14, fontweight="bold")
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Add statistical annotation if provided
        if annotation is not None and "p_global" in annotation:
            self._add_statistical_annotation(ax, annotation)

        plt.margins(x=0.02)
        plt.tight_layout()
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"{city}_{metric}_means.png"), dpi=200, bbox_inches="tight"
        )
        plt.close()

    def _add_statistical_annotation(self, ax: plt.Axes, annotation: Dict) -> None:
        """Add statistical test results as annotation box.

        Args:
            ax: Matplotlib axes object
            annotation: Dictionary with statistical test results
        """
        test_name = annotation.get("teste_global", "")
        p = annotation.get("p_global", np.nan)
        eff = annotation.get("efeito", np.nan)

        # Format p-value
        p_txt = (
            "p < 0.001"
            if isinstance(p, float) and p < 0.001
            else (f"p = {p:.3f}" if isinstance(p, float) else "p = n/a")
        )

        # Format effect size
        eff_sym = "η²" if ("ANOVA" in str(test_name)) else "ε²"
        eff_txt = (
            f"{eff_sym} = {eff:.2f}"
            if isinstance(eff, float) and not np.isnan(eff)
            else f"{eff_sym} = n/a"
        )

        # Add significance indicator
        sig = "★" if (isinstance(p, float) and p < self.config.alpha) else ""

        box_txt = f"{test_name}\n{p_txt}   {eff_txt}  {sig}"

        ax.text(
            0.98,
            0.98,
            box_txt,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray", lw=1),
            fontsize=10,
        )


class BoxPlotter(BasePlotter):
    """Generates box plots for comparing distributions across classes."""

    def plot(
        self,
        df: pd.DataFrame,
        metric: str,
        city: str,
        label_col: str,
        outdir: str,
        annotation: Optional[Dict] = None,
    ) -> None:
        """Generate box plot comparing distributions across classes.

        Note: This requires raw data, not just summary statistics.
        Currently a placeholder for future implementation.

        Args:
            df: DataFrame with data (should contain raw values)
            metric: Name of the metric
            city: City name
            label_col: Column name for class labels
            outdir: Output directory
            annotation: Optional statistical test results
        """
        # TODO: Implement box plot using raw data arrays
        # This requires passing raw data arrays to the plotter
        pass


class ViolinPlotter(BasePlotter):
    """Generates violin plots for comparing distributions across classes."""

    def plot(
        self,
        df: pd.DataFrame,
        metric: str,
        city: str,
        label_col: str,
        outdir: str,
        annotation: Optional[Dict] = None,
    ) -> None:
        """Generate violin plot comparing distributions across classes.

        Note: This requires raw data, not just summary statistics.
        Currently a placeholder for future implementation.

        Args:
            df: DataFrame with data (should contain raw values)
            metric: Name of the metric
            city: City name
            label_col: Column name for class labels
            outdir: Output directory
            annotation: Optional statistical test results
        """
        # TODO: Implement violin plot using raw data arrays
        pass


class StackedBarPlotter:
    """Generates stacked bar charts comparing classes across cities.
    
    NOTE: This plotter intentionally has a different interface than BasePlotter
    because it operates on combined data from multiple cities, whereas
    BasePlotter implementations work on single-city data. This architectural
    separation is intentional for clarity and reflects different use cases.
    """

    def __init__(self, config: AnalysisConfig, class_colors: Optional[Dict[int, str]] = None):
        """Initialize stacked bar plotter.

        Args:
            config: Analysis configuration object
            class_colors: Optional dictionary mapping class codes to hex colors.
                         If not provided, uses CLASS_COLORS constant.
        """
        self.config = config
        self.class_colors = class_colors or CLASS_COLORS.copy()

    def plot(
        self,
        combined_df: pd.DataFrame,
        metric: str,
        outdir: str,
        value_type: str = "mean",
        normalize: bool = False,
    ) -> None:
        """Generate stacked bar chart with cities on X-axis and classes stacked on Y-axis.

        Args:
            combined_df: Combined DataFrame with statistics from all cities
            metric: Name of the metric to plot
            outdir: Output directory
            value_type: Type of value to plot ("mean", "sum", "count", "total_kg", "percentage")
            normalize: If True, normalize to percentages (0-100), otherwise use absolute values
        """
        # Validate input
        if not self._validate_input(combined_df, metric, value_type):
            return

        # Prepare data
        value_col, label_col = self._determine_columns(combined_df, metric, value_type)
        pivot_df = self._prepare_pivot_data(combined_df, value_col, label_col, normalize)

        # Get colors for classes
        colors = self._get_class_colors(pivot_df.columns, combined_df, label_col)

        # Create and configure plot
        fig, ax = self._create_plot(pivot_df, colors)
        self._configure_plot_axes(ax, metric, value_type, normalize)
        self._add_legend(ax)

        # Save plot
        self._save_plot(fig, outdir, metric, value_type, normalize)

    def _validate_input(
        self, combined_df: pd.DataFrame, metric: str, value_type: str
    ) -> bool:
        """Validate input DataFrame and columns.

        Args:
            combined_df: DataFrame to validate
            metric: Metric name
            value_type: Value type

        Returns:
            True if valid, False otherwise
        """
        if "cidade" not in combined_df.columns:
            print("[Warning] 'cidade' column not found. Cannot generate stacked bar chart.")
            return False
        return True

    def _determine_columns(
        self, combined_df: pd.DataFrame, metric: str, value_type: str
    ) -> Tuple[str, str]:
        """Determine value and label columns from DataFrame.

        Args:
            combined_df: Combined DataFrame
            metric: Metric name
            value_type: Value type

        Returns:
            Tuple of (value_column, label_column)

        Raises:
            ValueError: If required column not found
        """
        # Determine value column
        if value_type == "percentage":
            value_col = "percentage"
        elif value_type == "total_kg":
            value_col = f"{metric}_total_kg"
        else:
            value_col = f"{metric}_{value_type}"

        if value_col not in combined_df.columns:
            available = list(combined_df.columns)
            print(
                f"[Warning] Column '{value_col}' not found. "
                f"Available columns: {available}"
            )
            raise ValueError(f"Column '{value_col}' not found in DataFrame")

        # Determine label column
        label_col = (
            "classe_nome" if "classe_nome" in combined_df.columns else "classe"
        )

        return value_col, label_col

    def _prepare_pivot_data(
        self,
        combined_df: pd.DataFrame,
        value_col: str,
        label_col: str,
        normalize: bool,
    ) -> pd.DataFrame:
        """Prepare pivoted data for plotting.

        Args:
            combined_df: Combined DataFrame with statistics from all cities
            value_col: Column name for values
            label_col: Column name for class labels
            normalize: Whether to normalize to percentages

        Returns:
            Pivoted DataFrame with cities as index and classes as columns
        """
        # Pivot data: cities as index, classes as columns
        pivot_df = combined_df.pivot_table(
            index="cidade",
            columns=label_col,
            values=value_col,
            aggfunc="sum",
            fill_value=0,
        )

        # Sort cities alphabetically
        pivot_df = pivot_df.sort_index()

        # Normalize if requested (convert to percentages)
        if normalize:
            pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

        return pivot_df

    def _get_class_colors(
        self,
        class_order: List[str],
        combined_df: pd.DataFrame,
        label_col: str,
    ) -> List[str]:
        """Get colors for each class in the specified order.

        Args:
            class_order: Ordered list of class labels
            combined_df: Original DataFrame to lookup class codes
            label_col: Column name for class labels

        Returns:
            List of hex color codes in same order as class_order
        """
        colors = []

        for i, cls in enumerate(class_order):
            # Try to get class code from the original DataFrame
            class_code = None
            if "classe" in combined_df.columns:
                mask = combined_df[label_col] == cls
                if mask.any():
                    sample = combined_df[mask]["classe"].iloc[0]
                    if pd.notna(sample):
                        class_code = int(sample)

            # Use color from mapping if available, otherwise use default
            if class_code is not None and class_code in self.class_colors:
                colors.append(self.class_colors[class_code])
            else:
                colors.append(DEFAULT_CLASS_COLORS[i % len(DEFAULT_CLASS_COLORS)])

        return colors

    def _create_plot(self, pivot_df: pd.DataFrame, colors: List[str]) -> Tuple:
        """Create the matplotlib figure and axes with stacked bar chart.

        Args:
            pivot_df: Pivoted DataFrame with cities and classes
            colors: List of colors for each class

        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # Create stacked bar chart
        pivot_df.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=colors,
            width=0.8,
            edgecolor="white",
            linewidth=0.5,
        )

        return fig, ax

    def _configure_plot_axes(
        self, ax: plt.Axes, metric: str, value_type: str, normalize: bool
    ) -> None:
        """Configure plot axes, labels, and title.

        Args:
            ax: Matplotlib axes object
            metric: Metric name
            value_type: Value type
            normalize: Whether values are normalized
        """
        # Set y-axis label
        if value_type == "percentage":
            ylabel = "Area Coverage (%)"
        else:
            ylabel = f"{metric} ({value_type})"
            if normalize:
                ylabel = f"{metric} ({value_type}) - Percentage (%)"

        ax.set_xlabel("City", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")

        # Set title
        if value_type == "percentage":
            title = "Stacked Bar Chart: Land Use Class Coverage (%) by City"
        else:
            title = f"Stacked Bar Chart: {metric} by City and Land Use Class"

        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha="right")

        # Add grid
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    def _add_legend(self, ax: plt.Axes) -> None:
        """Add legend to the plot.

        Args:
            ax: Matplotlib axes object
        """
        ax.legend(
            title="Land Use Class",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            frameon=True,
            fancybox=True,
            shadow=True,
        )

    def _save_plot(
        self, fig: plt.Figure, outdir: str, metric: str, value_type: str, normalize: bool
    ) -> None:
        """Save the plot to file.

        Args:
            fig: Matplotlib figure object
            outdir: Output directory
            metric: Metric name for filename
            value_type: Value type for filename
            normalize: Whether values are normalized (for filename)
        """
        os.makedirs(outdir, exist_ok=True)

        # Determine filename
        if value_type == "percentage":
            filename = "stacked_bar_land_use_percentage.png"
        else:
            filename = f"stacked_bar_{metric}_{value_type}"
            if normalize:
                filename += "_normalized"
            filename += ".png"

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, filename), dpi=200, bbox_inches="tight")
        plt.close()


class Plotter:
    """Main plotting interface that delegates to specific plotter types."""

    def __init__(self, config: AnalysisConfig, plot_type: str = "bar"):
        """Initialize plotter with specific plot type.

        Args:
            config: Analysis configuration object
            plot_type: Type of plot to generate ("bar", "box", "violin")
        """
        self.config = config
        self.plotter = self._create_plotter(plot_type)

    def _create_plotter(self, plot_type: str) -> BasePlotter:
        """Factory method to create appropriate plotter.

        Args:
            plot_type: Type of plot ("bar", "box", "violin")

        Returns:
            Appropriate plotter instance

        Raises:
            ValueError: If plot_type is not supported
        """
        if plot_type == "bar":
            return BarPlotter(self.config)
        elif plot_type == "box":
            return BoxPlotter()
        elif plot_type == "violin":
            return ViolinPlotter()
        elif plot_type == "stacked_bar":
            # StackedBarPlotter uses a different interface, handled separately
            raise ValueError(
                "StackedBarPlotter must be created directly via GraphicsFactory.create_stacked_bar_plotter()"
            )
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

    def plot(
        self,
        df: pd.DataFrame,
        metric: str,
        city: str,
        label_col: str,
        outdir: str,
        annotation: Optional[Dict] = None,
    ) -> None:
        """Generate and save plot using the configured plotter.

        Args:
            df: DataFrame with data to plot
            metric: Name of the metric
            city: City name
            label_col: Column name for class labels
            outdir: Output directory
            annotation: Optional statistical test results
        """
        self.plotter.plot(df, metric, city, label_col, outdir, annotation)

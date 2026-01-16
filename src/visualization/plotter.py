"""Base plotting interface and concrete implementations."""

import os
from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..config import AnalysisConfig


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

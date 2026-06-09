"""Base plotting interface and concrete implementations."""

import os
import re
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import kruskal

from ..config import AnalysisConfig
from ..utils.constants import (
    CLASS_COLORS,
    DEFAULT_CLASS_COLORS,
    LULC_LEGEND_ORDER,
    NULL_LULC_CLASS,
    rotulo_metrica,
    rotulo_tipo_valor,
)


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
        metric_label = rotulo_metrica(metric)
        ax.set_title(
            f"{city}: média de {metric_label} por classe de uso do solo",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylabel(metric_label, fontsize=12)
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
        if isinstance(p, np.generic):
            p = float(p)
        if isinstance(eff, np.generic):
            eff = float(eff)

        # Format p-value
        p_txt = (
            "p < 0.001"
            if isinstance(p, float) and p < 0.001
            else (f"p = {p:.3f}" if isinstance(p, float) else "p = n/d")
        )

        # Format effect size
        eff_sym = "η²" if ("ANOVA" in str(test_name)) else "ε²"
        eff_txt = (
            f"{eff_sym} = {eff:.2f}"
            if isinstance(eff, float) and not np.isnan(eff)
            else f"{eff_sym} = n/d"
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


class ViolinPlotter(BasePlotter):
    """Generates violin plots for comparing distributions across classes."""

    _plot_extension = "violin"

    def __init__(self, config: AnalysisConfig):
        """Initialize violin plotter.

        Args:
            config: Analysis configuration object
        """
        self.config = config

    def _metric_axis_label(self, metric: str) -> str:
        """Y-axis label; reflects biomass → carbon conversion when configured."""
        fraction = getattr(self.config, "biomass_carbon_fraction", 1.0)
        if metric == "Carbono":
            return f"Carbono (biomassa × {fraction:g})" if fraction != 1.0 else "Carbono"
        if metric in ("Biomassa", "Biomass") and fraction != 1.0:
            return f"Carbono (biomassa × {fraction:g})"
        return rotulo_metrica(metric)

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

        Note: This requires raw data arrays, not summary statistics.
        Use plot_with_raw_data() method instead.

        Args:
            df: DataFrame with data (not used, kept for interface compatibility)
            metric: Name of the metric
            city: City name
            label_col: Column name for class labels
            outdir: Output directory
            annotation: Optional statistical test results
        """
        # This method is kept for interface compatibility
        # Use plot_with_raw_data() for actual implementation
        pass

    def plot_with_raw_data(
        self,
        values: np.ndarray,
        classes: np.ndarray,
        metric: str,
        city: str,
        class_map: Optional[Dict[int, str]],
        outdir: str,
        annotation: Optional[Dict] = None,
    ) -> None:
        """Generate violin plot with raw data arrays.

        Args:
            values: Raw metric values array
            classes: Classification array
            metric: Name of the metric
            city: City name
            class_map: Optional mapping from class codes to names
            outdir: Output directory
            annotation: Optional statistical test results (Kruskal-Wallis, effect size)
        """
        # Prepare data for plotting
        data_by_class, labels = self._prepare_data_for_plotting(
            values, classes, class_map
        )

        if not data_by_class:
            print(f"[Warning] No valid data for {city} - {metric}. Skipping plot.")
            return

        # Create the plot — one column for the city, classes overlaid in the same violin
        fig, ax = plt.subplots(figsize=(8, 8))

        self._plot_stacked_distributions(ax, data_by_class, labels, class_map, position=0)

        metric_label = self._metric_axis_label(metric)
        ax.set_xticks([0])
        ax.set_xticklabels([city], rotation=45, ha="right")
        ax.set_xlabel("Cidade", fontsize=12, fontweight="bold")
        ax.set_ylabel(metric_label, fontsize=12, fontweight="bold")
        ax.set_title(
            f"{city}: distribuição de {metric_label} por classe de uso do solo",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        self._add_class_legend(ax, labels, class_map)

        # Add statistical annotation
        if annotation is not None:
            self._add_statistical_annotation(ax, annotation)

        # Add grid
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        # Adjust layout and save
        plt.tight_layout()
        os.makedirs(outdir, exist_ok=True)
        filename = f"{city}_{metric}_{self._plot_extension}.png"
        plt.savefig(os.path.join(outdir, filename), dpi=200, bbox_inches="tight")
        plt.close()

    def plot_combined_cities(
        self,
        city_data_list: List[Dict],
        metric: str,
        class_map: Optional[Dict[int, str]],
        outdir: str,
    ) -> None:
        """Generate one distribution plot per land-use class, comparing all cities.

        Args:
            city_data_list: List of dictionaries, each containing:
                - 'values': np.ndarray of metric values
                - 'classes': np.ndarray of classification codes
                - 'city': str city name
                - 'annotation': Optional dict with statistical test results
            metric: Name of the metric
            class_map: Optional mapping from class codes to names
            outdir: Output directory
        """
        if not city_data_list:
            print("[Warning] No city data provided for combined plot. Skipping.")
            return

        class_city_data: Dict[str, List[Tuple[str, np.ndarray]]] = {}
        all_labels: List[str] = []

        for city_data in city_data_list:
            city = city_data["city"]
            data_by_class, labels = self._prepare_data_for_plotting(
                city_data["values"], city_data["classes"], class_map
            )
            all_labels = self._merge_class_labels(all_labels, labels)
            for label in labels:
                class_city_data.setdefault(label, []).append(
                    (city, data_by_class[label])
                )

        if not all_labels:
            print("[Warning] No valid class data for combined plot. Skipping.")
            return

        os.makedirs(outdir, exist_ok=True)
        metric_label = self._metric_axis_label(metric)
        city_order = [city_data["city"] for city_data in city_data_list]

        class_panels: List[Tuple[str, List[Tuple[str, Optional[np.ndarray]]]]] = []
        for label in all_labels:
            entries_map = dict(class_city_data.get(label, []))
            aligned_entries = [(city, entries_map.get(city)) for city in city_order]
            if any(values is not None and len(values) > 0 for _, values in aligned_entries):
                class_panels.append((label, aligned_entries))

        if not class_panels:
            return

        self._plot_all_classes_in_one_figure(
            class_panels,
            metric,
            metric_label,
            class_map,
            outdir,
            city_order,
        )

    def _plot_all_classes_in_one_figure(
        self,
        class_panels: List[Tuple[str, List[Tuple[str, Optional[np.ndarray]]]]],
        metric: str,
        metric_label: str,
        class_map: Optional[Dict[int, str]],
        outdir: str,
        city_order: List[str],
    ) -> None:
        """Save all land-use class panels in a single vertically stacked image."""
        n_classes = len(class_panels)
        n_cities = len(city_order)
        fig, axes = plt.subplots(
            n_classes,
            1,
            figsize=(max(14, 2.5 * n_cities), 5 * n_classes),
            sharex=True,
            layout="constrained",
        )
        if n_classes == 1:
            axes = [axes]

        for idx, (ax, (class_label, city_entries)) in enumerate(
            zip(axes, class_panels)
        ):
            self._draw_class_panel_on_ax(
                ax,
                city_entries,
                class_label,
                metric_label,
                class_map,
                show_xlabel=(idx == n_classes - 1),
            )
            if idx < n_classes - 1:
                ax.tick_params(labelbottom=False)

        fig.suptitle(
            f"Distribuição de {metric_label} por classe de uso do solo — todas as cidades",
            fontsize=16,
            fontweight="bold",
        )
        filename = f"all_classes_{metric}_{self._plot_extension}_by_class.png"
        plt.savefig(os.path.join(outdir, filename), dpi=200)
        plt.close()
        print(f"[OK] {self._plot_extension} plot generated: {filename}")

    def _draw_class_panel_on_ax(
        self,
        ax: plt.Axes,
        city_entries: List[Tuple[str, Optional[np.ndarray]]],
        class_label: str,
        metric_label: str,
        class_map: Optional[Dict[int, str]],
        show_xlabel: bool = True,
    ) -> None:
        """Draw one land-use class panel with one distribution per city."""
        color = self._get_class_colors_for_labels([class_label], class_map)[0]
        city_names: List[str] = []
        groups_for_test: List[np.ndarray] = []

        for idx, (city, values) in enumerate(city_entries):
            city_names.append(city)
            if values is None or len(values) == 0:
                continue
            self._plot_single_distribution(ax, values, color, idx)
            groups_for_test.append(values)

        ax.set_xlim(-0.5, len(city_names) - 0.5)
        ax.set_xticks(np.arange(len(city_names)))
        ax.set_xticklabels(city_names, rotation=45, ha="right")
        if show_xlabel:
            ax.set_xlabel("Cidade", fontsize=12, fontweight="bold")
        ax.set_ylabel(metric_label, fontsize=11, fontweight="bold")
        ax.set_title(class_label, fontsize=12, fontweight="bold", loc="left")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        annotation = self._kruskal_annotation_for_groups(groups_for_test)
        if annotation is not None:
            self._add_statistical_annotation(ax, annotation)

    def _plot_class_across_cities(
        self,
        city_entries: List[Tuple[str, np.ndarray]],
        class_label: str,
        metric: str,
        metric_label: str,
        class_map: Optional[Dict[int, str]],
        outdir: str,
    ) -> None:
        """Plot one land-use class with one distribution per city (standalone file)."""
        n_cities = len(city_entries)
        fig, ax = plt.subplots(figsize=(max(14, 2.5 * n_cities), 8))
        self._draw_class_panel_on_ax(
            ax,
            city_entries,
            class_label,
            metric_label,
            class_map,
            show_xlabel=True,
        )
        ax.set_title(
            f"{class_label}: distribuição de {metric_label} — todas as cidades",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()
        slug = self._slugify_label(class_label)
        filename = f"{slug}_{metric}_{self._plot_extension}_by_class.png"
        plt.savefig(os.path.join(outdir, filename), dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[OK] {self._plot_extension} plot generated: {filename}")

    def _prepare_data_for_plotting(
        self,
        values: np.ndarray,
        classes: np.ndarray,
        class_map: Optional[Dict[int, str]],
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """Prepare data organized by class for plotting.

        Excludes classes specified in config.exclude_classes (e.g., NULL LULC = 0).

        Args:
            values: Raw metric values
            classes: Classification array
            class_map: Optional mapping from class codes to names

        Returns:
            Tuple of (data_by_class dict, ordered_labels list)
        """
        # Remove NaN values
        mask = ~np.isnan(values) & ~np.isnan(classes)
        if not mask.any():
            return {}, []

        vals = values[mask]
        cls = classes[mask].astype(int)

        # Organize data by class, excluding specified classes
        data_by_class: Dict[str, np.ndarray] = {}
        unique_classes = sorted(np.unique(cls))

        for class_code in unique_classes:
            if class_code == NULL_LULC_CLASS:
                continue
            # Skip excluded classes (e.g., extra codes beyond NULL)
            if class_code in self.config.exclude_classes:
                continue

            class_values = vals[cls == class_code]
            if len(class_values) > 0:
                # Use class name if available, otherwise use code
                label = (
                    class_map.get(class_code, str(class_code))
                    if class_map
                    else str(class_code)
                )
                data_by_class[label] = class_values

        # Get ordered labels (same order as stacked bar legend)
        labels = self._order_class_labels(list(data_by_class.keys()))

        return data_by_class, labels

    def _order_class_labels(self, labels: List[str]) -> List[str]:
        """Order class labels consistently with the stacked bar chart legend."""
        ordered = [label for label in LULC_LEGEND_ORDER if label in labels]
        extras = sorted(label for label in labels if label not in LULC_LEGEND_ORDER)
        return ordered + extras

    def _merge_class_labels(
        self, existing: List[str], new_labels: List[str]
    ) -> List[str]:
        """Merge class label lists preserving legend order."""
        merged = list(existing)
        for label in self._order_class_labels(new_labels):
            if label not in merged:
                merged.append(label)
        return merged

    def _plot_single_distribution(
        self,
        ax: plt.Axes,
        data: np.ndarray,
        color: str,
        position: float,
        width: float = 0.7,
    ) -> None:
        """Draw a single violin or box plot at one x position."""
        if self._plot_extension == "box":
            bp = ax.boxplot(
                [data],
                positions=[position],
                widths=width,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
            )
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.7)
            bp["boxes"][0].set_edgecolor("black")
            for element in ("whiskers", "caps", "medians"):
                for line in bp[element]:
                    line.set_color("black")
                    line.set_linewidth(1)
        else:
            parts = ax.violinplot(
                [data],
                positions=[position],
                showmeans=True,
                showmedians=True,
                widths=width,
            )
            parts["bodies"][0].set_facecolor(color)
            parts["bodies"][0].set_alpha(0.7)
            parts["bodies"][0].set_edgecolor("black")
            parts["bodies"][0].set_linewidth(1)
            for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
                if key not in parts:
                    continue
                element = parts[key]
                if isinstance(element, list):
                    for line in element:
                        line.set_color("black")
                        line.set_linewidth(1)
                else:
                    element.set_color("black")
                    element.set_linewidth(1)

    def _slugify_label(self, label: str) -> str:
        """Convert a class label into a filesystem-safe slug."""
        slug = label.strip().lower()
        slug = (
            slug.replace("á", "a")
            .replace("ã", "a")
            .replace("â", "a")
            .replace("é", "e")
            .replace("ê", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ô", "o")
            .replace("ú", "u")
            .replace("ç", "c")
        )
        slug = re.sub(r"[^\w]+", "_", slug)
        return slug.strip("_")

    def _kruskal_annotation_for_groups(
        self, groups: List[np.ndarray]
    ) -> Optional[Dict]:
        """Run Kruskal-Wallis across city groups for a single land-use class."""
        valid_groups = [group for group in groups if len(group) > 0]
        if len(valid_groups) < 2:
            return None

        result = kruskal(*valid_groups)
        p_global = result.pvalue
        n_total = sum(len(group) for group in valid_groups)
        h_stat = result.statistic
        efeito = h_stat / (n_total - 1) if n_total > 1 else np.nan

        return {
            "teste_global": "Kruskal–Wallis",
            "p_global": float(p_global),
            "efeito": float(efeito) if not np.isnan(efeito) else np.nan,
        }

    def _plot_stacked_distributions(
        self,
        ax: plt.Axes,
        data_by_class: Dict[str, np.ndarray],
        labels: List[str],
        class_map: Optional[Dict[int, str]],
        position: float,
        width: float = 0.8,
    ) -> None:
        """Draw all land-use classes overlaid at a single x position (one city column)."""
        dataset = [data_by_class[label] for label in labels]
        positions = [position] * len(labels)
        if self._plot_extension == "box":
            self._draw_stacked_boxes(ax, dataset, positions, labels, class_map, width)
        else:
            parts = ax.violinplot(
                dataset,
                positions=positions,
                showmeans=False,
                showmedians=False,
                widths=width,
            )
            self._style_violin_plot(parts, labels, class_map)

    def _draw_stacked_boxes(
        self,
        ax: plt.Axes,
        dataset: List[np.ndarray],
        positions: List[float],
        labels: List[str],
        class_map: Optional[Dict[int, str]],
        width: float,
    ) -> None:
        """Draw overlapping box plots for each land-use class at one x position."""
        colors = self._get_class_colors_for_labels(labels, class_map)
        bp = ax.boxplot(
            dataset,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        for box, color in zip(bp["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.7)
            box.set_edgecolor("black")
        for element in ("whiskers", "caps", "medians"):
            for line in bp[element]:
                line.set_color("black")
                line.set_linewidth(1)

    def _get_class_colors_for_labels(
        self, labels: List[str], class_map: Optional[Dict[int, str]]
    ) -> List[str]:
        """Resolve plot colors for a list of class labels."""
        colors = []
        for label in labels:
            class_code = None
            if class_map:
                for code, name in class_map.items():
                    if name == label:
                        class_code = code
                        break
            else:
                try:
                    class_code = int(label)
                except ValueError:
                    pass

            if class_code is not None and class_code in CLASS_COLORS:
                colors.append(CLASS_COLORS[class_code])
            else:
                colors.append(DEFAULT_CLASS_COLORS[len(colors) % len(DEFAULT_CLASS_COLORS)])
        return colors

    def _add_class_legend(
        self,
        ax: plt.Axes,
        labels: List[str],
        class_map: Optional[Dict[int, str]],
    ) -> None:
        """Add a land-use class legend matching the stacked bar chart style."""
        colors = self._get_class_colors_for_labels(labels, class_map)
        handles = [
            Patch(facecolor=color, edgecolor="black", alpha=0.7, label=label)
            for label, color in zip(labels, colors)
        ]
        ax.legend(
            handles,
            labels,
            title="Classe de uso do solo",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            frameon=True,
            fancybox=True,
            shadow=True,
        )

    def _style_violin_plot(
        self,
        parts: Dict,
        labels: List[str],
        class_map: Optional[Dict[int, str]],
    ) -> None:
        """Style the violin plot with class colors.

        Args:
            parts: Violin plot parts dictionary from matplotlib
            labels: List of class labels
            class_map: Optional mapping from class codes to names
        """
        # Get colors for each class
        colors = self._get_class_colors_for_labels(labels, class_map)

        # Apply colors to violin plot parts
        for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor("black")
            pc.set_linewidth(1)

    def _add_statistical_annotation(self, ax: plt.Axes, annotation: Dict) -> None:
        """Add statistical test results as annotation box.

        Args:
            ax: Matplotlib axes object
            annotation: Dictionary with statistical test results
        """
        test_name = annotation.get("teste_global", "")
        p = annotation.get("p_global", np.nan)
        eff = annotation.get("efeito", np.nan)
        if isinstance(p, np.generic):
            p = float(p)
        if isinstance(eff, np.generic):
            eff = float(eff)

        # Format p-value
        p_txt = (
            "p < 0.001"
            if isinstance(p, float) and p < 0.001
            else (f"p = {p:.3f}" if isinstance(p, float) else "p = n/d")
        )

        # Format effect size (epsilon squared)
        eff_txt = (
            f"ε² = {eff:.3f}"
            if isinstance(eff, float) and not np.isnan(eff)
            else "ε² = n/d"
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
            bbox=dict(
                boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray", lw=1
            ),
            fontsize=10,
        )

    def _add_city_statistical_annotation(
        self, ax: plt.Axes, x_pos: float, annotation: Dict
    ) -> None:
        """Add compact Kruskal-Wallis results above a city column."""
        p = annotation.get("p_global", np.nan)
        if not isinstance(p, float) or np.isnan(p):
            return

        p_txt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        sig = " ★" if p < self.config.alpha else ""
        y_pos = ax.get_ylim()[1]

        ax.text(
            x_pos,
            y_pos,
            f"{p_txt}{sig}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="dimgray",
        )


class BoxPlotter(ViolinPlotter):
    """Generates box plots for comparing distributions across classes."""

    _plot_extension = "box"


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

        if "classe" in combined_df.columns:
            _excl = set(self.config.exclude_classes or [])
            _excl.add(NULL_LULC_CLASS)
            combined_df = combined_df[~combined_df["classe"].isin(_excl)].copy()

        # Prepare data
        value_col, label_col = self._determine_columns(combined_df, metric, value_type)
        pivot_df = self._prepare_pivot_data(combined_df, value_col, label_col, normalize)
        if value_type == "percentage":
            pivot_df = self._reorder_lulc_columns(pivot_df)

        # Get colors for classes
        colors = self._get_class_colors(pivot_df.columns.tolist(), combined_df, label_col)

        # Create and configure plot
        fig, ax = self._create_plot(pivot_df, colors)
        self._configure_plot_axes(ax, metric, value_type, normalize)
        legend_order = (
            [c for c in LULC_LEGEND_ORDER if c in pivot_df.columns]
            if value_type == "percentage"
            else None
        )
        self._add_legend(ax, legend_order)

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

    def _reorder_lulc_columns(self, pivot_df: pd.DataFrame) -> pd.DataFrame:
        """Reorder stack columns so the legend matches LULC_LEGEND_ORDER (top → bottom)."""
        present = set(pivot_df.columns)
        stack_order = [
            cls for cls in reversed(LULC_LEGEND_ORDER) if cls in present
        ]
        extras = [c for c in pivot_df.columns if c not in stack_order]
        return pivot_df[stack_order + extras]

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
            ylabel = "Cobertura de área (%)"
        else:
            metric_label = rotulo_metrica(metric)
            tipo_label = rotulo_tipo_valor(value_type)
            ylabel = f"{metric_label} ({tipo_label})"
            if normalize:
                ylabel = f"{metric_label} ({tipo_label}) — porcentagem (%)"

        ax.set_xlabel("Cidade", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")

        # Set title
        if value_type == "percentage":
            title = "Cobertura das classes de uso do solo (%) por cidade"
        else:
            title = (
                f"Barras empilhadas: {rotulo_metrica(metric)} por cidade "
                f"e classe de uso do solo"
            )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha="right")

        # Add grid
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    def _add_legend(
        self, ax: plt.Axes, class_order: Optional[List[str]] = None
    ) -> None:
        """Add legend to the plot.

        Args:
            ax: Matplotlib axes object
            class_order: Optional explicit legend order (top → bottom)
        """
        handles, labels = ax.get_legend_handles_labels()
        if class_order:
            lookup = dict(zip(labels, handles))
            handles = [lookup[label] for label in class_order if label in lookup]
            labels = [label for label in class_order if label in lookup]

        ax.legend(
            handles,
            labels,
            title="Classe de uso do solo",
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
            return BoxPlotter(self.config)
        elif plot_type == "violin":
            return ViolinPlotter(self.config)
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

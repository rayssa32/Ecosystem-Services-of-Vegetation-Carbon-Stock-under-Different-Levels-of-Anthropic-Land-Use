"""Data aggregation by land use classes."""

from typing import Dict, List

import numpy as np
import pandas as pd

from ..utils.raster_utils import pixel_area_from_transform


class DataAggregator:
    """Aggregates metric data by land use classes."""

    @staticmethod
    def summarize_by_classes(values: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
        """Aggregate statistics by class for a clipped metric raster.

        Args:
            values: Metric values array
            classes: Classification array

        Returns:
            DataFrame with statistics (mean, median, std, sum, count) per class
        """
        mask = ~np.isnan(values) & ~np.isnan(classes)
        if not mask.any():
            return pd.DataFrame(columns=["classe", "mean", "median", "std", "sum", "count"])

        v = values[mask]
        g = classes[mask].astype(int)

        df = pd.DataFrame({"classe": g, "val": v})
        return df.groupby("classe", as_index=False)["val"].agg(
            mean="mean", median="median", std="std", sum="sum", count="count"
        )

    @staticmethod
    def add_total_kg(
        stats: pd.DataFrame, metric_name: str, transform, pixel_area_m2: float = None
    ) -> pd.DataFrame:
        """Add total_kg column to statistics DataFrame.

        Args:
            stats: Statistics DataFrame
            metric_name: Name of the metric
            transform: Affine transform for calculating pixel area
            pixel_area_m2: Pre-calculated pixel area (optional)

        Returns:
            DataFrame with total_kg column added
        """
        if pixel_area_m2 is None:
            pixel_area_m2 = pixel_area_from_transform(transform)

        if not stats.empty:
            stats["total_kg"] = stats["sum"] * pixel_area_m2
            stats = stats.rename(
                columns={
                    "mean": f"{metric_name}_mean",
                    "median": f"{metric_name}_median",
                    "std": f"{metric_name}_std",
                    "sum": f"{metric_name}_sum",
                    "count": f"{metric_name}_count",
                    "total_kg": f"{metric_name}_total_kg",
                }
            )
        return stats

    @staticmethod
    def merge_metric_stats(metric_stats: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge statistics from multiple metrics by class.

        Args:
            metric_stats: List of DataFrames with statistics per metric

        Returns:
            Merged DataFrame with all metrics
        """
        if not metric_stats:
            return pd.DataFrame()

        merged = metric_stats[0]
        for df in metric_stats[1:]:
            merged = merged.merge(df, on="classe", how="outer")

        return merged

    @staticmethod
    def add_metadata(
        df: pd.DataFrame, city: str, class_map: Dict[int, str] = None
    ) -> pd.DataFrame:
        """Add city name and class labels to DataFrame.

        Args:
            df: Statistics DataFrame
            city: City name
            class_map: Mapping from class codes to names

        Returns:
            DataFrame with cidade and classe_nome columns added
        """
        df = df.copy()
        df["cidade"] = city

        if class_map:
            df["classe_nome"] = df["classe"].map(class_map).fillna(df["classe"].astype(str))

        return df

    @staticmethod
    def calculate_class_area_percentages(classes: np.ndarray) -> pd.DataFrame:
        """Calculate percentage area covered by each land use class.

        Args:
            classes: Classification array

        Returns:
            DataFrame with classe and percentage columns
        """
        # Remove NaN values
        mask = ~np.isnan(classes)
        if not mask.any():
            return pd.DataFrame(columns=["classe", "percentage"])

        cls = classes[mask].astype(int)
        
        # Count pixels per class
        unique, counts = np.unique(cls, return_counts=True)
        total_pixels = cls.size
        
        # Calculate percentages
        percentages = (counts / total_pixels) * 100
        
        df = pd.DataFrame({"classe": unique, "percentage": percentages})
        df = df.sort_values("classe")
        
        return df

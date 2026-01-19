"""Statistical analysis and hypothesis testing."""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import shapiro, levene, f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

from ..config import AnalysisConfig


class StatisticsAnalyzer:
    """Performs inferential statistical tests and effect size calculations."""

    def __init__(self, config: AnalysisConfig):
        """Initialize statistics analyzer with configuration.

        Args:
            config: Analysis configuration object
        """
        self.config = config

    def sample_per_class(
        self,
        values: np.ndarray,
        classes: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """Sample up to k values per class with exclusion rules.

        Args:
            values: Metric values array
            classes: Classification array

        Returns:
            Dictionary mapping class codes to sampled arrays
        """
        samples: Dict[int, np.ndarray] = {}
        mask = ~np.isnan(values) & ~np.isnan(classes)

        if not mask.any():
            return samples

        vals = values[mask]
        cls = classes[mask].astype(int)

        rng = np.random.default_rng(self.config.rng_seed)

        for c in np.unique(cls):
            if c in self.config.exclude_classes:
                continue

            v = vals[cls == c]
            if v.size >= self.config.min_n_for_tests:
                if v.size > self.config.sample_per_class:
                    idx = rng.choice(v.size, size=self.config.sample_per_class, replace=False)
                    v = v[idx]
                samples[c] = v

        return samples

    def effect_size_anova(self, groups: List[np.ndarray]) -> float:
        """Calculate η² (eta-squared) effect size for ANOVA.

        Args:
            groups: List of arrays, one per group

        Returns:
            Effect size value or NaN if calculation not possible
        """
        all_vals = np.concatenate(groups)
        grand = np.mean(all_vals)
        ss_between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
        ss_total = sum(((g - grand) ** 2).sum() for g in groups)
        return float(ss_between / ss_total) if ss_total > 0 else np.nan

    def effect_size_kruskal(self, groups: List[np.ndarray]) -> float:
        """Calculate ε² (epsilon-squared) effect size for Kruskal-Wallis.

        Uses formula from Tomczak & Tomczak (2014).

        Args:
            groups: List of arrays, one per group

        Returns:
            Effect size value or NaN if calculation not possible
        """
        k = len(groups)
        n = sum(len(g) for g in groups)
        H = kruskal(*groups).statistic
        return float((H - k + 1) / (n - k)) if (n - k) > 0 else np.nan

    def check_normality(self, groups: List[np.ndarray]) -> bool:
        """Check normality assumption using Shapiro-Wilk test.

        Args:
            groups: List of arrays to test

        Returns:
            True if all groups pass normality test, False otherwise
        """
        for g in groups:
            if len(g) < 3:
                return False
            try:
                if shapiro(g).pvalue < self.config.alpha:
                    return False
            except Exception:
                return False
        return True

    def check_homogeneity(self, groups: List[np.ndarray]) -> bool:
        """Check homogeneity of variances using Levene's test.

        Args:
            groups: List of arrays to test

        Returns:
            True if groups have homogeneous variances, False otherwise
        """
        try:
            lev_p = levene(*groups, center="median").pvalue
            return (lev_p >= self.config.alpha) if not np.isnan(lev_p) else False
        except Exception:
            return False

    def run_tukey_posthoc(
        self, long_df: pd.DataFrame, city: str, metric: str, outdir: str
    ) -> None:
        """Run Tukey HSD post-hoc test and save results.

        Args:
            long_df: Long-format DataFrame with values and class names
            city: City name for file naming
            metric: Metric name for file naming
            outdir: Output directory for CSV file
        """
        try:
            tuk = pairwise_tukeyhsd(
                endog=long_df["valor"].values,
                groups=long_df["classe_nome"].values,
                alpha=self.config.alpha,
            )
            tuk_df = pd.DataFrame(
                tuk._results_table.data[1:], columns=tuk._results_table.data[0]
            )
            os.makedirs(outdir, exist_ok=True)
            tuk_df.to_csv(
                os.path.join(outdir, f"pairwise_{city}_{metric}_tukey.csv"), index=False
            )
        except Exception:
            pass  # Silently fail if post-hoc cannot be computed

    def run_dunn_holm_posthoc(
        self, long_df: pd.DataFrame, city: str, metric: str, outdir: str
    ) -> None:
        """Run Dunn-Holm post-hoc test and save results.

        Args:
            long_df: Long-format DataFrame with values and class names
            city: City name for file naming
            metric: Metric name for file naming
            outdir: Output directory for CSV file
        """
        try:
            dunn = sp.posthoc_dunn(
                long_df, val_col="valor", group_col="classe_nome", p_adjust="holm"
            )
            os.makedirs(outdir, exist_ok=True)
            dunn.to_csv(os.path.join(outdir, f"pairwise_{city}_{metric}_dunn_holm.csv"))
        except Exception:
            pass  # Silently fail if post-hoc cannot be computed

    def run_inferential_tests(
        self,
        city: str,
        metric: str,
        metric_clip: np.ndarray,
        class_clip: np.ndarray,
        class_map: Optional[Dict[int, str]],
        outdir: str,
    ) -> Dict[str, object]:
        """Run complete inferential analysis: choose test, run global test and post-hoc.

        Args:
            city: City name
            metric: Metric name
            metric_clip: Clipped metric array
            class_clip: Clipped classification array
            class_map: Optional mapping from class codes to names
            outdir: Output directory for post-hoc CSVs

        Returns:
            Dictionary with test results (teste_global, p_global, efeito)
        """
        samples = self.sample_per_class(metric_clip, class_clip)

        if len(samples) < 2:
            return {
                "cidade": city,
                "metrica": metric,
                "teste_global": "—",
                "p_global": np.nan,
                "efeito": np.nan,
            }

        codes = sorted(samples.keys())
        labels = [class_map.get(c, str(c)) if class_map else str(c) for c in codes]
        groups = [samples[c] for c in codes]

        # Check assumptions
        all_norm = self.check_normality(groups)
        homo = self.check_homogeneity(groups)

        # Choose and run test
        if all_norm and homo:
            teste_global = "ANOVA"
            p_global = f_oneway(*groups).pvalue
            efeito = self.effect_size_anova(groups)

            # Post-hoc Tukey
            long_df = pd.DataFrame(
                {
                    "valor": np.concatenate(groups),
                    "classe_nome": np.repeat(labels, [len(g) for g in groups]),
                }
            )
            self.run_tukey_posthoc(long_df, city, metric, outdir)

        else:
            teste_global = "Kruskal–Wallis"
            p_global = kruskal(*groups).pvalue
            efeito = self.effect_size_kruskal(groups)

            # Post-hoc Dunn-Holm
            long_df = pd.DataFrame(
                {
                    "valor": np.concatenate(groups),
                    "classe_nome": np.repeat(labels, [len(g) for g in groups]),
                }
            )
            self.run_dunn_holm_posthoc(long_df, city, metric, outdir)

        return {
            "cidade": city,
            "metrica": metric,
            "teste_global": teste_global,
            "p_global": p_global,
            "efeito": efeito,
        }

    def run_kruskal_wallis_test(
        self,
        metric_clip: np.ndarray,
        class_clip: np.ndarray,
        class_map: Optional[Dict[int, str]],
    ) -> Dict[str, object]:
        """Run Kruskal-Wallis test and calculate epsilon-squared effect size.

        Excludes classes specified in config.exclude_classes from the test.

        Args:
            metric_clip: Clipped metric array
            class_clip: Clipped classification array
            class_map: Optional mapping from class codes to names

        Returns:
            Dictionary with test results (teste_global, p_global, efeito)
        """
        # sample_per_class already respects config.exclude_classes
        samples = self.sample_per_class(metric_clip, class_clip)

        if len(samples) < 2:
            return {
                "teste_global": "Kruskal–Wallis",
                "p_global": np.nan,
                "efeito": np.nan,
            }

        codes = sorted(samples.keys())
        groups = [samples[c] for c in codes]

        # Run Kruskal-Wallis test
        p_global = kruskal(*groups).pvalue
        efeito = self.effect_size_kruskal(groups)

        return {
            "teste_global": "Kruskal–Wallis",
            "p_global": p_global,
            "efeito": efeito,
        }

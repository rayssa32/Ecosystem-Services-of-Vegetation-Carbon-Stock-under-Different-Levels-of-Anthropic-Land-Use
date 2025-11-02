"""
Análises por cidade + estatística inferencial com anotação nos gráficos.

Pipeline:
1) Valida caminhos e CRS do raster de classes (projetado em metros).
2) Para cada cidade (polígono), recorta classes e "n" rasters de métricas (WarpedVRT).
3) Agrega por classe: média, mediana, desvio, soma, contagem e total_kg (área de pixel).
4) (Opcional) Teste global (ANOVA ou Kruskal–Wallis) + pós-hoc (Tukey / Dunn-Holm).
5) Gera CSV por cidade, CSV combinado e CSV de síntese inferencial.
6) (Opcional) Plota barras de médias por classe com anotação (teste/p/efeito).
"""

from __future__ import annotations

import os
from contextlib import ExitStack
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from affine import Affine
from rasterio.mask import mask as rio_mask
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from shapely.geometry import mapping
from scipy.stats import f_oneway, kruskal, levene, shapiro
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp


# ============================== Config ============================== #
RESAMPLE_METRICS = "nearest"     # nearest | bilinear | cubic | average | ...
MAKE_PLOTS = True                # gerar PNGs por cidade/métrica
OUTDIR = "./dados_gerados"       # pasta de saída
RUN_INFERENTIAL_TESTS = True     # executar ANOVA/Kruskal e pós-hoc
EXCLUDE_CLASSES: List[int] = [5] # classes a ignorar (se houver no raster)
SAMPLE_PER_CLASS = 5000          # amostragem por classe p/ testes
MIN_N_FOR_TESTS = 10             # mínimo de observações por classe p/ testes
ALPHA = 0.05                     # significância p/ normalidade/homogeneidade e pós-hoc
RNG_SEED = 42                    # reprodutibilidade da amostragem


# ========================= Utils / Helpers ========================= #
def _pixel_area_from_transform(transform: Affine) -> float:
    """Retorna a área do pixel em m² a partir do transform (CRS projetado)."""
    return abs(transform.a * transform.e)


def _resampling_mode(name: str) -> Resampling:
    """Mapeia nome textual para enum do rasterio.Resampling."""
    name = (name or "nearest").lower()
    return {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "cubicspline": Resampling.cubic_spline,
        "lanczos": Resampling.lanczos,
        "average": Resampling.average,
        "mode": Resampling.mode,
        "max": Resampling.max,
        "min": Resampling.min,
        "med": Resampling.med,
        "q1": Resampling.q1,
        "q3": Resampling.q3,
    }.get(name, Resampling.nearest)


def _summarize_by_classes(values: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    """Agrega estatísticas por classe para um raster de métricas já recortado."""
    mask = ~np.isnan(values) & ~np.isnan(classes)
    if not mask.any():
        return pd.DataFrame(columns=["classe", "mean", "median", "std", "sum", "count"])

    v = values[mask]
    g = classes[mask].astype(int)
    df = pd.DataFrame({"classe": g, "val": v})
    return df.groupby("classe", as_index=False)["val"].agg(
        mean="mean",
        median="median",
        std="std",
        sum="sum",
        count="count",
    )


def _barplot_with_annotation(
    df_city: pd.DataFrame,
    metric: str,
    label_col: str,
    outdir: str,
    city: str,
    annot: Optional[dict] = None,
) -> None:
    """Plota médias por classe com anotação (teste / p-valor / efeito)."""
    labels = (
        df_city[label_col].astype(str).values
        if label_col in df_city.columns
        else df_city["classe"].astype(str).values
    )

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in df_city.columns:
        return

    means = df_city[mean_col].astype(float).values
    errs = df_city[std_col].astype(float).values if std_col in df_city.columns else None

    x = np.arange(len(labels))
    bar_width = 0.45
    fig_w = max(6.0, 0.9 * len(labels))
    plt.figure(figsize=(fig_w, 4.6))
    ax = plt.gca()

    ax.bar(
        x,
        means,
        width=bar_width,
        yerr=None,  # opcional: yerr=errs, ecolor="gray", elinewidth=0.5
        capsize=3,
        linewidth=0.8,
        edgecolor="black",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(f"{city}: Média de {metric} por classe")

    # Ajuste de escala Y (zoom no intervalo dos dados)
    if errs is not None and np.isfinite(errs).any():
        y_low = np.nanmin(means - errs)
        y_high = np.nanmax(means + errs)
    else:
        y_low = np.nanmin(means)
        y_high = np.nanmax(means)

    if not np.isfinite(y_low) or not np.isfinite(y_high):
        y_low, y_high = 0.0, 1.0

    rng = y_high - y_low
    if rng <= 0:
        rng = max(abs(y_high), 1.0)

    pad = max(0.08 * rng, 1e-6)
    ax.set_ylim(y_low - pad, y_high + pad)

    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.6)
    try:
        import matplotlib.ticker as mticker
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
    except Exception:
        pass

    voff = 0.015 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for xi, yi in enumerate(means):
        if np.isfinite(yi):
            ax.text(xi, yi + voff, f"{yi:.1f}", ha="center", va="bottom", fontsize=8)

    # Anotação do teste
    if annot is not None and "p_global" in annot:
        test_name = annot.get("teste_global", "")
        p = annot.get("p_global", np.nan)
        eff = annot.get("efeito", np.nan)

        p_txt = "p < 0.001" if isinstance(p, float) and p < 0.001 else (
            f"p = {p:.3f}" if isinstance(p, float) else "p = n/a"
        )
        eff_sym = "η²" if ("ANOVA" in str(test_name)) else "ε²"
        eff_txt = (
            f"{eff_sym} = {eff:.2f}" if isinstance(eff, float) and not np.isnan(eff) else f"{eff_sym} = n/a"
        )
        sig = "★" if (isinstance(p, float) and p < ALPHA) else ""
        box_txt = f"{test_name}\n{p_txt}   {eff_txt}  {sig}"

        ax.text(
            0.98,
            0.98,
            box_txt,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    plt.margins(x=0.02)
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    safe_city = city.replace("/", "-").replace("\\", "-")
    plt.savefig(os.path.join(outdir, f"{safe_city}_{metric}_means.png"), dpi=220)
    plt.close()


def _sample_per_class(
    values: np.ndarray,
    classes: np.ndarray,
    exclude: List[int],
    k: int,
    min_n: int,
    seed: int = RNG_SEED,
) -> Dict[int, np.ndarray]:
    """Amostra até k valores por classe (exclui classes e NaNs), com seed fixa."""
    samples: Dict[int, np.ndarray] = {}

    mask = ~np.isnan(values) & ~np.isnan(classes)
    if not mask.any():
        return samples

    vals = values[mask]
    cls = classes[mask].astype(int)
    rng = np.random.default_rng(seed)

    for c in np.unique(cls):
        if c in exclude:
            continue
        v = vals[cls == c]
        if v.size >= min_n:
            if v.size > k:
                idx = rng.choice(v.size, size=k, replace=False)
                v = v[idx]
            samples[c] = v
    return samples


def _effect_size_anova(groups: List[np.ndarray]) -> float:
    """η² (ANOVA)."""
    all_vals = np.concatenate(groups)
    grand = np.mean(all_vals)
    ss_between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    ss_total = sum(((g - grand) ** 2).sum() for g in groups)
    return float(ss_between / ss_total) if ss_total > 0 else np.nan


def _effect_size_kruskal(groups: List[np.ndarray]) -> float:
    """ε² (Kruskal–Wallis), Tomczak & Tomczak (2014)."""
    k = len(groups)
    n = sum(len(g) for g in groups)
    H = kruskal(*groups).statistic
    return float((H - k + 1) / (n - k)) if (n - k) > 0 else np.nan


def _infer_tests_for_city_metric(
    city: str,
    metric: str,
    metr_clip: np.ndarray,
    class_clip: np.ndarray,
    class_map: Optional[dict],
    outdir: str,
) -> Dict[str, object]:
    """Escolhe ANOVA/Kruskal por diagnósticos; roda teste global + pós-hoc; salva CSVs."""
    samples = _sample_per_class(
        metr_clip, class_clip, EXCLUDE_CLASSES, SAMPLE_PER_CLASS, MIN_N_FOR_TESTS
    )
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

    # Normalidade & homocedasticidade
    all_norm = True
    for g in groups:
        if len(g) >= 3:
            try:
                if shapiro(g).pvalue < ALPHA:
                    all_norm = False
            except Exception:
                all_norm = False
        else:
            all_norm = False

    try:
        lev_p = levene(*groups, center="median").pvalue
    except Exception:
        lev_p = np.nan
    homo = (lev_p >= ALPHA) if not np.isnan(lev_p) else False

    if all_norm and homo:
        teste_global = "ANOVA"
        p_global = f_oneway(*groups).pvalue
        efeito = _effect_size_anova(groups)

        long_df = pd.DataFrame(
            {"valor": np.concatenate(groups), "classe_nome": np.repeat(labels, [len(g) for g in groups])}
        )
        try:
            tuk = pairwise_tukeyhsd(
                endog=long_df["valor"].values,
                groups=long_df["classe_nome"].values,
                alpha=ALPHA,
            )
            tuk_df = pd.DataFrame(tuk._results_table.data[1:], columns=tuk._results_table.data[0])
            os.makedirs(outdir, exist_ok=True)
            tuk_df.to_csv(os.path.join(outdir, f"pairwise_{city}_{metric}_tukey.csv"), index=False)
        except Exception:
            pass
    else:
        teste_global = "Kruskal–Wallis"
        p_global = kruskal(*groups).pvalue
        efeito = _effect_size_kruskal(groups)

        long_df = pd.DataFrame(
            {"valor": np.concatenate(groups), "classe_nome": np.repeat(labels, [len(g) for g in groups])}
        )
        try:
            dunn = sp.posthoc_dunn(long_df, val_col="valor", group_col="classe_nome", p_adjust="holm")
            os.makedirs(outdir, exist_ok=True)
            dunn.to_csv(os.path.join(outdir, f"pairwise_{city}_{metric}_dunn_holm.csv"))
        except Exception:
            pass

    return {
        "cidade": city,
        "metrica": metric,
        "teste_global": teste_global,
        "p_global": p_global,
        "efeito": efeito,
    }


def _apply_class_filters(class_arr: np.ndarray, class_map: Optional[dict]) -> np.ndarray:
    """
    Mantém apenas classes definidas em class_map (menos EXCLUDE_CLASSES).
    Outras viram NaN e são ignoradas por stats, testes e gráficos.
    """
    if class_arr is None:
        return class_arr

    out = class_arr.copy()
    mask_valid = ~np.isnan(out)

    if class_map and len(class_map) > 0:
        allowed = set(class_map.keys())
    else:
        allowed = set(np.unique(out[mask_valid].astype(int)))

    if EXCLUDE_CLASSES:
        allowed -= set(EXCLUDE_CLASSES)

    cls_i = out[mask_valid].astype(np.int64, copy=False)
    keep_mask = np.isin(cls_i, np.array(list(allowed), dtype=np.int64))
    tmp = out[mask_valid]
    tmp[~keep_mask] = np.nan
    out[mask_valid] = tmp
    return out


# ============================== Pipeline ============================== #
def run_mode_A_with_stats_annot(
    class_raster_path: str,
    metrics_rasters: Dict[str, str],
    vector_cities_path: str,
    city_field: str = "municipio",
    class_map: Optional[dict] = None,
    resample_mode: str = RESAMPLE_METRICS,
    make_plots: bool = MAKE_PLOTS,
    outdir: str = OUTDIR,
) -> pd.DataFrame:
    """Clip por cidade → sumariza → inferência → gráficos → CSVs."""
    # Checagem de caminhos
    for p in [class_raster_path, *metrics_rasters.values(), vector_cities_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Caminho não encontrado: {p}")

    os.makedirs(outdir, exist_ok=True)
    stats_dir = os.path.join(outdir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    metrics_order = list(metrics_rasters.keys())

    combined_rows: List[pd.DataFrame] = []
    infer_rows: List[dict] = []

    with rasterio.open(class_raster_path) as src_class, ExitStack() as stack:
        if not (src_class.crs and src_class.crs.is_projected):
            raise ValueError("O raster de classes precisa estar em CRS projetado (metros).")

        # Abrir rasters de métricas
        src_metrics = {m: stack.enter_context(rasterio.open(p)) for m, p in metrics_rasters.items()}
        resampling = _resampling_mode(resample_mode)

        # Carregar vetor e reprojetar
        gdf = gpd.read_file(vector_cities_path)
        if city_field not in gdf.columns:
            raise ValueError(f"Campo '{city_field}' não encontrado no vetor.")
        gdf = gdf.to_crs(src_class.crs)

        # Loop por cidade
        for _, row in gdf.iterrows():
            city = str(row[city_field]).strip()
            if row.geometry is None or row.geometry.is_empty:
                continue

            geom = [mapping(row.geometry)]

            # Clip das classes
            try:
                class_ma, class_transform = rio_mask(src_class, geom, crop=True, filled=False)
            except ValueError:
                # geometria fora do raster
                continue

            class_clip = class_ma[0].astype("float32", copy=False)
            if np.ma.isMaskedArray(class_ma):
                class_clip[class_ma.mask[0]] = np.nan
            if src_class.nodata is not None:
                class_clip[class_clip == float(src_class.nodata)] = np.nan

            class_clip = _apply_class_filters(class_clip, class_map)
            pixel_area_m2 = _pixel_area_from_transform(class_transform)

            # Clips das métricas + sumarização (coleta apenas)
            metric_stats: List[pd.DataFrame] = []
            raw_arrays: Dict[str, np.ndarray] = {}

            for m, srcm in src_metrics.items():
                SENTINEL = np.float32(-3.4e38)

                # WarpedVRT garante alinhamento no grid do clip de classes
                with WarpedVRT(
                    srcm,
                    crs=src_class.crs,
                    transform=class_transform,
                    width=class_clip.shape[1],
                    height=class_clip.shape[0],
                    resampling=resampling,
                    src_nodata=srcm.nodata,
                    dst_nodata=float(SENTINEL),
                ) as vrt:
                    metr_ma, _ = rio_mask(vrt, geom, crop=False, filled=False)
                    metr = metr_ma[0].astype("float32", copy=False)

                if np.ma.isMaskedArray(metr_ma):
                    metr[metr_ma.mask[0]] = np.nan
                metr[metr == SENTINEL] = np.nan
                if srcm.nodata is not None:
                    metr[metr == float(srcm.nodata)] = np.nan
                metr[metr <= -1e10] = np.nan  # cortes agressivos de lixo

                stats = _summarize_by_classes(metr, class_clip)
                if not stats.empty:
                    stats = stats.rename(columns={
                        "mean":   f"{m}_mean",
                        "median": f"{m}_median",
                        "std":    f"{m}_std",
                        "sum":    f"{m}_sum",
                        "count":  f"{m}_count",
                    })
                    stats[f"{m}_total_kg"] = stats[f"{m}_sum"] * pixel_area_m2
                    metric_stats.append(stats)
                    raw_arrays[m] = metr

            # Após processar todas as métricas da cidade:
            if not metric_stats:
                continue

            # Merge das métricas por classe
            df_city = metric_stats[0]
            for d in metric_stats[1:]:
                df_city = df_city.merge(d, on="classe", how="outer")
            if df_city.empty:
                continue

            df_city["cidade"] = city
            if class_map:
                df_city["classe_nome"] = df_city["classe"].map(class_map).fillna(df_city["classe"].astype(str))

            # CSV por cidade
            safe_city = city.replace("/", "-").replace("\\", "-")
            city_csv = os.path.join(outdir, f"{safe_city}_stats_por_classe.csv")
            df_city.to_csv(city_csv, index=False)
            combined_rows.append(df_city)

            # Inferência + anotação (somente métricas existentes)
            stats_annots: Dict[str, dict] = {}
            if RUN_INFERENTIAL_TESTS:
                city_stats_dir = os.path.join(stats_dir, safe_city.replace(" ", "_"))
                os.makedirs(city_stats_dir, exist_ok=True)
                for m in metrics_order:
                    if m in raw_arrays:
                        res = _infer_tests_for_city_metric(city, m, raw_arrays[m], class_clip, class_map, city_stats_dir)
                        infer_rows.append(res)
                        stats_annots[m] = res

            # Gráficos
            if MAKE_PLOTS:
                label_col = "classe_nome" if "classe_nome" in df_city.columns else "classe"
                for m in metrics_order:
                    if m in raw_arrays:
                        _barplot_with_annotation(df_city, m, label_col, outdir, city, annot=stats_annots.get(m))

    # ============ Saídas combinadas (após TODAS as cidades) ============
    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        combined_path = os.path.join(outdir, "todas_cidades_stats_por_classe.csv")
        combined.to_csv(combined_path, index=False)
        print(f"[OK] Saída combinada: {combined_path}")
    else:
        combined = pd.DataFrame()
        print("[Aviso] Nenhuma estatística gerada.")

    if RUN_INFERENTIAL_TESTS and infer_rows:
        infer_df = pd.DataFrame(infer_rows)
        infer_path = os.path.join(outdir, "stats", "resumo_inferencial_por_cidade.csv")
        os.makedirs(os.path.dirname(infer_path), exist_ok=True)
        infer_df.to_csv(infer_path, index=False)
        print(f"[OK] Resumo inferencial: {infer_path}")

    return combined


# ============================== Entrypoint ============================== #
def main() -> None:
    """Define caminhos e executa o pipeline completo."""
    class_raster_path = "classificacao/final_class3.tif"
    metrics_rasters = {
        "GPP": "metricas/GPP_sete_cidades.tif",
        "NPP": "metricas/NPP_sete_cidades.tif",
        "Biomassa": "metricas/Biomass_sete_cidades.tif",
    }
    vector_cities_path = "shapefile/sete_cidades.shp"
    city_field = "NM_MUN"
    class_map = {1: "Vegetação", 2: "Urbano", 3: "Água", 4: "Solo"}

    run_mode_A_with_stats_annot(
        class_raster_path=class_raster_path,
        metrics_rasters=metrics_rasters,
        vector_cities_path=vector_cities_path,
        city_field=city_field,
        class_map=class_map,
        resample_mode=RESAMPLE_METRICS,
        make_plots=MAKE_PLOTS,
        outdir=OUTDIR,
    )


if __name__ == "__main__":
    main()

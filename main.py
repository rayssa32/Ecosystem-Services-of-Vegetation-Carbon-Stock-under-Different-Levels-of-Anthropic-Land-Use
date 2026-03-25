"""
Ponto único de entrada: configure aqui arquivos, cidades e tipos de gráfico.

Uso: python main.py
"""

import os
from typing import List, Optional

from src import AnalysisConfig, AnalysisPipeline, run_moran_analysis
from src.config import PathsConfig, MoranConfig, SankeyConfig
from src.utils.constants import NULL_LULC_CLASS, WATER_LULC_CLASS


# =============================================================================
# CONFIGURAÇÃO — edite apenas esta seção
# =============================================================================

# ---- Arquivos ----
PATHS = {
    "class_raster": "classificacao/LULC_7Cidades_10m_20250710_20250730_projected.tif",
    "biomass_raster": "metricas/Biomass_sete_cidades_projected.tif",
    "vector_cities": "shapefile/sete_cidades.shp",
    "city_field": "NM_MUN",
    "outdir": "./dados_gerados",
}

# ---- Cidades (None = todas) ----
CITIES_FILTER: Optional[List[str]] = None
# Exemplos: None  |  ["Lavras"]  |  ["Lavras", "Varginha", "Alfenas"]

# ---- O que rodar ----
RUN_VIOLIN = False  # Gráfico de violino (biomassa por classe de uso, cidades combinadas)
RUN_SANKEY = False   # Sankey: uso do solo → classe de biomassa (quantis); um por cidade ou geral
RUN_MORAN = False   # Moran's I + scatter por cidade (resolução nativa da biomassa)
RUN_STACKED_LULC = False  # Barras empilhadas: cobertura (%) por classe de uso do solo, por cidade
RUN_SHANNON = False  # Shannon (H') e equitabilidade de Pielou (J') sobre proporções LULC por cidade

# ---- Opções do gráfico de violino ----
PLOT_TYPES = ["violin"]   # Opções: "violin" | "bar" | "box"
# NULL (0) e Água (1) não entram em violino/Sankey
EXCLUDE_NULL_LULC = [NULL_LULC_CLASS]
EXCLUDE_CLASSES_VIOLIN = [NULL_LULC_CLASS, WATER_LULC_CLASS]
# Shannon: NULL (0) é sempre excluído no cálculo; adicione aqui outras classes se quiser (ex.: água)
EXCLUDE_CLASSES_SHANNON: List[int] = []

# ---- Opções do Sankey (se RUN_SANKEY = True) ----
SANKEY_PER_CITY = True   # True = um Sankey por cidade; False = um Sankey geral (todas as cidades)
SANKEY_N_QUANTILES = 3   # Classes de biomassa por quantis (ex.: 3 = Low, Medium, High)
SANKEY_USE_PERCENT = True   # True = espessura = % de pixels; False = nº de pixels

# ---- Opções do Moran (se RUN_MORAN = True) ----
MORAN_NATIVE_RESOLUTION = True   # True = resolução nativa; False = reamostrado 10 m
MORAN_PERMUTATIONS = 999
MORAN_SAVE_SCATTER = True

# ---- Nomes das classes (LULC): 0 = NULL; cobertura válida 1–5 ----
CLASS_MAP = {
    0: "NULL",
    1: "Água",
    2: "Urbano",
    3: "Solo",
    4: "Vegetação",
    5: "Agro/Pasto",
}

# =============================================================================


def _paths_config() -> PathsConfig:
    return PathsConfig(
        class_raster_path=PATHS["class_raster"],
        biomass_raster_path=PATHS["biomass_raster"],
        vector_cities_path=PATHS["vector_cities"],
        city_field=PATHS["city_field"],
        outdir=PATHS["outdir"],
    )


def _validate_paths(paths: PathsConfig, need_class: bool) -> bool:
    required = [paths.biomass_raster_path, paths.vector_cities_path]
    if need_class:
        required.append(paths.class_raster_path)
    for p in required:
        if not os.path.exists(p):
            print(f"[ERRO] Arquivo não encontrado: {p}")
            return False
    return True


def main() -> None:
    paths = _paths_config()

    if RUN_VIOLIN:
        if not _validate_paths(paths, need_class=True):
            return
        config = AnalysisConfig(
            resample_metrics="nearest",
            make_plots=True,
            outdir=paths.outdir,
            plot_types=PLOT_TYPES,
            make_stacked_bar_charts=False,
            save_csv_files=False,
            run_inferential_tests=False,
            exclude_classes=EXCLUDE_CLASSES_VIOLIN,
            sample_per_class=5000,
            min_n_for_tests=10,
            alpha=0.05,
            rng_seed=42,
        )
        pipeline = AnalysisPipeline(config)
        pipeline.run_violin_plots_analysis(
            class_raster_path=paths.class_raster_path,
            biomass_raster_path=paths.biomass_raster_path,
            vector_cities_path=paths.vector_cities_path,
            city_field=paths.city_field,
            class_map=CLASS_MAP,
            cities_filter=CITIES_FILTER,
        )
        print("[OK] Violino concluído.")

    if RUN_SANKEY:
        if not _validate_paths(paths, need_class=True):
            return
        sankey_cfg = SankeyConfig(
            per_city=SANKEY_PER_CITY,
            n_quantiles=SANKEY_N_QUANTILES,
            use_percentage=SANKEY_USE_PERCENT,
        )
        config = AnalysisConfig(
            resample_metrics="nearest",
            make_plots=True,
            outdir=paths.outdir,
            plot_types=PLOT_TYPES,
            make_stacked_bar_charts=False,
            save_csv_files=False,
            run_inferential_tests=False,
            exclude_classes=EXCLUDE_CLASSES_VIOLIN,
            sample_per_class=5000,
            min_n_for_tests=10,
            alpha=0.05,
            rng_seed=42,
        )
        pipeline = AnalysisPipeline(config)
        pipeline.run_sankey_analysis(
            class_raster_path=paths.class_raster_path,
            biomass_raster_path=paths.biomass_raster_path,
            vector_cities_path=paths.vector_cities_path,
            city_field=paths.city_field,
            class_map=CLASS_MAP,
            cities_filter=CITIES_FILTER,
            sankey_config=sankey_cfg,
        )
        print("[OK] Sankey concluído.")

    if RUN_STACKED_LULC:
        if not _validate_paths(paths, need_class=True):
            return
        config = AnalysisConfig(
            resample_metrics="nearest",
            make_plots=False,
            outdir=paths.outdir,
            make_stacked_bar_charts=True,
            save_csv_files=True,
            run_inferential_tests=False,
            exclude_classes=EXCLUDE_NULL_LULC,
            sample_per_class=5000,
            min_n_for_tests=10,
            alpha=0.05,
            rng_seed=42,
        )
        pipeline = AnalysisPipeline(config)
        pipeline.run(
            class_raster_path=paths.class_raster_path,
            metrics_rasters=None,
            vector_cities_path=paths.vector_cities_path,
            city_field=paths.city_field,
            class_map=CLASS_MAP,
            cities_filter=CITIES_FILTER,
        )
        out_png = os.path.join(paths.outdir, "stacked_bar_land_use_percentage.png")
        print(f"[OK] Gráfico LULC empilhado: {out_png}")

    if RUN_SHANNON:
        if not _validate_paths(paths, need_class=True):
            return
        config = AnalysisConfig(
            resample_metrics="nearest",
            make_plots=False,
            outdir=paths.outdir,
            make_stacked_bar_charts=False,
            save_csv_files=True,
            run_inferential_tests=False,
            exclude_classes=EXCLUDE_CLASSES_SHANNON,
            sample_per_class=5000,
            min_n_for_tests=10,
            alpha=0.05,
            rng_seed=42,
        )
        pipeline = AnalysisPipeline(config)
        pipeline.run_shannon_index_analysis(
            class_raster_path=paths.class_raster_path,
            vector_cities_path=paths.vector_cities_path,
            city_field=paths.city_field,
            cities_filter=CITIES_FILTER,
        )
        print("[OK] Shannon / equitabilidade concluído.")

    if RUN_MORAN:
        if not _validate_paths(paths, need_class=not MORAN_NATIVE_RESOLUTION):
            return
        moran_cfg = MoranConfig(
            use_native_resolution=MORAN_NATIVE_RESOLUTION,
            cities_filter=CITIES_FILTER,
            permutations=MORAN_PERMUTATIONS,
            contiguity="rook",
            save_scatter_plots=MORAN_SAVE_SCATTER,
        )
        print(f"Moran's I — {moran_cfg.permutations} permutações\n")
        df = run_moran_analysis(paths, moran_cfg)
        if not df.empty:
            moran_dir = os.path.join(paths.outdir, "moran")
            csv_name = "moran_global_por_cidade_nativo.csv" if moran_cfg.use_native_resolution else "moran_global_por_cidade.csv"
            print(f"\n{os.path.join(moran_dir, csv_name)}")
            print(df.to_string(index=False))
        print("[OK] Moran concluído.")


if __name__ == "__main__":
    main()

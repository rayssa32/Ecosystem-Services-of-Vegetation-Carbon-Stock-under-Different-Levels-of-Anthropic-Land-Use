"""
Ponto único de entrada: configure aqui arquivos, cidades e tipos de gráfico.

Uso: python main.py
"""

import os
from typing import List, Optional

from src import AnalysisConfig, AnalysisPipeline, run_moran_analysis
from src.config import PathsConfig, MoranConfig


# =============================================================================
# CONFIGURAÇÃO — edite apenas esta seção
# =============================================================================

# ---- Arquivos ----
PATHS = {
    "class_raster": "classificacao/LULC_7_cidades_2025-07-10_2025-07-30_projected.tif",
    "biomass_raster": "metricas/Biomass_sete_cidades_projected.tif",
    "vector_cities": "shapefile/sete_cidades.shp",
    "city_field": "NM_MUN",
    "outdir": "./dados_gerados",
}

# ---- Cidades (None = todas) ----
CITIES_FILTER: Optional[List[str]] = None
# Exemplos: None  |  ["Lavras"]  |  ["Lavras", "Varginha", "Alfenas"]

# ---- O que rodar ----
RUN_VIOLIN = True   # Gráfico de violino (biomassa por classe de uso, cidades combinadas)
RUN_MORAN = True    # Moran's I + scatter por cidade (resolução nativa da biomassa)

# ---- Opções do gráfico de violino ----
PLOT_TYPES = ["violin"]   # Opções: "violin" | "bar" | "box"
EXCLUDE_CLASSES_VIOLIN = [0]   # Ex.: [0] = Água

# ---- Opções do Moran (se RUN_MORAN = True) ----
MORAN_NATIVE_RESOLUTION = True   # True = resolução nativa; False = reamostrado 10 m
MORAN_PERMUTATIONS = 999
MORAN_SAVE_SCATTER = True

# ---- Nomes das classes (LULC) ----
CLASS_MAP = {
    0: "Água",
    1: "Urbano",
    2: "Solo",
    3: "Vegetação",
    4: "Agro/Pasto",
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

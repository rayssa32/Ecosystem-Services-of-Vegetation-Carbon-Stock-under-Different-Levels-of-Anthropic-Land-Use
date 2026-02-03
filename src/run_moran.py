"""
Moran's I global por cidade.

Modos: use_native_resolution=True (resolução nativa) ou False (10 m).
Uso: python -m src.run_moran
"""

import os

from .config import PathsConfig, MoranConfig
from .pipeline.moran_pipeline import run_moran_analysis


def main() -> None:
    paths = PathsConfig(
        class_raster_path="classificacao/LULC_7_cidades_2025-07-10_2025-07-30_projected.tif",
        biomass_raster_path="metricas/Biomass_sete_cidades_projected.tif",
        vector_cities_path="shapefile/sete_cidades.shp",
        city_field="NM_MUN",
        outdir="dados_gerados",
    )
    moran_cfg = MoranConfig(
        use_native_resolution=True,
        cities_filter=None,  # None = todas; ou ["Lavras", "Varginha"]
        permutations=999,
        contiguity="rook",
        save_scatter_plots=True,
    )

    required = [paths.biomass_raster_path, paths.vector_cities_path]
    if not moran_cfg.use_native_resolution:
        required.append(paths.class_raster_path)
    for p in required:
        if not os.path.exists(p):
            print(f"[ERRO] Não encontrado: {p}")
            return

    print(f"Moran's I — {moran_cfg.permutations} permutações\n")
    df = run_moran_analysis(paths, moran_cfg)
    if not df.empty:
        moran_dir = os.path.join(paths.outdir, "moran")
        csv_name = "moran_global_por_cidade_nativo.csv" if moran_cfg.use_native_resolution else "moran_global_por_cidade.csv"
        print(f"\n{os.path.join(moran_dir, csv_name)}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()

"""Pipeline para Moran's I global por cidade (biomassa na resolução nativa ou 10 m)."""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import mapping

from ..config import PathsConfig, MoranConfig
from ..data.raster_loader import RasterLoader
from ..data.vector_loader import VectorLoader
from ..processing.moran import moran_global, moran_scatter_plot


def run_moran_analysis(
    paths: PathsConfig,
    moran_cfg: MoranConfig,
) -> pd.DataFrame:
    """
    Executa Moran's I por cidade e gera scatter plots e CSV.

    Modos: use_native_resolution=True (biomassa na resolução nativa)
    ou False (biomassa reamostrada para 10 m, grade do LULC).
    """
    moran_dir = os.path.join(paths.outdir, "moran")
    os.makedirs(moran_dir, exist_ok=True)

    loader = RasterLoader(resample_mode="bilinear")
    vector_loader = VectorLoader()
    suffix = "_nativo" if moran_cfg.use_native_resolution else ""

    if moran_cfg.use_native_resolution:
        with rasterio.open(paths.biomass_raster_path) as src_biomass:
            gdf = vector_loader.load_cities(
                paths.vector_cities_path, paths.city_field, src_biomass.crs
            )
            results = _process_cities(
                gdf, None, src_biomass, loader, paths, moran_cfg, moran_dir, suffix,
            )
    else:
        with loader.load_classification_raster(paths.class_raster_path) as src_class, \
                rasterio.open(paths.biomass_raster_path) as src_biomass:
            gdf = vector_loader.load_cities(
                paths.vector_cities_path, paths.city_field, src_class.crs
            )
            results = _process_cities(
                gdf, src_class, src_biomass, loader, paths, moran_cfg, moran_dir, suffix,
            )

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    csv_name = "moran_global_por_cidade_nativo.csv" if moran_cfg.use_native_resolution else "moran_global_por_cidade.csv"
    csv_path = os.path.join(moran_dir, csv_name)
    df.to_csv(csv_path, index=False)
    return df


def _process_cities(
    gdf,
    src_class,
    src_biomass,
    loader: RasterLoader,
    paths: PathsConfig,
    moran_cfg: MoranConfig,
    moran_dir: str,
    suffix: str,
) -> List[dict]:
    results = []
    for _, row in gdf.iterrows():
        city = str(row[paths.city_field]).strip()
        if moran_cfg.cities_filter is not None and city not in moran_cfg.cities_filter:
            continue
        if row.geometry is None or row.geometry.is_empty:
            continue
        geom = [mapping(row.geometry)]

        if moran_cfg.use_native_resolution:
            try:
                biomass_clip, _ = loader.clip_raster_native(src_biomass, geom)
            except ValueError:
                print(f"[SKIP] {city}: geometria fora do raster.")
                continue
        else:
            try:
                class_clip, class_transform = loader.clip_classification(src_class, geom)
            except ValueError:
                print(f"[SKIP] {city}: geometria fora do raster.")
                continue
            biomass_clip = loader.clip_metric_raster(
                src_biomass, src_class, geom, class_transform, class_clip.shape,
            )

        try:
            mi, y, w = moran_global(
                biomass_clip,
                contiguity=moran_cfg.contiguity,
                permutations=moran_cfg.permutations,
                two_tailed=True,
            )
        except ValueError as e:
            print(f"[SKIP] {city}: {e}")
            continue

        I = mi.I
        p_val = mi.p_sim if moran_cfg.permutations else getattr(mi, "p_norm", None)
        n_unique = len(np.unique(y))
        results.append({
            "cidade": city,
            "Moran_I": round(I, 6),
            "p_valor": round(p_val, 6) if p_val is not None else None,
            "n_pixels": len(y),
            "n_valores_unicos": n_unique,
            "modo": "nativo" if moran_cfg.use_native_resolution else "10m",
        })
        print(f"{city}: I = {I:.4f}, p = {p_val:.4f}, n = {len(y)}")

        if moran_cfg.save_scatter_plots:
            safe_name = city.replace(" ", "_")
            plot_path = os.path.join(moran_dir, f"moran_scatter_{safe_name}{suffix}.png")
            moran_scatter_plot(
                mi,
                title=city,
                xlabel="Biomassa (padronizada)",
                ylabel="Lag espacial (W·z)",
                save_path=plot_path,
            )
    return results

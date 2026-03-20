# Ecosystem Services of Vegetation Carbon Stock under Different Levels of Anthropic Land Use

Code and methodological description for the analytical stage of the research **"Ecosystem Services of Vegetation Carbon Stock under Different Intensities of Anthropogenic Land Use"**.

The goal is to compare **GPP**, **NPP**, and **Biomass** against different **land uses** derived from **supervised classification** of Sentinel-2 imagery, cross-referencing with MODIS products in TIFF format and analyzing them with inferential tests and visualizations.

---

## What the project does

The pipeline:

1. **Spatial intersection** of:
   - Land use / land cover raster (Sentinel-2 classification);
   - MODIS rasters (GPP, NPP, Biomass);
   - Municipality shapefile.

2. **Reprojection and clipping** of metric rasters to the CRS and grid of the class raster (the classification raster is the coordinate reference).

3. **Statistics by class and municipality**: mean, median, standard deviation, sum, and count per land use class; totals (sum × pixel area).

4. **Inferential tests**: ANOVA or Kruskal–Wallis depending on assumptions; post-hoc (Tukey or Dunn–Holm); effect sizes (η² or ε²).

5. **Plots**:
   - **Violin** (and optionally bar/box): biomass distribution by land use class, per city or all cities combined.
   - **Sankey**: land use → biomass class (quantile-based); proportional flows, thickness = count or % of pixels; one diagram per city or one combined (all cities).
   - **Moran's I**: spatial autocorrelation of biomass per municipality (Global Moran's I, p-value by permutation, optional Moran scatter plot).

6. **Export** of CSVs and images to `dados_gerados/`.

---

## Project organization

Only **`main.py`** lives at the project root; the rest of the code is under **`src/`**.

```
project/
├── main.py                 # Single entry point: configure and run python main.py
├── src/
│   ├── config.py           # PathsConfig, MoranConfig, AnalysisConfig
│   ├── data/               # Raster and vector loading
│   │   ├── raster_loader.py
│   │   └── vector_loader.py
│   ├── processing/        # Aggregation, statistics, Moran's I, biomass classes
│   │   ├── aggregator.py
│   │   ├── statistics.py
│   │   ├── moran.py
│   │   └── biomass_classes.py
│   ├── pipeline/          # Workflow orchestration
│   │   ├── analysis_pipeline.py   # Violin, bar, box, stacked bar
│   │   └── moran_pipeline.py     # Moran's I per city
│   ├── visualization/     # Plot generation
│   │   ├── plotter.py
│   │   ├── graphics_factory.py
│   │   └── sankey_plotter.py
│   ├── utils/
│   ├── run_moran.py       # Moran only: python -m src.run_moran
│   └── reproject_raster.py # Reproject rasters: python -m src.reproject_raster
├── shapefile/             # Municipality shapefile
└── dados_gerados/         # Outputs (CSVs and PNGs)
```

- **Configuration**: `PathsConfig` (paths), `MoranConfig` (Moran), `SankeyConfig` (Sankey), `AnalysisConfig` (violin/bar/box pipeline). `main.py` builds these from the CONFIGURATION section at the top of the file.
- **Data**: `RasterLoader` (load, clip, resample, native-resolution clip) and `VectorLoader`.
- **Processing**: aggregation by class, tests (ANOVA/Kruskal, Tukey/Dunn), Moran's I, biomass quantile classification for Sankey.
- **Visualization**: plotters by type (violin, bar, box, stacked bar, Sankey) via factory or dedicated module.
- **Pipeline**: `AnalysisPipeline` for violin and Sankey; `run_moran_analysis()` for Moran per city. Execution order: violin → Sankey → Moran.

Further architecture details are in **`ARCHITECTURE.md`**.

---

## How to use

### Requirements

- Python **3.10+**
- Main dependencies: `numpy`, `pandas`, `rasterio`, `geopandas`, `shapely`, `affine`, `scipy`, `statsmodels`, `scikit-posthocs`, `matplotlib`

For **Moran's I** and **Sankey** you also need: `libpysal`, `esda`, `plotly` (and the rest of the project dependencies to run the `src` package). Use **`requirements-moran.txt`** for these deps; on PEP 668 systems (e.g. Fedora) use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install -r requirements-moran.txt
```

(Install the main project dependencies in the same environment if needed.)

### Configuration

Edit only the **CONFIGURATION** section at the top of **`main.py`**:

- **PATHS**: class raster path, biomass raster path, cities shapefile path, municipality name field, output directory (`outdir`).
- **CITIES_FILTER**: `None` for all cities, or a list of names, e.g. `["Lavras", "Varginha"]`.
- **RUN_VIOLIN** / **RUN_SANKEY** / **RUN_MORAN**: enable or disable violin, Sankey, and Moran analyses (order: violin → Sankey → Moran).
- **PLOT_TYPES**: plot types (e.g. `["violin"]`, `["bar"]`, `["box"]`).
- **EXCLUDE_CLASSES_VIOLIN**: classes to exclude from violin plots (e.g. `[0]` for Water).
- **Sankey**: `SANKEY_PER_CITY` (True = one Sankey per city; False = one combined), `SANKEY_N_QUANTILES` (e.g. 3 → Low/Medium/High), `SANKEY_USE_PERCENT` (thickness = % or count of pixels).
- **Moran**: `MORAN_NATIVE_RESOLUTION` (True = biomass native resolution; False = resampled 10 m), `MORAN_PERMUTATIONS`, `MORAN_SAVE_SCATTER`.
- **CLASS_MAP**: class id → label (e.g. `{0: "Water", 1: "Urban", ...}`).

The **classification (Sentinel) raster** must be in a **projected CRS (meters)**; MODIS and shapefile are reprojected to the class raster CRS when needed.

### Running

**Full analysis** (violin and/or Moran according to flags):

```bash
python main.py
```

**Moran's I only** (uses default config in `src.run_moran`):

```bash
python -m src.run_moran
```

**Reproject rasters only**:

```bash
python -m src.reproject_raster
```

### Outputs

All outputs go to **PATHS["outdir"]** (default: `./dados_gerados`).

| Output | Description |
|--------|-------------|
| `dados_gerados/<City>_stats_por_classe.csv` | Statistics by class and metric per city. |
| `dados_gerados/todas_cidades_stats_por_classe.csv` | Aggregated statistics for all cities. |
| `dados_gerados/stats/resumo_inferencial_por_cidade.csv` | Summary: city, metric, global test, p-value, effect size. |
| `dados_gerados/stats/pairwise_<city>_<metric>_tukey.csv` / `_dunn_holm.csv` | Post-hoc comparisons. |
| `dados_gerados/<city>_<metric>_violin.png` (and similar) | Plots per city/metric. |
| `dados_gerados/all_cities_<metric>_violin_combined.png` | Combined violin (all cities). |
| `dados_gerados/sankey/sankey_<City>.html` (or `sankey_all_cities.html`) | Sankey: land use → biomass class (interactive HTML). |
| `dados_gerados/moran/moran_global_por_cidade.csv` or `moran_global_por_cidade_nativo.csv` | Global Moran's I per city. |
| `dados_gerados/moran/moran_scatter_<City>.png` | Moran scatter per city (if enabled). |

On main pipeline plots, the annotation box shows the test (ANOVA or Kruskal), p-value, effect size (η² or ε²), and ★ when p < 0.05.

---

## Statistical methods

- **Diagnostics**: normality (Shapiro), homogeneity of variances (Levene, median).
- **Global comparison**: one-way ANOVA or Kruskal–Wallis depending on assumptions.
- **Post-hoc**: Tukey HSD (ANOVA) or Dunn with Holm adjustment (Kruskal).
- **Effect sizes**: η² (ANOVA) and ε² (Kruskal).
- **Moran's I**: contiguity weights (rook/queen), Global Moran's I with permutation p-value.

---

## CRS and data

- The **class raster** is the CRS reference; it is not reprojected by the code — it must be in a projected CRS (meters).
- **MODIS rasters**: reprojected to the class raster CRS (e.g. `WarpedVRT`).
- **Shapefile**: reprojected to the class raster CRS.

If the class raster is not projected, the script may fail.

---

## Troubleshooting

| Problem | Likely cause | Solution |
|---------|---------------|----------|
| `ValueError: geometries do not overlap` | Polygon outside raster extent | Check CRS and data extent. |
| Empty CSVs | Class with no valid pixels | Check `nodata` and raster masks. |
| Empty or missing plots | Missing data or flags off | Check paths and `RUN_VIOLIN` / plot options. |
| `ImportError` (rasterio/geopandas) | GDAL/Fiona missing | Install via conda-forge or system packages. |
| p-value NaN | Too few points per group (< 10) | Increase area or `MIN_N_FOR_TESTS` in config. |

---

## Statistical references

- Field, A. (2018). Discovering Statistics Using R. SAGE.
- Anselin, L. (1995). Local Indicators of Spatial Association—LISA. Geographical Analysis.
- Fortin, M.-J.; Dale, M. R. T. (2014). Spatial Analysis: A Guide for Ecologists. Cambridge University Press.
- Zar, J. H. (2010). *Biostatistical Analysis.* Pearson.
- Legendre & Fortin (1989). *Spatial pattern and ecological analysis.* Vegetatio.
- Goslee & Urban (2007). *The ecodist package for dissimilarity-based analysis of ecological data.* Journal of Statistical Software.

---

## Authors

**Rayssa de Oliveira Dias and Luiz Felipe Sá**

---

## License

This project is under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 (CC BY-NC-ND 4.0)**. Copy and redistribution with attribution are allowed; commercial use and derivatives require permission. See `LICENSE` for details.

*Note: this workflow is part of ongoing research; license terms may be updated after publication.*

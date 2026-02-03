# Architecture Overview

This document describes the clean architecture implementation of the Ecosystem Services analysis pipeline, with a focus on extensible graphics generation.

## Project Structure

```
project/
├── src/                          # Main source code package
│   ├── __init__.py              # Package exports
│   ├── config.py                # Configuration management
│   ├── data/                    # Data access layer
│   │   ├── __init__.py
│   │   ├── raster_loader.py     # Raster data loading and clipping
│   │   └── vector_loader.py     # Vector (shapefile) data loading
│   ├── processing/              # Business logic layer
│   │   ├── __init__.py
│   │   ├── aggregator.py        # Data aggregation by classes
│   │   └── statistics.py        # Statistical analysis and tests
│   ├── visualization/           # Graphics generation layer
│   │   ├── __init__.py
│   │   ├── plotter.py           # Plotting base classes and implementations
│   │   └── graphics_factory.py  # Factory for creating different plot types
│   ├── pipeline/                # Orchestration layer
│   │   ├── __init__.py
│   │   ├── analysis_pipeline.py # Main pipeline (violin, bar, box, stacked bar)
│   │   └── moran_pipeline.py    # Moran's I por cidade
│   ├── run_moran.py             # Moran's I (python -m src.run_moran)
│   ├── reproject_raster.py      # Reprojetar rasters (python -m src.reproject_raster)
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       └── raster_utils.py      # Raster utility functions
├── main.py                      # Único entry point na raiz: configure e rode python main.py
└── ...                          # Data files and outputs
```

## Architecture Principles

### 1. **Separation of Concerns**
Each module has a single, well-defined responsibility:
- **Data Layer**: Handles all I/O operations (raster, vector)
- **Processing Layer**: Contains business logic (aggregation, statistics)
- **Visualization Layer**: Focuses exclusively on graphics generation
- **Pipeline Layer**: Orchestrates the workflow

### 2. **Dependency Injection**
Configuration is injected through constructors, making components testable and flexible.

### 3. **Extensibility**
The visualization layer uses:
- **Abstract Base Classes** (`BasePlotter`) for plot types
- **Factory Pattern** (`GraphicsFactory`) for creating plotters
- **Strategy Pattern** for different plotting implementations

### 4. **Clean Interfaces**
Clear boundaries between layers with well-defined interfaces and minimal coupling.

## Key Components

### Configuration (`src/config.py`)
- `PathsConfig`: Paths comuns (rasters, shapefile, outdir)
- `MoranConfig`: Opções do Moran's I (resolução nativa, permutações, etc.)
- `AnalysisConfig`: Parâmetros do pipeline (plot_types, bar, violin, stacked bar)
- Single source of truth for settings

### Data Layer (`src/data/`)
- **RasterLoader**: Handles raster operations (loading, clipping, resampling)
- **VectorLoader**: Handles vector/shapefile operations
- Encapsulates all file I/O and data format specifics

### Processing Layer (`src/processing/`)
- **DataAggregator**: Aggregates statistics by land use classes
- **StatisticsAnalyzer**: Performs inferential tests (ANOVA, Kruskal-Wallis)
- Business logic separated from I/O

### Visualization Layer (`src/visualization/`)
- **BasePlotter**: Abstract interface for all plot types
- **BarPlotter**: Bar plots with statistical annotations
- **BoxPlotter**: Box plots (placeholder for future implementation)
- **ViolinPlotter**: Violin plots (placeholder for future implementation)
- **Plotter**: Delegates to specific plotter implementations
- **GraphicsFactory**: Factory for creating and managing plotters

### Pipeline (`src/pipeline/`)
- **AnalysisPipeline**: Orchestrates analysis (violin, bar, box, stacked bar)
- **run_moran_analysis()**: Moran's I por cidade, scatter plots e CSV (resolução nativa ou 10 m)

## Adding New Plot Types

To add a new plot type (e.g., scatter plots, heatmaps):

1. **Create a new plotter class** in `src/visualization/plotter.py`:
   ```python
   class ScatterPlotter(BasePlotter):
       def plot(self, df, metric, city, label_col, outdir, annotation=None):
           # Implementation here
           pass
   ```

2. **Update the factory method** in `Plotter._create_plotter()`:
   ```python
   elif plot_type == "scatter":
       return ScatterPlotter()
   ```

3. **Use it**:
   ```python
   graphics = GraphicsFactory(config)
   plotter = graphics.create_plotter("scatter")
   plotter.plot(df, metric, city, label_col, outdir)
   ```

## Usage Example

```python
from src import AnalysisConfig, AnalysisPipeline

# Configure analysis
config = AnalysisConfig(
    make_plots=True,
    run_inferential_tests=True,
    # ... other parameters
)

# Initialize pipeline
pipeline = AnalysisPipeline(config)

# Run analysis
results = pipeline.run(
    class_raster_path="path/to/classification.tif",
    metrics_rasters={"GPP": "path/to/gpp.tif", ...},
    vector_cities_path="path/to/cities.shp",
    city_field="NM_MUN",
    class_map={1: "Vegetação", 2: "Urbano", ...}
)
```

## Benefits of This Architecture

1. **Maintainability**: Clear separation makes code easy to understand and modify
2. **Testability**: Each component can be tested independently
3. **Extensibility**: Easy to add new plot types, statistics, or data sources
4. **Reusability**: Components can be reused in different contexts
5. **Flexibility**: Configuration-driven design allows easy customization

## Future Enhancements

- Add more plot types (scatter, heatmap, violin, box plots with raw data)
- Implement custom plotter registration in GraphicsFactory
- Add CLI interface for configuration
- Add unit tests for each module
- Implement caching for expensive operations
- Add parallel processing support

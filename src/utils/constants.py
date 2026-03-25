"""Constants for visualization and analysis."""

# LULC raster: class 0 = NULL / no label; valid land-cover codes are 1–5.
NULL_LULC_CLASS = 0
# Água (water); often excluded from violin/Sankey alongside NULL
WATER_LULC_CLASS = 1

# Land use class color mapping (keys match raster codes; NULL has no plot color)
# 1-Água | 2-Urbano | 3-Solo | 4-Vegetação | 5-Agro/Pasto
CLASS_COLORS = {
    1: "#3b83bd",  # Água (Water)
    2: "#8c8c8c",  # Urbano (Urban)
    3: "#c8a165",  # Solo (Soil)
    4: "#2ca25f",  # Vegetação (Vegetation)
    5: "#a1d99b",  # Agro/Pasto (Agriculture/Pasture)
}

# Default fallback colors if class code not in CLASS_COLORS (order: 1 → 5)
DEFAULT_CLASS_COLORS = ["#3b83bd", "#8c8c8c", "#c8a165", "#2ca25f", "#a1d99b"]

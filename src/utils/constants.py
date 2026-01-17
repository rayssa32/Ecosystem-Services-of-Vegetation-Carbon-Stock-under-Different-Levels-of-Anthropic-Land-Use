"""Constants for visualization and analysis."""

# Land use class color mapping
# Colors: 0-Água | 1-Urbano | 2-Solo | 3-Vegetação | 4-Agro/Pasto
CLASS_COLORS = {
    0: "#3b83bd",  # Água (Water)
    1: "#8c8c8c",  # Urbano (Urban)
    2: "#c8a165",  # Solo (Soil)
    3: "#2ca25f",  # Vegetação (Vegetation)
    4: "#a1d99b",  # Agro/Pasto (Agriculture/Pasture)
}

# Default fallback colors if class code not in CLASS_COLORS
DEFAULT_CLASS_COLORS = ["#3b83bd", "#8c8c8c", "#c8a165", "#2ca25f", "#a1d99b"]

"""Constants for visualization and analysis."""

# LULC raster: class 0 = NULL / no label; valid land-cover codes are 1–5.
NULL_LULC_CLASS = 0
# Água (water); often excluded from violin/Sankey alongside NULL
WATER_LULC_CLASS = 1

# Land use class color mapping (keys match raster codes; NULL has no plot color)
# 1-Água | 2-Áreas urbanizadas | 3-Solo exposto | 4-Áreas de vegetação natural | 5-Áreas antrópicas agrícolas
CLASS_COLORS = {
    1: "#3b83bd",  # Água
    2: "#8c8c8c",  # Áreas urbanizadas
    3: "#c8a165",  # Solo exposto
    4: "#2ca25f",  # Áreas de vegetação natural
    5: "#a1d99b",  # Áreas antrópicas agrícolas
}

# Default fallback colors if class code not in CLASS_COLORS (order: 1 → 5)
DEFAULT_CLASS_COLORS = ["#3b83bd", "#8c8c8c", "#c8a165", "#2ca25f", "#a1d99b"]

# Rótulos em português para métricas e tipos de valor exibidos nos gráficos
METRIC_LABELS = {
    "Biomass": "Biomassa",
    "GPP": "PPG",
    "NPP": "PPN",
    "Area": "Área",
}

VALUE_TYPE_LABELS = {
    "mean": "média",
    "sum": "soma",
    "count": "contagem",
    "total_kg": "total (kg)",
    "percentage": "porcentagem",
}


def rotulo_metrica(nome: str) -> str:
    """Retorna o rótulo em português de uma métrica (ex.: Biomass → Biomassa)."""
    return METRIC_LABELS.get(nome, nome)


def rotulo_tipo_valor(tipo: str) -> str:
    """Retorna o rótulo em português de um tipo de valor (ex.: mean → média)."""
    return VALUE_TYPE_LABELS.get(tipo, tipo)

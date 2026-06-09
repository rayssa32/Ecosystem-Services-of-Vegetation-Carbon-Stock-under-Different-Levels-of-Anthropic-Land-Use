"""
Ponto único de entrada do projeto.

Como usar:
  1. Instale as dependências (veja README.md)
  2. Edite apenas a seção CONFIGURAÇÃO abaixo
  3. Execute no terminal: python main.py
"""

from src.user_runner import ConfiguracaoUsuario, executar_analises


# =============================================================================
# CONFIGURAÇÃO — edite apenas esta seção
# =============================================================================

# ---- Passo 1: onde estão seus arquivos? ----
ARQUIVO_USO_SOLO = "classificacao/LULC_7Cidades_10m_20250710_20250730_projected.tif"
ARQUIVO_BIOMASSA = "metricas/Biomass_sete_cidades_projected.tif"
ARQUIVO_CIDADES = "shapefile/sete_cidades.shp"
COLUNA_NOME_CIDADE = "NM_MUN"   # coluna do shapefile com o nome da cidade
PASTA_SAIDA = "./dados_gerados"  # pasta onde os resultados serão salvos

# ---- Passo 2: quais cidades analisar? ----
# Lista vazia [] = todas as cidades do shapefile
CIDADES = []
# Exemplos:
# CIDADES = ["Lavras"]
# CIDADES = ["Lavras", "Varginha", "Alfenas"]

# ---- Passo 3: predefinição rápida (opcional) ----
# Descomente UMA linha abaixo, ou deixe PRESET = None para usar GERAR_* manualmente
PRESET = None
# PRESET = "rapido"     # só barras de cobertura do solo
# PRESET = "artigo"     # gráfico de biomassa + diagrama de fluxo
# PRESET = "completo"   # todas as análises

# ---- Passo 4: o que gerar? (True = sim, False = não) ----
# Ignorado se PRESET estiver definido acima.

GERAR_GRAFICO_BIOMASSA_POR_USO = False
# Saída: dados_gerados/all_classes_Carbono_box_by_class.png (com fração ≠ 1)

GERAR_DIAGRAMA_FLUXO = False
# Saída: dados_gerados/sankey/sankey_<Cidade>.html (ou sankey_all_cities.html)

GERAR_BARRAS_USO_SOLO = False
# Saída: dados_gerados/stacked_bar_land_use_percentage.png e CSVs de estatísticas

GERAR_INDICE_SHANNON = False
# Saída: CSVs com índice de Shannon (H') e equitabilidade de Pielou (J')

GERAR_AUTOCORRELACAO_MORAN = False
# Saída: dados_gerados/moran/ (CSV e gráficos de dispersão, se habilitado)

# ---- Opções do gráfico de biomassa (violin / bar / box) ----
TIPOS_GRAFICO_BIOMASSA = ["violin", "box"]  # opções: "violin", "bar", "box"
# Biomassa × fração → estoque de carbono nos gráficos. Use 1.0 se ARQUIVO_BIOMASSA já for carbono.
FRACAO_CARBONO_NA_BIOMASSA = 0.47

# Classes a ignorar (use os nomes de MAPA_CLASSES, não números)
EXCLUIR_DO_GRAFICO_BIOMASSA = ["NULL", "Água"]
EXCLUIR_DO_DIAGRAMA_FLUXO = ["NULL", "Água"]
EXCLUIR_DAS_BARRAS_USO_SOLO = ["NULL"]
EXCLUIR_DO_INDICE_SHANNON = []  # NULL é sempre excluído automaticamente

# ---- Opções do diagrama de fluxo (Sankey) ----
SANKEY_UM_POR_CIDADE = True       # True = um diagrama por cidade; False = um geral
SANKEY_NUMERO_CLASSES_BIOMASSA = 3  # ex.: 3 = Baixa, Média, Alta
SANKEY_USAR_PORCENTAGEM = True    # True = espessura em %; False = número de pixels

# ---- Opções da autocorrelação Moran ----
MORAN_RESOLUCAO_NATIVA = True     # True = resolução original da biomassa; False = 10 m
MORAN_SALVAR_GRAFICO_DISPERSAO = True

# ---- Nomes das classes de uso do solo (código no raster → nome legível) ----
MAPA_CLASSES = {
    0: "NULL",
    1: "Água",
    2: "Áreas urbanizadas",
    3: "Solo exposto",
    4: "Áreas de vegetação natural",
    5: "Áreas antrópicas agrícolas",
}

# =============================================================================
# Não é necessário editar abaixo desta linha
# =============================================================================


def main() -> None:
    cfg = ConfiguracaoUsuario(
        arquivo_uso_solo=ARQUIVO_USO_SOLO,
        arquivo_biomassa=ARQUIVO_BIOMASSA,
        arquivo_cidades=ARQUIVO_CIDADES,
        coluna_nome_cidade=COLUNA_NOME_CIDADE,
        pasta_saida=PASTA_SAIDA,
        cidades=CIDADES,
        preset=PRESET,
        gerar_grafico_biomassa=GERAR_GRAFICO_BIOMASSA_POR_USO,
        gerar_diagrama_fluxo=GERAR_DIAGRAMA_FLUXO,
        gerar_barras_uso_solo=GERAR_BARRAS_USO_SOLO,
        gerar_indice_shannon=GERAR_INDICE_SHANNON,
        gerar_autocorrelacao_moran=GERAR_AUTOCORRELACAO_MORAN,
        tipos_grafico_biomassa=TIPOS_GRAFICO_BIOMASSA,
        fracao_carbono_biomassa=FRACAO_CARBONO_NA_BIOMASSA,
        excluir_do_grafico_biomassa=EXCLUIR_DO_GRAFICO_BIOMASSA,
        excluir_do_diagrama_fluxo=EXCLUIR_DO_DIAGRAMA_FLUXO,
        excluir_das_barras_uso_solo=EXCLUIR_DAS_BARRAS_USO_SOLO,
        excluir_do_indice_shannon=EXCLUIR_DO_INDICE_SHANNON,
        sankey_um_por_cidade=SANKEY_UM_POR_CIDADE,
        sankey_numero_classes_biomassa=SANKEY_NUMERO_CLASSES_BIOMASSA,
        sankey_usar_porcentagem=SANKEY_USAR_PORCENTAGEM,
        moran_resolucao_nativa=MORAN_RESOLUCAO_NATIVA,
        moran_salvar_grafico_dispersao=MORAN_SALVAR_GRAFICO_DISPERSAO,
        mapa_classes=MAPA_CLASSES,
    )
    executar_analises(cfg)


if __name__ == "__main__":
    main()

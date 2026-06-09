"""Execução simplificada a partir de main.py (usuários sem conhecimento de código)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import AnalysisConfig, MoranConfig, PathsConfig, SankeyConfig
from .pipeline import AnalysisPipeline, run_moran_analysis
from .utils.constants import NULL_LULC_CLASS

PRESETS: Dict[str, Dict[str, bool]] = {
    "rapido": {
        "grafico_biomassa": False,
        "diagrama_fluxo": False,
        "barras_uso_solo": True,
        "indice_shannon": False,
        "autocorrelacao_moran": False,
    },
    "artigo": {
        "grafico_biomassa": True,
        "diagrama_fluxo": True,
        "barras_uso_solo": False,
        "indice_shannon": False,
        "autocorrelacao_moran": False,
    },
    "completo": {
        "grafico_biomassa": True,
        "diagrama_fluxo": True,
        "barras_uso_solo": True,
        "indice_shannon": True,
        "autocorrelacao_moran": True,
    },
}

ROTULOS_ANALISE: Dict[str, str] = {
    "grafico_biomassa": "Gráfico de biomassa por uso do solo",
    "diagrama_fluxo": "Diagrama de fluxo (uso do solo → biomassa)",
    "barras_uso_solo": "Barras empilhadas de cobertura do solo",
    "indice_shannon": "Índice de Shannon e equitabilidade",
    "autocorrelacao_moran": "Autocorrelação espacial (Moran)",
}


@dataclass
class ConfiguracaoUsuario:
    """Configuração vinda exclusivamente da seção CONFIGURAÇÃO do main.py."""

    arquivo_uso_solo: str
    arquivo_biomassa: str
    arquivo_cidades: str
    coluna_nome_cidade: str
    pasta_saida: str
    cidades: List[str] = field(default_factory=list)
    preset: Optional[str] = None
    gerar_grafico_biomassa: bool = False
    gerar_diagrama_fluxo: bool = False
    gerar_barras_uso_solo: bool = False
    gerar_indice_shannon: bool = False
    gerar_autocorrelacao_moran: bool = False
    tipos_grafico_biomassa: List[str] = field(default_factory=lambda: ["violin"])
    excluir_do_grafico_biomassa: List[str] = field(
        default_factory=lambda: ["NULL", "Água"]
    )
    excluir_do_diagrama_fluxo: List[str] = field(
        default_factory=lambda: ["NULL", "Água"]
    )
    excluir_das_barras_uso_solo: List[str] = field(default_factory=lambda: ["NULL"])
    excluir_do_indice_shannon: List[str] = field(default_factory=list)
    sankey_um_por_cidade: bool = True
    sankey_numero_classes_biomassa: int = 3
    sankey_usar_porcentagem: bool = True
    moran_resolucao_nativa: bool = True
    moran_salvar_grafico_dispersao: bool = True
    fracao_carbono_biomassa: float = 0.47
    mapa_classes: Dict[int, str] = field(default_factory=dict)


def _filtro_cidades(cidades: List[str]) -> Optional[List[str]]:
    if not cidades:
        return None
    return cidades


def _nomes_para_codigos(nomes: List[str], mapa_classes: Dict[int, str]) -> List[int]:
    nome_para_codigo = {nome: codigo for codigo, nome in mapa_classes.items()}
    codigos: List[int] = []
    for nome in nomes:
        if nome not in nome_para_codigo:
            opcoes = ", ".join(sorted(set(mapa_classes.values())))
            raise ValueError(
                f"Classe desconhecida: '{nome}'. Use nomes de MAPA_CLASSES: {opcoes}"
            )
        codigos.append(nome_para_codigo[nome])
    return codigos


def _codigos_exclusao_shannon(
    nomes_excluir: List[str], mapa_classes: Dict[int, str]
) -> List[int]:
    codigos = _nomes_para_codigos(nomes_excluir, mapa_classes)
    if NULL_LULC_CLASS not in codigos:
        codigos = [NULL_LULC_CLASS, *codigos]
    return codigos


def _analises_ativas(cfg: ConfiguracaoUsuario) -> Dict[str, bool]:
    if cfg.preset:
        preset = cfg.preset.strip().lower()
        if preset not in PRESETS:
            opcoes = ", ".join(f'"{p}"' for p in PRESETS)
            raise ValueError(f"PRESET inválido: '{cfg.preset}'. Opções: {opcoes}")
        return dict(PRESETS[preset])
    return {
        "grafico_biomassa": cfg.gerar_grafico_biomassa,
        "diagrama_fluxo": cfg.gerar_diagrama_fluxo,
        "barras_uso_solo": cfg.gerar_barras_uso_solo,
        "indice_shannon": cfg.gerar_indice_shannon,
        "autocorrelacao_moran": cfg.gerar_autocorrelacao_moran,
    }


def _paths_config(cfg: ConfiguracaoUsuario) -> PathsConfig:
    return PathsConfig(
        class_raster_path=cfg.arquivo_uso_solo,
        biomass_raster_path=cfg.arquivo_biomassa,
        vector_cities_path=cfg.arquivo_cidades,
        city_field=cfg.coluna_nome_cidade,
        outdir=cfg.pasta_saida,
    )


def _validar_arquivos(
    paths: PathsConfig, analises: Dict[str, bool], moran_resolucao_nativa: bool
) -> bool:
    precisa_uso_solo = any(
        analises[k]
        for k in (
            "grafico_biomassa",
            "diagrama_fluxo",
            "barras_uso_solo",
            "indice_shannon",
        )
    ) or (analises["autocorrelacao_moran"] and not moran_resolucao_nativa)

    arquivos: Dict[str, str] = {
        "Raster de biomassa": paths.biomass_raster_path,
        "Shapefile das cidades": paths.vector_cities_path,
    }
    if precisa_uso_solo:
        arquivos["Raster de uso do solo"] = paths.class_raster_path

    faltando = [
        f"{rotulo}: {caminho}"
        for rotulo, caminho in arquivos.items()
        if not os.path.exists(caminho)
    ]
    if faltando:
        print("[ERRO] Arquivo(s) não encontrado(s):")
        for item in faltando:
            print(f"  • {item}")
        print("\nVerifique os caminhos na seção CONFIGURAÇÃO do main.py.")
        return False
    return True


def _config_base(outdir: str) -> AnalysisConfig:
    return AnalysisConfig(
        resample_metrics="nearest",
        outdir=outdir,
        sample_per_class=5000,
        min_n_for_tests=10,
        alpha=0.05,
        rng_seed=42,
    )


def _imprimir_cabecalho(cfg: ConfiguracaoUsuario, analises: Dict[str, bool]) -> None:
    print("=" * 60)
    print("  Análise de serviços ecossistêmicos — execução")
    print("=" * 60)
    if cfg.preset:
        print(f"Predefinição: {cfg.preset}")
    if cfg.cidades:
        print(f"Cidades: {', '.join(cfg.cidades)}")
    else:
        print("Cidades: todas do shapefile")
    print(f"Pasta de saída: {cfg.pasta_saida}")
    print("\nAnálises selecionadas:")
    for chave, ativo in analises.items():
        marca = "✓" if ativo else " "
        print(f"  [{marca}] {ROTULOS_ANALISE[chave]}")
    print("=" * 60)
    print()


def _imprimir_resumo_saida(paths: PathsConfig, analises: Dict[str, bool]) -> None:
    print("\n" + "=" * 60)
    print("  Resumo dos resultados")
    print("=" * 60)
    if analises["grafico_biomassa"]:
        print(f"  • Gráficos de biomassa: {paths.outdir}/all_classes_*_by_class.png")
    if analises["diagrama_fluxo"]:
        print(f"  • Diagramas Sankey: {paths.outdir}/sankey/")
    if analises["barras_uso_solo"]:
        print(f"  • Barras de uso do solo: {paths.outdir}/stacked_bar_land_use_percentage.png")
        print(f"  • CSVs de estatísticas: {paths.outdir}/*_stats_por_classe.csv")
    if analises["indice_shannon"]:
        print(f"  • Índices Shannon/Pielou: {paths.outdir}/ (arquivos CSV)")
    if analises["autocorrelacao_moran"]:
        print(f"  • Moran (CSV e gráficos): {paths.outdir}/moran/")
    print("=" * 60)


def executar_analises(cfg: ConfiguracaoUsuario) -> None:
    """Ponto de entrada único chamado pelo main.py."""
    analises = _analises_ativas(cfg)

    if not any(analises.values()):
        print("[AVISO] Nenhuma análise selecionada.")
        print(
            "Ative GERAR_* no main.py ou escolha um PRESET "
            "('rapido', 'artigo', 'completo')."
        )
        return

    paths = _paths_config(cfg)
    cidades = _filtro_cidades(cfg.cidades)

    if not _validar_arquivos(paths, analises, cfg.moran_resolucao_nativa):
        return

    _imprimir_cabecalho(cfg, analises)

    excluir_biomassa = _nomes_para_codigos(
        cfg.excluir_do_grafico_biomassa, cfg.mapa_classes
    )
    excluir_fluxo = _nomes_para_codigos(
        cfg.excluir_do_diagrama_fluxo, cfg.mapa_classes
    )
    excluir_barras = _nomes_para_codigos(
        cfg.excluir_das_barras_uso_solo, cfg.mapa_classes
    )
    excluir_shannon = _codigos_exclusao_shannon(
        cfg.excluir_do_indice_shannon, cfg.mapa_classes
    )

    if analises["grafico_biomassa"]:
        print("→ Gerando gráfico de biomassa por uso do solo...")
        config = _config_base(paths.outdir)
        config.make_plots = True
        config.plot_types = cfg.tipos_grafico_biomassa
        config.make_stacked_bar_charts = False
        config.save_csv_files = False
        config.run_inferential_tests = False
        config.exclude_classes = excluir_biomassa
        config.biomass_carbon_fraction = cfg.fracao_carbono_biomassa
        AnalysisPipeline(config).run_violin_plots_analysis(
            class_raster_path=paths.class_raster_path,
            biomass_raster_path=paths.biomass_raster_path,
            vector_cities_path=paths.vector_cities_path,
            city_field=paths.city_field,
            class_map=cfg.mapa_classes,
            cities_filter=cidades,
        )
        print("[OK] Gráfico de biomassa concluído.\n")

    if analises["diagrama_fluxo"]:
        print("→ Gerando diagrama de fluxo (Sankey)...")
        sankey_cfg = SankeyConfig(
            per_city=cfg.sankey_um_por_cidade,
            n_quantiles=cfg.sankey_numero_classes_biomassa,
            use_percentage=cfg.sankey_usar_porcentagem,
        )
        config = _config_base(paths.outdir)
        config.make_plots = True
        config.plot_types = cfg.tipos_grafico_biomassa
        config.make_stacked_bar_charts = False
        config.save_csv_files = False
        config.run_inferential_tests = False
        config.exclude_classes = excluir_fluxo
        config.biomass_carbon_fraction = cfg.fracao_carbono_biomassa
        AnalysisPipeline(config).run_sankey_analysis(
            class_raster_path=paths.class_raster_path,
            biomass_raster_path=paths.biomass_raster_path,
            vector_cities_path=paths.vector_cities_path,
            city_field=paths.city_field,
            class_map=cfg.mapa_classes,
            cities_filter=cidades,
            sankey_config=sankey_cfg,
        )
        print("[OK] Diagrama de fluxo concluído.\n")

    if analises["barras_uso_solo"]:
        print("→ Gerando barras empilhadas de cobertura do solo...")
        config = _config_base(paths.outdir)
        config.make_plots = False
        config.make_stacked_bar_charts = True
        config.save_csv_files = True
        config.run_inferential_tests = False
        config.exclude_classes = excluir_barras
        AnalysisPipeline(config).run(
            class_raster_path=paths.class_raster_path,
            metrics_rasters=None,
            vector_cities_path=paths.vector_cities_path,
            city_field=paths.city_field,
            class_map=cfg.mapa_classes,
            cities_filter=cidades,
        )
        print("[OK] Barras de uso do solo concluídas.\n")

    if analises["indice_shannon"]:
        print("→ Calculando índice de Shannon e equitabilidade...")
        config = _config_base(paths.outdir)
        config.make_plots = False
        config.make_stacked_bar_charts = False
        config.save_csv_files = True
        config.run_inferential_tests = False
        config.exclude_classes = excluir_shannon
        AnalysisPipeline(config).run_shannon_index_analysis(
            class_raster_path=paths.class_raster_path,
            vector_cities_path=paths.vector_cities_path,
            city_field=paths.city_field,
            cities_filter=cidades,
        )
        print("[OK] Índice de Shannon concluído.\n")

    if analises["autocorrelacao_moran"]:
        print("→ Calculando autocorrelação espacial (Moran)...")
        moran_cfg = MoranConfig(
            use_native_resolution=cfg.moran_resolucao_nativa,
            cities_filter=cidades,
            permutations=999,
            contiguity="rook",
            save_scatter_plots=cfg.moran_salvar_grafico_dispersao,
            biomass_carbon_fraction=cfg.fracao_carbono_biomassa,
        )
        df = run_moran_analysis(paths, moran_cfg)
        if not df.empty:
            moran_dir = os.path.join(paths.outdir, "moran")
            csv_name = (
                "moran_global_por_cidade_nativo.csv"
                if moran_cfg.use_native_resolution
                else "moran_global_por_cidade.csv"
            )
            print(f"\n{os.path.join(moran_dir, csv_name)}")
            print(df.to_string(index=False))
        print("[OK] Autocorrelação Moran concluída.\n")

    _imprimir_resumo_saida(paths, analises)

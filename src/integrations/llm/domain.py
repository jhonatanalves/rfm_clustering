from __future__ import annotations

import html
from collections import Counter
import time
from typing import Annotated, Protocol

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict, StringConstraints

from src.observability.logger import get_logger
from src.integrations.llm.utils import chunk_dataframe, estimate_max_output_tokens
from src.integrations.llm.prompts import CLUSTER_ANALYSIS_TEMPLATE, JSON_REPAIR_TEMPLATE

LLM_MAX_PROMPT_CHARS = 20000

class DomainValidationError(ValueError):
    """Erro de validação do JSON no domínio de clusters."""


class LLMInputTooLargeError(ValueError):
    """Erro específico quando o payload (CSV) excede o limite seguro de tokens/caracteres."""


def build_cluster_naming_prompt(cluster_profile: pd.DataFrame, business_context: str) -> str:
    """
    Constrói o prompt para a LLM, otimizando o payload CSV e sanitizando o contexto.

    Arredonda floats para 2 casas decimais e verifica se o tamanho do CSV excede
    o limite seguro (LLM_MAX_PROMPT_CHARS).

    Args:
        cluster_profile (pd.DataFrame): Dados agregados dos clusters.
        business_context (str): Contexto de negócio fornecido pelo usuário.

    Returns:
        str: Prompt formatado pronto para envio.
    """
    prof_csv = cluster_profile.round(2).to_csv(index=False)

    if len(prof_csv) > LLM_MAX_PROMPT_CHARS:
        raise LLMInputTooLargeError("O volume de dados do chunk excede o limite seguro. Reduza o número de clusters, diminua o chunk_size ou simplifique colunas.")

    raw_context = (business_context or "").strip()
    # quote=False preserva aspas, mantendo a semântica do texto do usuário mas evitando injeção de tags
    safe_context = html.escape(raw_context[:2000], quote=False) if raw_context else "Não informado."

    return CLUSTER_ANALYSIS_TEMPLATE.format(
        business_context=safe_context,
        csv_data=prof_csv
    )


ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
LongText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=5, max_length=500)]
StrategyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3)]


class ClusterItemSchema(BaseModel):
    """Schema para validação de um único cluster retornado pela LLM."""
    model_config = ConfigDict(extra="forbid")

    ClusterId: int = Field(..., ge=0, description="ID numérico do cluster")
    SegmentoNome: ShortText = Field(..., description="Nome curto do segmento")
    SegmentoDescricao: LongText = Field(..., description="Descrição detalhada")
    Estrategias: list[StrategyText] = Field(..., min_length=3, max_length=3, description="Exatamente 3 estratégias")

    @field_validator("Estrategias")
    @classmethod
    def validate_strategies(cls, v: list[str]) -> list[str]:
        if len(v) != 3:
            raise ValueError(f"Esperado exatamente 3 estratégias válidas, recebido {len(v)}.")
        return v


class ClusterListSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clusters: list[ClusterItemSchema] = Field(..., min_length=1)

    @field_validator("clusters")
    @classmethod
    def check_unique_ids(cls, v: list[ClusterItemSchema]) -> list[ClusterItemSchema]:
        ids = [c.ClusterId for c in v]
        if len(ids) != len(set(ids)):
            dups = sorted([k for k, count in Counter(ids).items() if count > 1])
            raise ValueError(f"ClusterIds duplicados encontrados: {dups}")
        return v


def build_cluster_labels(cluster_profile: pd.DataFrame, llm_json: dict) -> pd.DataFrame:
    """
    Converte o JSON bruto da LLM para DataFrame, aplicando validação estrita de schema
    e verificando integridade referencial dos ClusterIds.

    Args:
        cluster_profile (pd.DataFrame): Dados originais para validação de IDs.
        llm_json (dict): Resposta JSON parseada da LLM.

    Returns:
        pd.DataFrame: DataFrame contendo nomes, descrições e estratégias validados.
    """
    try:
        validated = ClusterListSchema.model_validate(llm_json)
    except ValidationError as e:
        raise DomainValidationError(f"Erro de validação do schema JSON: {e}")

    rows = [item.model_dump() for item in validated.clusters]
    labels = pd.DataFrame(rows)

    expected = set(cluster_profile["ClusterId"].tolist())
    got = set(labels["ClusterId"].tolist())
    missing = expected - got
    if missing:
        raise DomainValidationError(f"LLM não retornou rótulos para ClusterId: {sorted(missing)}")

    extra_ids = got - expected
    if extra_ids:
        raise DomainValidationError(f"LLM retornou ClusterId inesperado (não existe nos dados): {sorted(extra_ids)}")

    return labels


def repair_prompt_for_invalid_json(original_prompt: str, last_text: str) -> str:
    """
    Prompt de reparo usado no retry quando a saída não for JSON parseável.

    Args:
        original_prompt (str): O prompt que gerou o erro.
        last_text (str): A resposta inválida recebida.

    Returns:
        str: Novo prompt solicitando correção do JSON.
    """
    return JSON_REPAIR_TEMPLATE.format(
        original_prompt=original_prompt,
        last_text=last_text
    )


class LLMProviderProtocol(Protocol):
    """Protocolo para tipagem do Provider (já que não temos o arquivo de providers)."""
    def generate_json(self, model: str, prompt: str, temperature: float, max_tokens: int, timeout: float) -> dict:
        ...


def orchestrate_cluster_labeling(
    provider: LLMProviderProtocol,
    cluster_profile: pd.DataFrame,
    business_context: str,
    model: str,
    temperature: float,
    chunk_size: int = 30
) -> pd.DataFrame:
    """
    Orquestra a rotulagem dos clusters dividindo o DataFrame em chunks.

    Implementa estratégia de retry adaptativo: se um chunk for muito grande para o prompt,
    reduz automaticamente o tamanho do lote (chunk_size) e tenta novamente.

    Args:
        provider (LLMProviderProtocol): Instância do provedor de LLM.
        cluster_profile (pd.DataFrame): Dados agregados dos clusters.
        business_context (str): Contexto do negócio.
        model (str): Nome do modelo a ser usado.
        temperature (float): Temperatura para geração.
        chunk_size (int): Tamanho inicial do lote de clusters por chamada.

    Returns:
        pd.DataFrame: DataFrame consolidado com todos os rótulos gerados.
    """
    logger = get_logger("domain.orchestrator")
    
    all_labels_dfs = []
    
    current_idx = 0
    total_rows = len(cluster_profile)
    
    BASE_FALLBACK = [30, 20, 10, 5, 3, 1]
    fallback_sizes = [s for s in BASE_FALLBACK if s <= chunk_size]

    logger.info(f"Iniciando processamento. Total clusters: {total_rows}. Chunk inicial: {chunk_size}")

    while current_idx < total_rows:
        chunk_success = False
        
        for size in fallback_sizes:
            end_idx = min(current_idx + size, total_rows)
            chunk = cluster_profile.iloc[current_idx:end_idx].copy()
            
            try:
                prompt = build_cluster_naming_prompt(chunk, business_context)
                
                n_clusters = len(chunk)
                max_tokens = estimate_max_output_tokens(n_clusters)
                
                logger.info(f"Processando chunk {current_idx}-{end_idx} (Size: {size}). Max tokens est: {max_tokens}")
                
                start_time = time.time()
                llm_json = provider.generate_json(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=30.0
                )
                duration = time.time() - start_time
                logger.info(f"LLM Response recebida. Duração: {duration:.2f}s")
                
                chunk_labels = build_cluster_labels(chunk, llm_json)
                all_labels_dfs.append(chunk_labels)
                
                current_idx = end_idx
                chunk_success = True
                break
            
            except LLMInputTooLargeError:
                logger.warning(f"Chunk size {size} muito grande. Tentando reduzir...")
                # Se falhar por tamanho e for o menor tamanho possível, propaga o erro
                if size == 1:
                    raise
                continue
        
        if not chunk_success:
            logger.error("Falha crítica: Não foi possível processar o chunk mesmo com tamanho mínimo.")
            raise RuntimeError("Falha inesperada no processamento de chunks.")

    final_labels = pd.concat(all_labels_dfs, ignore_index=True)

    # Validação Global de Integridade
    expected_ids = set(cluster_profile["ClusterId"])
    got_ids = set(final_labels["ClusterId"])
    
    missing = expected_ids - got_ids
    if missing:
        raise DomainValidationError(f"Erro de integridade global: Clusters perdidos no processamento: {sorted(missing)}")
    
    extra = got_ids - expected_ids
    if extra:
        raise DomainValidationError(f"Erro de integridade global: Clusters extras gerados: {sorted(extra)}")

    return final_labels
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from llm.prompts import CLUSTER_ANALYSIS_TEMPLATE, JSON_REPAIR_TEMPLATE

class DomainValidationError(ValueError):
    """Erro de validação do JSON no domínio de clusters."""


def build_cluster_naming_prompt(cluster_profile: pd.DataFrame, business_context: str) -> str:
    prof_csv = cluster_profile.to_csv(index=False)

    contexto = (business_context or "").strip() or "Não informado."

    return CLUSTER_ANALYSIS_TEMPLATE.format(
        business_context=contexto,
        csv_data=prof_csv
    )


def _ensure_list_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    # se vier string única, transforma em lista
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else []
    return [str(value).strip()] if str(value).strip() else []


def build_cluster_labels(cluster_profile: pd.DataFrame, llm_json: dict) -> pd.DataFrame:
    """
    Converte JSON da LLM para DataFrame e valida campos mínimos.
    """
    if not isinstance(llm_json, dict):
        raise DomainValidationError("JSON inválido: resposta não é um objeto/dict.")

    clusters = llm_json.get("clusters")
    if not isinstance(clusters, list):
        raise DomainValidationError("JSON inválido: chave 'clusters' ausente ou não é lista.")

    rows = []
    for item in clusters:
        if not isinstance(item, dict):
            continue

        if "ClusterId" not in item or "SegmentoNome" not in item or "SegmentoDescricao" not in item:
            raise DomainValidationError(f"Item de cluster inválido (campos ausentes): {item}")

        rows.append(
            {
                "ClusterId": int(item["ClusterId"]),
                "SegmentoNome": str(item["SegmentoNome"]).strip(),
                "SegmentoDescricao": str(item["SegmentoDescricao"]).strip(),
                "Estrategias": _ensure_list_str(item.get("Estrategias")),
            }
        )

    labels = pd.DataFrame(rows)
    if labels.empty:
        raise DomainValidationError("Nenhum cluster válido retornado pela LLM.")

    # valida cobertura mínima
    expected = set(cluster_profile["ClusterId"].tolist())
    got = set(labels["ClusterId"].tolist())
    missing = expected - got
    if missing:
        raise DomainValidationError(f"LLM não retornou rótulos para ClusterId: {sorted(missing)}")

    return labels


def repair_prompt_for_invalid_json(original_prompt: str, last_text: str) -> str:
    """
    Prompt de reparo usado no retry quando a saída não for JSON parseável.
    """
    return JSON_REPAIR_TEMPLATE.format(
        original_prompt=original_prompt,
        last_text=last_text
    )
# app/llm/core.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional


class LLMParseError(RuntimeError):
    """Erro ao converter a resposta da LLM em JSON utilizável."""


def strip_code_fences(text: str) -> str:
    """
    Remove cercas do tipo ```json ... ``` ou ``` ... ``` de forma conservadora.
    """
    t = text.strip()
    if not t.startswith("```"):
        return t

    # Remove a primeira linha ``` ou ```json
    t = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n", "", t, count=1)
    # Remove fence final
    t = re.sub(r"\n```$", "", t, count=1).strip()
    return t


def extract_first_json_object(text: str) -> str:
    """
    Tenta extrair o primeiro objeto JSON {...} de um texto que pode conter
    explicações extras. Faz varredura por chaves balanceadas.
    """
    s = text.strip()
    start = s.find("{")
    if start == -1:
        raise LLMParseError("Resposta não contém '{' para iniciar um JSON.")

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]

    raise LLMParseError("Não foi possível encontrar um objeto JSON completo e balanceado.")


def parse_llm_json(text: str) -> dict[str, Any]:
    """
    Pipeline: strip fences -> tentar json.loads direto -> fallback extrair {...}.
    """
    cleaned = strip_code_fences(text)

    # 1) tentativa direta
    try:
        obj = json.loads(cleaned)
        if not isinstance(obj, dict):
            raise LLMParseError("JSON decodificado não é um objeto/dict no topo.")
        return obj
    except json.JSONDecodeError:
        pass

    # 2) fallback: extrair primeiro objeto JSON do texto
    try:
        candidate = extract_first_json_object(cleaned)
        obj = json.loads(candidate)
        if not isinstance(obj, dict):
            raise LLMParseError("JSON extraído não é um objeto/dict no topo.")
        return obj
    except json.JSONDecodeError as e:
        raise LLMParseError(f"Falha ao decodificar JSON. Erro: {e}\n\nResposta recebida:\n{cleaned}") from e


@dataclass
class RetryConfig:
    max_attempts: int = 2
    # callback para construir "prompt de reparo" caso a saída venha inválida
    repair_prompt_builder: Optional[Callable[[str, str], str]] = None


def run_with_json_retries(
    call_model: Callable[[str], str],
    prompt: str,
    retry: RetryConfig | None = None,
) -> dict[str, Any]:
    """
    call_model(prompt)->text  e retorna dict JSON com retries se parsing falhar.

    repair_prompt_builder(original_prompt, last_response_text) -> new_prompt
    """
    retry = retry or RetryConfig()
    last_text = ""

    for attempt in range(1, retry.max_attempts + 1):
        last_text = call_model(prompt)
        try:
            return parse_llm_json(last_text)
        except LLMParseError:
            if attempt >= retry.max_attempts:
                raise
            if retry.repair_prompt_builder:
                prompt = retry.repair_prompt_builder(prompt, last_text)
            else:
                # fallback simples: reforça JSON estrito
                prompt = (
                    prompt
                    + "\n\nIMPORTANTE: Sua resposta anterior não estava em JSON válido. "
                      "Retorne APENAS JSON válido, sem texto extra, sem markdown."
                )

    raise LLMParseError(f"Falha inesperada após retries. Última resposta:\n{last_text}")
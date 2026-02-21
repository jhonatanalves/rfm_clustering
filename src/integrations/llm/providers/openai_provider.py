from __future__ import annotations

from typing import Any, Optional

from openai import OpenAI
from openai import APIError, AuthenticationError, RateLimitError

from src.integrations.llm.core import run_with_json_retries, RetryConfig, LLMParseError
from src.integrations.llm.domain import repair_prompt_for_invalid_json
from src.integrations.llm.providers.base import ProviderConfig
from src.integrations.llm.prompts import SYSTEM_PROMPT_JSON_ENV



class OpenAIProvider:

    def __init__(self, config: ProviderConfig):
        self._client = OpenAI(api_key=config.api_key)

    def list_models(self) -> list[str]:
        try:
            models = self._client.models.list()
            names = [m.id for m in models.data]
            return sorted(set(names))
        except Exception:
            return []

    def _call_text(self, model: str, prompt: str, temperature: float) -> str:

        resp = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_JSON_ENV},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    def generate_json(self, model: str, prompt: str, temperature: float = 0.2) -> dict[str, Any]:
        try:
            return run_with_json_retries(
                call_model=lambda p: self._call_text(model, p, temperature),
                prompt=prompt,
                retry=RetryConfig(max_attempts=2, repair_prompt_builder=repair_prompt_for_invalid_json),
            )
        except AuthenticationError as e:
            raise RuntimeError("OpenAI: chave inválida ou sem permissão para o projeto/modelo.") from e
        except RateLimitError as e:
            raise RuntimeError("OpenAI: rate limit atingido (muitas requisições).") from e
        except APIError as e:
            raise RuntimeError(f"OpenAI APIError: {e}") from e
        except LLMParseError as e:
            raise RuntimeError(f"Falha ao decodificar JSON da LLM (OpenAI): {e}") from e

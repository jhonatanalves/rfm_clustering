from __future__ import annotations

from typing import Any

from google import genai
from google.genai import errors as genai_errors

from llm.core import run_with_json_retries, RetryConfig, LLMParseError
from llm.domain import repair_prompt_for_invalid_json
from llm.providers.base import ProviderConfig
from llm.prompts import SYSTEM_PROMPT_JSON_ENV


class GeminiProvider:
    def __init__(self, config: ProviderConfig):
        self._client = genai.Client(api_key=config.api_key)

    def list_models(self) -> list[str]:
        models: list[str] = []
        for m in self._client.models.list():
            if hasattr(m, "supported_actions") and "generateContent" in m.supported_actions:
                models.append(m.name.replace("models/", ""))
        return sorted(set(models))

    def _call_text(self, model: str, prompt: str, temperature: float) -> str:
        resp = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": temperature,
                "system_instruction": SYSTEM_PROMPT_JSON_ENV,
            },
        )
        return (getattr(resp, "text", "") or "").strip()

    def generate_json(self, model: str, prompt: str, temperature: float = 0.2) -> dict[str, Any]:
        try:
            return run_with_json_retries(
                call_model=lambda p: self._call_text(model, p, temperature),
                prompt=prompt,
                retry=RetryConfig(max_attempts=2, repair_prompt_builder=repair_prompt_for_invalid_json),
            )
        except genai_errors.ClientError as e:
            raise RuntimeError(f"Gemini ClientError: {e}") from e
        except LLMParseError as e:
            raise RuntimeError(f"Falha ao decodificar JSON da LLM (Gemini): {e}") from e
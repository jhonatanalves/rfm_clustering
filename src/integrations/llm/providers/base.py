# app/llm/providers/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any


class LLMProvider(Protocol):
    def list_models(self) -> list[str]:
        ...

    def generate_json(self, model: str, prompt: str) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class ProviderConfig:
    api_key: str

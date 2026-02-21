from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ProviderConfig:
    api_key: str

class BaseProvider(ABC):
    """Classe base abstrata para provedores de LLM."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    def list_models(self) -> list[str]:
        """Retorna a lista de modelos disponíveis para este provedor."""
        pass

    @abstractmethod
    def generate_text(self, model: str, prompt: str, temperature: float, max_tokens: int, timeout: float) -> str:
        """Gera uma resposta de texto simples."""
        pass

    @abstractmethod
    def generate_json(self, model: str, prompt: str, temperature: float, max_tokens: int, timeout: float) -> dict:
        """Gera uma resposta estruturada em JSON, com lógica de retry/repair."""
        pass
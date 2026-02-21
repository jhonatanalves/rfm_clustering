import openai
from src.integrations.llm.providers.base import BaseProvider, ProviderConfig
from src.integrations.llm.core import run_with_json_retries
from src.integrations.llm.prompts import SYSTEM_PROMPT_JSON_ENV

class OpenAIProvider(BaseProvider):
    """Implementação do provedor OpenAI (ChatGPT)."""
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = openai.OpenAI(api_key=config.api_key)

    def list_models(self) -> list[str]:
        try:
            # A API retorna apenas modelos acessíveis pela API Key.
            # Filtra por "gpt" para trazer apenas modelos de chat compatíveis.
            models = self.client.models.list()
            return sorted([m.id for m in models.data if m.id.startswith("gpt")])
        except Exception:
            return ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"]

    def generate_text(self, model: str, prompt: str, temperature: float, max_tokens: int, timeout: float) -> str:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_JSON_ENV},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
            return response.choices[0].message.content
        except openai.APITimeoutError:
            raise TimeoutError(f"OpenAI: Tempo limite excedido ({timeout}s).")

    def generate_json(self, model: str, prompt: str, temperature: float, max_tokens: int, timeout: float) -> dict:
        def call_model(p: str) -> str:
            return self.generate_text(model, p, temperature, max_tokens, timeout)

        return run_with_json_retries(call_model, prompt)

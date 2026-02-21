import google.generativeai as genai
from src.integrations.llm.providers.base import BaseProvider, ProviderConfig
from src.integrations.llm.core import run_with_json_retries

class GeminiProvider(BaseProvider):
    """Implementação do provedor Google Gemini."""
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        genai.configure(api_key=config.api_key)

    def list_models(self) -> list[str]:
        try:
            # genai.list_models() usa a chave configurada para listar o que está disponível.
            models = genai.list_models()
            return sorted([
                m.name.replace("models/", "") 
                for m in models 
                if "generateContent" in m.supported_generation_methods
            ])
        except Exception:
            return ["gemini-pro", "gemini-1.5-flash", "gemini-1.5-pro"]

    def generate_text(self, model: str, prompt: str, temperature: float, max_tokens: int, timeout: float) -> str:
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        
        model_instance = genai.GenerativeModel(model)
        
        try:
            response = model_instance.generate_content(
                prompt,
                generation_config=generation_config,
                request_options={"timeout": timeout}
            )
            return response.text
        except Exception as e:
            if "deadline" in str(e).lower() or "timeout" in str(e).lower():
                raise TimeoutError(f"Gemini: Tempo limite excedido ({timeout}s).")
            raise e

    def generate_json(self, model: str, prompt: str, temperature: float, max_tokens: int, timeout: float) -> dict:
        def call_model(p: str) -> str:
            return self.generate_text(model, p, temperature, max_tokens, timeout)

        return run_with_json_retries(call_model, prompt)
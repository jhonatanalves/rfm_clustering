import google.generativeai as genai
from src.integrations.llm.providers.base import BaseProvider, ProviderConfig
from src.integrations.llm.core import run_with_json_retries, LLMTokenLimitExceededError
from src.observability.logger import get_logger

class GeminiProvider(BaseProvider):
    """Implementação do provedor Google Gemini."""
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        genai.configure(api_key=config.api_key)
        self.logger = get_logger("provider.gemini")

    def _log_snippet(self, text: str) -> str:
        if not text: return "[EMPTY]"
        if len(text) <= 1600:
            return text
        return f"{text[:800]} ... [skipped] ... {text[-800:]}"

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
        # Prefixo Forte para forçar JSON
        strong_prefix = (
            "RETORNE APENAS UM OBJETO JSON VÁLIDO. "
            "NÃO inclua texto extra, explicações, markdown, blocos de código ou comentários. "
            "Comece com '{' e termine com '}'. A raiz deve ser um objeto com a chave 'clusters'."
        )
        full_prompt = f"{strong_prefix}\n\n{prompt}"
        
        self.logger.info(f"Gerando texto (Gemini). Model: {model}, MaxTokens: {max_tokens}, Timeout: {timeout}s. Len(FullPrompt): {len(full_prompt)}")

        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        
        model_instance = genai.GenerativeModel(model)
        
        try:
            response = model_instance.generate_content(
                full_prompt,
                generation_config=generation_config,
                request_options={"timeout": timeout}
            )
            
            # Verifica se a resposta foi truncada por limite de tokens
            candidate = response.candidates[0] if response.candidates else None
            reason = candidate.finish_reason if candidate else None
            
            # Verifica MAX_TOKENS (pode vir como Enum, objeto com .name/.value ou int 2)
            is_max_tokens = (getattr(reason, "name", "") == "MAX_TOKENS" or 
                             getattr(reason, "value", -1) == 2 or 
                             reason == 2)

            if reason and is_max_tokens:
                self.logger.warning(f"Resposta truncada (MAX_TOKENS). Tokens usados: {response.usage_metadata.total_token_count if response.usage_metadata else 'N/A'}")
                raise LLMTokenLimitExceededError("A resposta foi cortada porque atingiu o limite máximo de tokens configurado.")

            self.logger.debug(f"Gemini Response recebida. Len: {len(response.text)}. Snippet:\n{self._log_snippet(response.text)}")
            return response.text
        except Exception as e:
            self.logger.error(f"Erro na chamada Gemini: {str(e)}", exc_info=True)
            if "deadline" in str(e).lower() or "timeout" in str(e).lower():
                self.logger.error(f"Timeout explícito detectado ({timeout}s).")
                raise TimeoutError(f"Gemini: Tempo limite excedido ({timeout}s).")
            raise e

    def generate_json(self, model: str, prompt: str, temperature: float, max_tokens: int, timeout: float) -> dict:
        def call_model(p: str) -> str:
            return self.generate_text(model, p, temperature, max_tokens, timeout)

        return run_with_json_retries(call_model, prompt)
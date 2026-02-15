import streamlit as st
from llm.providers.base import ProviderConfig
from llm.providers.gemini_provider import GeminiProvider
from llm.providers.openai_provider import OpenAIProvider


def init_session_state():
    """Inicializa variáveis de estado do Streamlit."""
    defaults = {
        "rfm_out": None,
        "cluster_profile": None,
        "cluster_labels": None,
        "llm_models": [],
        "llm_provider": "Gemini",
        "llm_model": None,
        "business_context": "",
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get_provider(provider_name: str, api_key: str):
    """Factory para instanciar provider."""
    cfg = ProviderConfig(api_key=api_key)
    if provider_name == "Gemini":
        return GeminiProvider(cfg)
    elif provider_name == "ChatGPT":
        return OpenAIProvider(cfg)
    else:
        raise ValueError(f"Provider inválido: {provider_name}")


def render_sidebar():
    """Renderiza a barra lateral e retorna as configurações."""
    with st.sidebar:
        st.header("📂 Dados & Parâmetros")
        up = st.file_uploader(
            label="",
            type=["csv"],
            label_visibility="collapsed",
            help="Faça upload do seu arquivo CSV contendo dados de transações."
        )

        st.caption("Estrutura do Arquivo")
        data_structure = st.radio(
            "O que cada linha representa?",
            ["Itens de um Pedido (ex: 1 pedido pode ter várias linhas)", "Um Pedido/Venda (1 linha = 1 transação consolidada)"],
            index=0,
            help="Defina se o CSV detalha itens (precisa agrupar) ou se já é consolidado."
        )

        st.caption("Configurações de Processamento")
        auto_clean = st.toggle("Aplicar limpeza automática", value=True)
        if auto_clean:
            st.caption("Remove duplicados e trata nulos automaticamente.")

        st.divider()
        st.subheader("⚙️ Clusterização")
        k_mode = st.radio("Seleção de k", ["Automático (cotovelo)", "Manual"], index=0)

        k_min, k_max = 2, 12
        if k_mode == "Manual":
            n_clusters = st.slider("k", k_min, k_max, min(4, k_max))
        else:
            st.info("O sistema calculará o melhor número de clusters.")
            n_clusters = None

        with st.expander("🔧 Avançado"):
            random_state = st.number_input("Seed", value=42, step=1)
            st.caption("Parâmetros K-Means")
            n_init = st.number_input("Tentativas (n_init)", value=10, min_value=1, step=1)
            max_iter = st.number_input("Máx. Iterações", value=300, min_value=100, step=50)
            st.caption("Parâmetros LLM")
            temperature = st.slider("Temperatura", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

        st.divider()
        st.header("🧠 Configuração LLM e Contexto")

        provider_name = st.radio("Provider", ["Gemini", "ChatGPT"], index=0)

        if provider_name != st.session_state.get("llm_provider"):
            st.session_state["llm_models"] = []
            st.session_state["llm_model"] = None
        st.session_state["llm_provider"] = provider_name

        if provider_name == "Gemini":
            api_key = st.text_input("Gemini API Key", type="password", key="api_key_gemini")
        else:
            api_key = st.text_input("OpenAI API Key", type="password", key="api_key_openai")

        load_disabled = not api_key
        if st.button("Carregar modelos", disabled=load_disabled):
            try:
                provider = get_provider(provider_name, api_key)
                models = provider.list_models()
                st.session_state["llm_models"] = models
                st.session_state["llm_model"] = None
                if models:
                    st.success("Modelos carregados.")
                else:
                    st.warning("Nenhum modelo retornado.")
            except Exception as e:
                st.session_state["llm_models"] = []
                st.session_state["llm_model"] = None
                st.error(str(e))

        if st.session_state["llm_models"]:
            idx = 0
            current = st.session_state.get("llm_model")
            if current and current in st.session_state["llm_models"]:
                idx = st.session_state["llm_models"].index(current)
            model = st.selectbox("Modelo", st.session_state["llm_models"], index=idx)
            st.session_state["llm_model"] = model

        st.caption("Contexto do Negócio (para a IA)")
        business_context = st.text_area(
            label="Contexto",
            placeholder="Ex: E-commerce de moda feminina...",
            height=100,
            label_visibility="collapsed"
        )
        st.session_state["business_context"] = business_context.strip()

    return {
        "up": up,
        "data_structure": data_structure,
        "auto_clean": auto_clean,
        "n_clusters": n_clusters,
        "random_state": random_state,
        "n_init": n_init,
        "max_iter": max_iter,
        "temperature": temperature,
        "api_key": api_key,
        "provider_name": provider_name,
        "model": model if st.session_state["llm_models"] else None
    }
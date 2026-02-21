import streamlit as st
from src.integrations.llm.providers.base import ProviderConfig
from src.integrations.llm.providers.gemini_provider import GeminiProvider
from src.integrations.llm.providers.openai_provider import OpenAIProvider


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
        st.header("📂 Carregamento dos Dados")
        up = st.file_uploader(
            label="",
            type=["csv"],
            label_visibility="collapsed"
        )

        data_structure = st.radio(
            "O que cada linha representa?",
            ["Dados Agregados", "Dados Transacionais"],
            index=0,
            help="Dados Transacionais: 1 compra pode ter 1 item por linha.\nDados Agregados: 1 linha = 1 transação consolidada."
        )

        st.divider()
        st.subheader("🧹 Limpeza dos Dados")
        
        # Configurações padrão (sempre ativas)
        auto_clean = True
        clean_duplicates = True

        # 1. Outliers
        outlier_opt = st.radio(
            "Tratamento de Outliers",
            ["Manter", "Remover", "Winsorization", "Substituir pela Mediana"],
            index=0,
            help="Winsorization limita os extremos aos percentis 5% e 95%. Remover exclui a linha inteira."
        )
        
        if outlier_opt == "Winsorization":
            clean_outliers = "winz"
        elif outlier_opt == "Substituir pela Mediana":
            clean_outliers = "median"
        elif outlier_opt == "Remover":
            clean_outliers = "delete"
        else:
            clean_outliers = False
        
        # 2. Dados Ausentes 
        missing_opt = st.radio(
            "Tratamento de Dados Ausentes",
            ["Manter", "Remover", "Aplicar KNN", "Aplicar Regressão", "Aplicar Média/Moda", "Aplicar Mediana/Moda", "Aplicar mais Frequente"],
            index=0,
            help="Define como preencher vazios. Colunas numéricas usam a 1ª opção (Média/Mediana). Colunas de texto usam sempre a Moda (valor mais comum)."
        )

        mapping_missing = {
            "Aplicar Mediana/Moda": "median",
            "Aplicar Média/Moda": "mean",
            "Aplicar mais Frequente": "most_frequent",
            "Aplicar KNN": "knn",
            "Aplicar Regressão": "linreg",
            "Remover": "delete",
            "Manter": False
        }
        clean_imputation = mapping_missing[missing_opt]

        st.divider()
        st.subheader("⚙️ Clusterização")
        k_mode = st.radio(
            "Seleção de k",
            ["Cotovelo", "Manual"],
            index=0,
            help="No modo Automático, o sistema usa o método do cotovelo para encontrar o número ideal de clusters."
        )

        k_min, k_max = 2, 12
        if k_mode == "Manual":
            n_clusters = st.slider("k", k_min, k_max, min(4, k_max))
        else:
            n_clusters = None

        st.caption("Parâmetros do K-Means")
        random_state = st.number_input("Semente", value=42, step=1, help="Número usado para garantir que o resultado seja o mesmo toda vez que rodar.")
        n_init = st.number_input("Tentativas de Otimização", value=10, min_value=1, step=1, help="Quantas vezes o algoritmo reinicia para encontrar o melhor agrupamento possível.")
        max_iter = st.number_input("Limite de Passos", value=300, min_value=100, step=50, help="Número máximo de ajustes que o algoritmo faz para refinar os grupos.")

        st.divider()
        st.header("🧠 Configuração da LLM")

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

        temperature = st.slider("Temperatura", min_value=0.0, max_value=1.0, value=0.2, step=0.1, help="Controla a criatividade da IA. Valores mais altos geram textos mais variados.")

        business_context = st.text_area(
            label="Contexto do Negócio",
            placeholder="Ex: E-commerce de moda feminina focado em público jovem. O objetivo é identificar clientes VIPs para fidelização e recuperar inativos.",
            height=170,
            help="Insira detalhes sobre o seu negócio e objetivos. Essas informações serão enviadas ao prompt da IA para gerar nomes de clusters e estratégias mais personalizadas."
        )
        st.session_state["business_context"] = business_context.strip()

    return {
        "up": up,
        "data_structure": data_structure,
        "auto_clean": auto_clean, 
        "clean_duplicates": clean_duplicates,
        "clean_imputation": clean_imputation,
        "clean_outliers": clean_outliers,
        "n_clusters": n_clusters,
        "random_state": random_state,
        "n_init": n_init,
        "max_iter": max_iter,
        "temperature": temperature,
        "api_key": api_key,
        "provider_name": provider_name,
        "model": model if st.session_state["llm_models"] else None
    }
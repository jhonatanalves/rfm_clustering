import pandas as pd
import streamlit as st
from uuid import uuid4

from src.ui.components import sidebar, mapping, results

from src.engine.calculator import build_rfm_table
from src.engine.clustering import (
    calcular_wcss,
    get_numero_otimo_clusters,
    cluster_rfm_joint,
)
from src.integrations.llm.domain import build_cluster_naming_prompt, orchestrate_cluster_labeling, LLMInputTooLargeError
from src.integrations.llm.core import LLMTokenLimitExceededError, LLMParseError
from src.engine.cleaning.pipeline import apply_autoclean
from src.observability.logger import get_logger, set_request_id, reset_request_id


@st.cache_data(show_spinner="Carregando dados...")
def load_data(uploaded_file):
    """
    Carrega o CSV e faz cache para evitar recarregamento a cada interação.

    Args:
        uploaded_file: Objeto de arquivo retornado pelo st.file_uploader.

    Returns:
        pd.DataFrame: DataFrame carregado do CSV.
    """
    return pd.read_csv(uploaded_file)


def validate_mapping_inputs(mapping_data: dict, data_structure: str):
    """
    Valida se todos os campos obrigatórios do mapeamento foram preenchidos.

    Args:
        mapping_data (dict): Dicionário com o mapeamento das colunas.
        data_structure (str): Tipo de estrutura de dados ('Dados Transacionais' ou 'Dados Agregados').
    
    Returns:
        None: Interrompe a execução (st.stop) se houver erro.
    """
    missing_fields = []
    if not mapping_data["customer_col"]: missing_fields.append("Id do cliente")
    if not mapping_data["date_col"]: missing_fields.append("Data da compra")
    if not mapping_data["monetary_col"]: missing_fields.append("Valor da compra")
    if not mapping_data["approved_col"]: missing_fields.append("Status do Pedido")
    if data_structure == "Dados Transacionais" and not mapping_data["order_col"]: missing_fields.append("ID do Pedido")
    
    if missing_fields:
        st.error(f"Por favor, preencha os seguintes campos obrigatórios: {', '.join(missing_fields)}")
        st.stop()
        
    if not mapping_data["approved_values"]:
        st.error("Por favor, selecione pelo menos um valor em 'Valores para considerar'.")
        st.stop()


def preprocess_dataframe(df: pd.DataFrame, mapping_data: dict, config: dict) -> pd.DataFrame:
    """
    Seleciona colunas, remove nulos e aplica limpeza automática (outliers/imputação).

    Args:
        df (pd.DataFrame): DataFrame original carregado.
        mapping_data (dict): Configuração de mapeamento de colunas.
        config (dict): Configurações de limpeza da sidebar.

    Returns:
        pd.DataFrame: DataFrame limpo e pronto para o cálculo de RFM.
    """
    cols_to_use = [
        mapping_data["customer_col"],
        mapping_data["date_col"],
        mapping_data["monetary_col"]
    ]
    if mapping_data["order_col"]:
        cols_to_use.append(mapping_data["order_col"])
    if mapping_data["approved_col"]:
        cols_to_use.append(mapping_data["approved_col"])

    df_clean = df[cols_to_use].copy()

    df_clean = df_clean.dropna(subset=[mapping_data["customer_col"], mapping_data["date_col"]])
    
    df_clean[mapping_data["customer_col"]] = df_clean[mapping_data["customer_col"]].astype(str)

    if config["auto_clean"]:
        if config["clean_duplicates"]:
            rows_before = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            if len(df_clean) < rows_before:
                st.toast(f"Limpeza: {rows_before - len(df_clean)} registros duplicados removidos.", icon="🗑️")

        monetary_col = mapping_data["monetary_col"]
        
        with st.spinner("Aplicando limpeza automática (apenas Valor Monetário)..."):
            df_money = df_clean[[monetary_col]].copy()
            
            df_money = apply_autoclean(
                df_money,
                tratar_duplicados=False, 
                imputacao=config["clean_imputation"],
                tratar_outliers=config["clean_outliers"]
            )
            
            df_clean = df_clean.loc[df_money.index]
            df_clean[monetary_col] = df_money[monetary_col]
    
    return df_clean


def execute_clustering_pipeline(df_clean: pd.DataFrame, mapping_data: dict, config: dict):
    """
    Executa o cálculo de RFM, determina o número ideal de clusters (K) e realiza a clusterização.

    Args:
        df_clean (pd.DataFrame): DataFrame pré-processado.
        mapping_data (dict): Mapeamento de colunas.
        config (dict): Configurações gerais (K-means, random_state, etc).
    
    Returns:
        None: Atualiza o st.session_state com os resultados ('rfm_out', 'cluster_profile').
    """
    try:
        rfm_table = build_rfm_table(
            data=df_clean,
            customer_col=mapping_data["customer_col"],
            date_col=mapping_data["date_col"],
            monetary_col=mapping_data["monetary_col"],
            order_col=mapping_data["order_col"],
            approved_col=mapping_data["approved_col"],
            approved_values=mapping_data["approved_values"],
        )

        if len(rfm_table) < 3:
            st.error(f"Filtros resultaram em apenas {len(rfm_table)} clientes. É necessário pelo menos 3 para clusterizar. Verifique os status selecionados.")
            st.stop()

        chosen_k = config["n_clusters"]
        if chosen_k is not None and chosen_k >= len(rfm_table):
            st.error(f"Número de clusters ({chosen_k}) não pode ser maior ou igual ao número de clientes ({len(rfm_table)}).")
            st.stop()

        if chosen_k is None:
            from src.engine.clustering import build_rfm_features
            Xs, _ = build_rfm_features(rfm_table)
            
            safe_k_max = min(10, len(rfm_table) - 1)
            wcss = calcular_wcss(Xs, k_min=2, k_max=safe_k_max, random_state=int(config["random_state"]))
            chosen_k = get_numero_otimo_clusters(wcss, k_min=2, k_max=safe_k_max)

        rfm_out, cluster_profile = cluster_rfm_joint(
            rfm=rfm_table,
            n_clusters=int(chosen_k),
            random_state=int(config["random_state"]),
            n_init=int(config["n_init"]),
            max_iter=int(config["max_iter"]),
        )

        st.session_state["rfm_out"] = rfm_out
        st.session_state["cluster_profile"] = cluster_profile
        st.session_state["cluster_labels"] = None

        st.success(f"Clusterização concluída! k = {int(chosen_k)}")

    except Exception as e:
        st.error(f"Erro ao rodar pipeline: {e}")
        st.stop()


def handle_llm_generation(cluster_profile: pd.DataFrame, config: dict):
    """
    Gerencia a interação com a LLM para nomeação e explicação dos clusters.

    Args:
        cluster_profile (pd.DataFrame): DataFrame com o perfil agregado dos clusters.
        config (dict): Configurações da LLM (provider, api_key, model).
    
    Returns:
        None: Atualiza o st.session_state['cluster_labels'] com os resultados.
    """
    st.header(f"🤖 Nomear e explicar clusters ({config['provider_name']})")

    if not config["api_key"] or not config["model"]:
        st.info("Selecione o provider, insira a API Key e defina um modelo na barra lateral.")
        return

    if st.button("Gerar nomes e estratégias (LLM)", type="primary"):
        business_context = st.session_state.get("business_context", "")
        
        # 1. Generate Correlation ID
        request_id = uuid4().hex[:8]
        token = set_request_id(request_id)
        logger = get_logger("app")

        # Mostra apenas o prompt do primeiro chunk como exemplo para o usuário não ficar confuso
        preview_prompt = build_cluster_naming_prompt(cluster_profile.head(5), business_context)
        with st.expander("🔎 Exemplo de Prompt (primeiros 5 clusters)", expanded=False):
            st.code(preview_prompt)

        try:
            provider = sidebar.get_provider(config["provider_name"], config["api_key"])
            with st.spinner(f"Gerando análises com {config['provider_name']} (processando em lotes)..."):
                logger.info(f"Iniciando orquestração LLM. Provider: {config['provider_name']}")
                labels_df = orchestrate_cluster_labeling(
                    provider, cluster_profile, business_context, config["model"], config["temperature"]
                )

            st.session_state["cluster_labels"] = labels_df
            st.success("Rótulos gerados com sucesso!")
            logger.info("Geração de rótulos concluída com sucesso.")

        except LLMInputTooLargeError:
            logger.warning("Erro: Input muito grande para a LLM.")
            st.error("Dados muito grandes para análise automática. Tente reduzir o número de clusters (k) ou usar uma base menor.")
        except LLMTokenLimitExceededError:
            logger.warning("Erro: Limite de tokens excedido na resposta.")
            st.error("⚠️ A resposta foi cortada pela IA por falta de espaço (tokens). \n\n**Sugestão:** Tente reduzir o número de clusters (k) na barra lateral ou simplifique o contexto do negócio.")
        except LLMParseError:
            logger.warning("Erro de Parse JSON (provável resposta incompleta).")
            st.error("⚠️ A resposta da IA veio incompleta (JSON inválido).\n\nIsso geralmente ocorre quando o limite de tokens é atingido antes do fim da resposta. Tente reduzir o número de clusters.")
        except Exception as e:
            logger.error(f"Falha na geração de rótulos: {e}", exc_info=True)
            st.error(f"Falha na geração de rótulos. ID de diagnóstico: {request_id}")
        finally:
            reset_request_id(token)


def main():
    st.set_page_config(page_title="RFM Segmentation", layout="wide")
    
    # Inicialização
    sidebar.init_session_state()
    config = sidebar.render_sidebar()

    st.title("📊 Clusterização de clientes via RFM")

    if not config["up"]:
        st.info("Faça upload de um CSV para começar.")
        st.stop()

    df = load_data(config["up"])
    with st.container(border=True):
        results.render_data_preview(df)

    mapping_data = mapping.render_column_mapping(df, config["data_structure"])

    # 1. Pipeline de Clusterização
    if mapping_data["run_pipeline"]:
        validate_mapping_inputs(mapping_data, config["data_structure"])
        df_clean = preprocess_dataframe(df, mapping_data, config)
        execute_clustering_pipeline(df_clean, mapping_data, config)

    # 2. Exibição de Resultados
    rfm_out = st.session_state.get("rfm_out")
    cluster_profile = st.session_state.get("cluster_profile")

    if rfm_out is None or cluster_profile is None:
        st.info("Configure o mapeamento e clique em **Rodar RFM + Clusterização**.")
        st.stop()

    results.render_results(rfm_out, cluster_profile)

    with st.container(border=True):
        st.subheader("📈 Visualização dos Clusters")
        results.render_cluster_charts(rfm_out)

    st.divider()

    # 3. Pipeline de LLM
    handle_llm_generation(cluster_profile, config)

    labels_df = st.session_state.get("cluster_labels")
    if labels_df is None:
        st.info("Gere os nomes via LLM para enriquecer os gráficos e as descrições.")
    else:
        results.render_llm_results(rfm_out, cluster_profile, labels_df)


if __name__ == "__main__":
    main()
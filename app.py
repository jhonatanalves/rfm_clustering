import pandas as pd
import streamlit as st

from components import sidebar, mapping, results

from rfm import (
    build_rfm_table,
    calcular_wcss,
    get_numero_otimo_clusters,
    cluster_rfm_joint,
)
from llm.domain import build_cluster_naming_prompt, build_cluster_labels

st.set_page_config(page_title="RFM Segmentation", layout="wide")

# Inicialização
sidebar.init_session_state()
config = sidebar.render_sidebar()

# Main
st.title("📊 Clusterização via RFM")

if not config["up"]:
    st.info("Faça upload de um CSV para começar.")
    st.stop()

df = pd.read_csv(config["up"])
results.render_data_preview(df)

mapping_data = mapping.render_column_mapping(df, config["data_structure"])

if mapping_data["run_pipeline"]:
    # Validação manual dos campos obrigatórios antes de prosseguir
    missing_fields = []
    if not mapping_data["customer_col"]: missing_fields.append("Id do cliente")
    if not mapping_data["date_col"]: missing_fields.append("Data da compra")
    if not mapping_data["monetary_col"]: missing_fields.append("Valor da compra")
    if not mapping_data["approved_col"]: missing_fields.append("Status do Pedido")
    if config["data_structure"].startswith("Itens") and not mapping_data["order_col"]: missing_fields.append("ID do Pedido")
    
    if missing_fields:
        st.error(f"Por favor, preencha os seguintes campos obrigatórios: {', '.join(missing_fields)}")
        st.stop()
        
    if not mapping_data["approved_values"]:
        st.error("Por favor, selecione pelo menos um valor em 'Valores para considerar'.")
        st.stop()

    try:
        rfm_table = build_rfm_table(
            data=df,
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
            from rfm import build_rfm_features  # import local para evitar circularidade
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


rfm_out = st.session_state.get("rfm_out")
cluster_profile = st.session_state.get("cluster_profile")

if rfm_out is None or cluster_profile is None:
    st.info("Configure o mapeamento e clique em **Rodar RFM + Clusterização**.")
    st.stop()

results.render_results(rfm_out, cluster_profile)

st.divider()

st.header(f"🤖 Nomear e explicar clusters ({config['provider_name']})")

if not config["api_key"] or not config["model"]:
    st.info("Selecione o provider, insira a API Key e defina um modelo na barra lateral.")
else:
    if st.button("Gerar nomes e estratégias (LLM)", type="primary"):
        business_context = st.session_state.get("business_context", "")
        prompt = build_cluster_naming_prompt(cluster_profile, business_context)

        with st.expander("🔎 Prompt enviado (agregados)", expanded=False):
            st.code(prompt)

        try:
            provider = sidebar.get_provider(config["provider_name"], config["api_key"])
            with st.spinner(f"Chamando {config['provider_name']}..."):
                llm_json = provider.generate_json(
                    model=config["model"], 
                    prompt=prompt, 
                    temperature=config["temperature"]
                )
                labels_df = build_cluster_labels(cluster_profile, llm_json)

            st.session_state["cluster_labels"] = labels_df
            st.success("Rótulos gerados com sucesso!")

        except Exception as e:
            st.error(f"Falha na geração de rótulos: {e}")


labels_df = st.session_state.get("cluster_labels")

if labels_df is None:
    st.info("Gere os nomes via LLM para enriquecer os gráficos e as descrições.")
else:
    results.render_llm_results(rfm_out, cluster_profile, labels_df)
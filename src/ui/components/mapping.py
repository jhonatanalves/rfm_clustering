import pandas as pd
import streamlit as st


def _render_mapping_form(df: pd.DataFrame, data_structure: str) -> tuple[dict, bool]:
    """
    Renderiza o formulário de seleção de colunas.

    Args:
        df (pd.DataFrame): DataFrame carregado.
        data_structure (str): Tipo de estrutura ('Dados Transacionais' ou 'Dados Agregados').

    Returns:
        tuple[dict, bool]: Dicionário com as colunas selecionadas e booleano indicando submissão.
    """
    cols = df.columns.tolist()
    
    with st.form("mapping_form"):
        st.caption("Selecione as colunas correspondentes no seu arquivo:")
        c1, c2 = st.columns(2)
        
        with c1:
            customer_col = st.selectbox("Id do cliente", cols, index=None, placeholder="Selecione ID do cliente")
            monetary_col = st.selectbox("Valor da compra", cols, index=None, placeholder="Selecione Valor")
            approved_col = st.selectbox("Status do Pedido", cols, index=None, placeholder="Selecione Status")
        
        with c2:
            date_col = st.selectbox("Data da compra", cols, index=None, placeholder="Selecione Data")
            order_col = None
            if data_structure.startswith("Itens"):
                order_col = st.selectbox("ID do Pedido", cols, index=None, placeholder="Selecione ID do Pedido")

        st.markdown("---")
        submitted = st.form_submit_button("Confirmar Mapeamento")

    return {
        "customer_col": customer_col,
        "date_col": date_col,
        "monetary_col": monetary_col,
        "order_col": order_col,
        "approved_col": approved_col,
    }, submitted


def _render_filter_form(df: pd.DataFrame, approved_col: str) -> tuple[list[str], bool]:
    """
    Renderiza o formulário de seleção de status (filtros).

    Args:
        df (pd.DataFrame): DataFrame carregado.
        approved_col (str): Nome da coluna que contém os status.

    Returns:
        tuple[list[str], bool]: Lista de valores selecionados e booleano indicando execução.
    """
    with st.form("filter_form"):
        st.write("Configuração de Filtros")
        
        counts = df[approved_col].value_counts()
        options = counts.index.tolist()

        def fmt_status(option):
            return f"{option} ({counts[option]})"

        st.write("Valores para considerar (Status):")
        approved_values = st.pills(
            "Status",
            options,
            selection_mode="multi",
            format_func=fmt_status,
            label_visibility="collapsed",
            help="Selecione os status que representam vendas confirmadas (ex: delivered)."
        ) or []

        st.markdown("---")
        run_pipeline = st.form_submit_button("Executar Clusterização", type="primary")
    
    return approved_values, run_pipeline


def render_column_mapping(df: pd.DataFrame, data_structure: str):
    """
    Função principal que orquestra a renderização dos formulários de mapeamento e filtros.

    Args:
        df (pd.DataFrame): DataFrame carregado.
        data_structure (str): Tipo de estrutura de dados.

    Returns:
        dict: Configuração completa contendo colunas mapeadas, valores de filtro e flag de execução.
    """
    with st.container(border=True):
        st.subheader("Mapeamento de colunas")

        mapping_inputs, submitted = _render_mapping_form(df, data_structure)

        approved_values = []
        run_pipeline = False

        # Mostr a seleção de valores e o botão de execução se as colunas estiverem definidas
        if (mapping_inputs["customer_col"] and 
            mapping_inputs["date_col"] and 
            mapping_inputs["monetary_col"] and 
            mapping_inputs["approved_col"]):
            
            st.divider()
            approved_values, run_pipeline = _render_filter_form(df, mapping_inputs["approved_col"])
        
        elif submitted:
            st.warning("Por favor, preencha todas as colunas do mapeamento.")

    return {
        **mapping_inputs,
        "approved_values": approved_values,
        "run_pipeline": run_pipeline
    }
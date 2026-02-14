import pandas as pd
import streamlit as st


def render_column_mapping(df: pd.DataFrame, data_structure: str):
    cols = df.columns.tolist()

    # Container com borda para simular um fieldset
    with st.container(border=True):
        st.subheader("Mapeamento de colunas")

        customer_col = st.selectbox("Id do cliente", cols, index=None, placeholder="Selecione ID do cliente")
        date_col = st.selectbox("Data da compra", cols, index=None, placeholder="Selecione Data")
        monetary_col = st.selectbox("Valor da compra", cols, index=None, placeholder="Selecione Valor")

        order_col = None
        if data_structure.startswith("Itens"):
            order_col = st.selectbox("ID do Pedido", cols, index=None, placeholder="Selecione ID do Pedido")

        approved_col = st.selectbox("Status do Pedido", cols, index=None, placeholder="Selecione Status")

        approved_values = []
        if approved_col:
            # Calcula contagem para exibir no botão
            counts = df[approved_col].value_counts()
            options = counts.index.tolist()

            def fmt_status(option):
                return f"{option} ({counts[option]})"

            st.write("Valores para considerar:")
            approved_values = st.pills(
                "Status",
                options,
                selection_mode="multi",
                format_func=fmt_status,
                label_visibility="collapsed",
                help="Selecione os status que representam vendas confirmadas (ex: delivered)."
            ) or []
        else:
            st.info("👆 Selecione a coluna de Status para ver as opções.")

        st.markdown("---")
        run_pipeline = st.button("Executar Clusterização", type="primary")

    return {
        "customer_col": customer_col,
        "date_col": date_col,
        "monetary_col": monetary_col,
        "order_col": order_col,
        "approved_col": approved_col,
        "approved_values": approved_values,
        "run_pipeline": run_pipeline
    }
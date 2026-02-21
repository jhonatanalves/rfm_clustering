import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def _segment_col(rfm_df: pd.DataFrame) -> str:
    """
    Determina a coluna de segmentação a ser usada (Nome ou ID).

    Args:
        rfm_df (pd.DataFrame): DataFrame contendo os dados de RFM.

    Returns:
        str: Nome da coluna ('SegmentoNome' ou 'ClusterId').
    """
    return "SegmentoNome" if "SegmentoNome" in rfm_df.columns else "ClusterId"


def scatter_by_group(rfm_df: pd.DataFrame, x: str, y: str) -> None:
    """
    Gera e renderiza um gráfico de dispersão (scatter plot) agrupado por segmento.

    Args:
        rfm_df (pd.DataFrame): DataFrame contendo os dados de RFM e a coluna de cluster/segmento.
        x (str): Nome da coluna para o eixo X.
        y (str): Nome da coluna para o eixo Y.

    Returns:
        None: Renderiza o gráfico diretamente no Streamlit.
    """
    group_col = _segment_col(rfm_df)

    fig, ax = plt.subplots()
    for name, group in rfm_df.groupby(rfm_df[group_col].astype(str)):
        ax.scatter(group[x], group[y], label=name, alpha=0.7)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def render_scatter_grid(rfm_df: pd.DataFrame) -> None:
    """
    Renderiza um grid com três gráficos de dispersão comparando as métricas RFM.

    Exibe:
    1. Recência x Frequência
    2. Frequência x Receita
    3. Recência x Receita

    Args:
        rfm_df (pd.DataFrame): DataFrame contendo as métricas RFM e identificadores de cluster.

    Returns:
        None: Renderiza os gráficos diretamente no Streamlit.
    """
    st.subheader("Visualizações (scatter)")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.caption("Recência x Frequência")
        scatter_by_group(rfm_df, "Recencia", "Frequencia")
    with p2:
        st.caption("Frequência x Receita")
        scatter_by_group(rfm_df, "Frequencia", "Receita")
    with p3:
        st.caption("Recência x Receita")
        scatter_by_group(rfm_df, "Recencia", "Receita")
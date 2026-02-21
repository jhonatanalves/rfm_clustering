import pandas as pd
import streamlit as st
try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
from src.ui.visualization.charts import render_scatter_grid


def render_data_preview(df: pd.DataFrame):
    st.subheader("Prévia do dataset")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)


def render_results(rfm_out, cluster_profile):
    st.subheader("Perfil dos clusters agregado")
    st.dataframe(cluster_profile, use_container_width=True, hide_index=True)

    st.subheader("Resultado por cliente")
    st.dataframe(rfm_out, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Baixar CSV (clientes)",
            rfm_out.to_csv(index=False).encode("utf-8"),
            "rfm_clientes.csv",
            "text/csv",
        )
    with c2:
        st.download_button(
            "⬇️ Baixar CSV (perfil clusters)",
            cluster_profile.to_csv(index=False).encode("utf-8"),
            "rfm_clusters_perfil.csv",
            "text/csv",
        )

def render_cluster_charts(rfm_df: pd.DataFrame):
    """
    Renderiza gráficos para análise técnica dos clusters (Boxplots e 3D),
    úteis antes da nomeação via LLM.
    """
    tab1, tab2, tab3 = st.tabs(["📦 Distribuição (Boxplots)", "🧊 Visualização 3D", "📍 Dispersão 2D"])

    if not HAS_PLOTLY:
        st.warning("A biblioteca 'plotly' não foi encontrada. Instale com `pip install plotly` para ver os gráficos.")
        return

    # Prepara dados para plotagem (ClusterId como string para cores discretas)
    plot_df = rfm_df.copy()
    plot_df["ClusterId"] = plot_df["ClusterId"].astype(str)
    
    with tab1:
        st.caption("Analise como cada variável se comporta dentro dos clusters. Isso ajuda a identificar manualmente quem são os 'Vips' ou 'Inativos'.")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            fig_r = px.box(plot_df, x="ClusterId", y="Recencia", color="ClusterId", title="Recência (Dias)")
            st.plotly_chart(fig_r, use_container_width=True)
            
        with c2:
            fig_f = px.box(plot_df, x="ClusterId", y="Frequencia", color="ClusterId", title="Frequência (Qtd)")
            st.plotly_chart(fig_f, use_container_width=True)
            
        with c3:
            fig_m = px.box(plot_df, x="ClusterId", y="Receita", color="ClusterId", title="Valor (R$)")
            st.plotly_chart(fig_m, use_container_width=True)

    with tab2:
        st.caption("Visão espacial dos agrupamentos.")
        fig_3d = px.scatter_3d(plot_df, x='Recencia', y='Frequencia', z='Receita',
                               color='ClusterId', opacity=0.7, title="Dispersão RFM 3D")
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab3:
        st.caption("Análise bidimensional dos pares de variáveis RFM.")
        p1, p2, p3 = st.columns(3)
        
        with p1:
            fig_rf = px.scatter(plot_df, x="Recencia", y="Frequencia", color="ClusterId", title="Recência x Frequência")
            st.plotly_chart(fig_rf, use_container_width=True)
        with p2:
            fig_fr = px.scatter(plot_df, x="Frequencia", y="Receita", color="ClusterId", title="Frequência x Receita")
            st.plotly_chart(fig_fr, use_container_width=True)
        with p3:
            fig_rr = px.scatter(plot_df, x="Recencia", y="Receita", color="ClusterId", title="Recência x Receita")
            st.plotly_chart(fig_rr, use_container_width=True)

def render_llm_results(rfm_out, cluster_profile, labels_df):
    rfm_named = rfm_out.merge(labels_df, on="ClusterId", how="left")
    prof_named = cluster_profile.merge(labels_df, on="ClusterId", how="left")

    st.subheader("Clusters nomeados (LLM)")
    st.dataframe(prof_named, use_container_width=True, hide_index=True)

    st.subheader("Explicações por segmento")
    for _, row in prof_named.sort_values("RankQualidade").iterrows():
        nome = row.get("SegmentoNome", f"Cluster {int(row['ClusterId'])}")
        st.markdown(f"### {nome}")
        if row.get("SegmentoDescricao"): st.write(row["SegmentoDescricao"])
        if row.get("Estrategias"):
            st.write("**Ações sugeridas:**")
            for a in row["Estrategias"]: st.write(f"- {a}")
        st.write("---")

    st.divider()
    st.header("📈 Gráficos (com rótulos descritivos)")
    render_scatter_grid(rfm_named)
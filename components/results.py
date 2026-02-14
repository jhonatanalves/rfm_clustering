import pandas as pd
import streamlit as st
from charts import render_scatter_grid


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
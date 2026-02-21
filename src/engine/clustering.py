import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def build_rfm_features(rfm: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Prepara as features matemáticas para o algoritmo de clusterização (K-Means).
    
    Transformações aplicadas:
    1. Inversão da Recência (R_inv = -Recencia).
    2. Logaritmo (log1p) em Frequência e Receita.
    3. Padronização (StandardScaler).

    Args:
        rfm (pd.DataFrame): Tabela RFM base.

    Returns:
        tuple[np.ndarray, pd.DataFrame]: 
            - Xs: Matriz numpy padronizada pronta para o modelo.
            - out: DataFrame com as colunas de features adicionadas.
    """
    out = rfm.copy()
    out["R_inv"] = -out["Recencia"]
    out["F_log"] = np.log1p(out["Frequencia"])
    out["M_log"] = np.log1p(out["Receita"])

    X = out[["R_inv", "F_log", "M_log"]].values
    Xs = StandardScaler().fit_transform(X)
    return Xs, out


def calcular_wcss(Xs: np.ndarray, k_min: int = 2, k_max: int = 10, random_state: int = 42) -> list[float]:
    """
    Calcula a inércia para diferentes valores de k (Elbow Method).
    """
    wcss = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=random_state)
        km.fit(Xs)
        wcss.append(float(km.inertia_))
    return wcss


def get_numero_otimo_clusters(wcss: list[float], k_min: int = 2, k_max: int = 10) -> int:
    """
    Calcula a maior distância perpendicular entre a curva WCSS e a reta (Knee/Elbow detection).
    """
    x1, y1 = k_min, wcss[0]
    x2, y2 = k_max, wcss[-1]

    distancias = []
    for i, y0 in enumerate(wcss):
        x0 = k_min + i
        numerador = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominador = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distancias.append(numerador / denominador)

    return (k_min + int(np.argmax(distancias)))


def cluster_rfm_joint(
    rfm: pd.DataFrame,
    n_clusters: int,
    random_state: int,
    n_init: int = 10,
    max_iter: int = 300,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executa o pipeline completo de clusterização:
    1. Gera features transformadas.
    2. Aplica K-Means.
    3. Calcula um ScoreComposto para ranquear os clusters.
    4. Gera estatísticas descritivas.
    """
    Xs, enriched = build_rfm_features(rfm)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init, max_iter=max_iter)
    cluster_id = km.fit_predict(Xs)
    enriched["ClusterId"] = cluster_id

    # Score composto
    enriched["ScoreComposto"] = Xs.sum(axis=1)

    # Perfil por cluster
    prof = (
        enriched.groupby("ClusterId")
        .agg(
            Clientes=("Cliente", "count"),
            Recencia_media=("Recencia", "mean"),
            Recencia_mediana=("Recencia", "median"),
            Frequencia_media=("Frequencia", "mean"),
            Frequencia_mediana=("Frequencia", "median"),
            Receita_media=("Receita", "mean"),
            Receita_mediana=("Receita", "median"),
            ScoreComposto_medio=("ScoreComposto", "mean"),
        )
        .reset_index()
    )

    # Ordena clusters por ScoreComposto_medio
    prof = prof.sort_values("ScoreComposto_medio", ascending=True).reset_index(drop=True)
    prof["RankQualidade"] = np.arange(len(prof))  # 0 pior, k-1 melhor
    total = prof["Clientes"].sum()
    prof["PctBase"] = (prof["Clientes"] / total).round(4)

    # junta o rank para cada cliente
    enriched = enriched.merge(prof[["ClusterId", "RankQualidade"]], on="ClusterId", how="left")

    # Limpa colunas auxiliares internas de features
    enriched = enriched.drop(columns=["R_inv", "F_log", "M_log"], errors="ignore")

    # Reordena colunas principais
    cols = ["Cliente", "ClusterId", "Recencia", "Frequencia", "Receita", "RankQualidade", "ScoreComposto"]
    enriched = enriched[cols]

    # Arredonda perfil para display
    prof_display = prof.copy()
    for c in prof_display.columns:
        if c.endswith("_media") or c.endswith("_mediana") or "ScoreComposto" in c:
            prof_display[c] = prof_display[c].astype(float).round(2)

    return enriched, prof_display
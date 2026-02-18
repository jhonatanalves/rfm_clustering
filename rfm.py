import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def build_rfm_table(
    data: pd.DataFrame,
    customer_col: str,
    date_col: str,
    monetary_col: str,
    order_col: str | None,
    approved_col: str | None,
    approved_values: list[str],
) -> pd.DataFrame:
    """
    Converte o DataFrame de transações brutas em uma tabela RFM (Recência, Frequência, Receita) por cliente.

    Args:
        data (pd.DataFrame): DataFrame original com as transações.
        customer_col (str): Nome da coluna de ID do cliente.
        date_col (str): Nome da coluna de data.
        monetary_col (str): Nome da coluna de valor monetário.
        order_col (str | None): Nome da coluna de ID do pedido. Se None, conta linhas como frequência.
        approved_col (str | None): Nome da coluna de status do pedido.
        approved_values (list[str]): Lista de status considerados válidos para o cálculo.

    Returns:
        pd.DataFrame: DataFrame contendo ['Cliente', 'Recencia', 'Frequencia', 'Receita'].
    """
    df = data.copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[customer_col, date_col])

    if approved_col and approved_values:
        df = df[df[approved_col].astype(str).isin([str(v) for v in approved_values])]

    df[monetary_col] = pd.to_numeric(df[monetary_col], errors="coerce").fillna(0)

    # Recência
    last_buy = df.groupby(customer_col)[date_col].max().reset_index()
    ref_date = last_buy[date_col].max()
    last_buy["Recencia"] = (ref_date - last_buy[date_col]).dt.days
    last_buy = last_buy[[customer_col, "Recencia"]]

    # Frequência
    if order_col:
        freq = df.groupby(customer_col)[order_col].nunique().reset_index(name="Frequencia")
    else:
        freq = df.groupby(customer_col).size().reset_index(name="Frequencia")

    # Receita
    rev = df.groupby(customer_col)[monetary_col].sum().reset_index(name="Receita")

    rfm = last_buy.merge(freq, on=customer_col).merge(rev, on=customer_col)
    rfm = rfm.rename(columns={customer_col: "Cliente"})
    return rfm


def build_rfm_features(rfm: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Prepara as features matemáticas para o algoritmo de clusterização (K-Means).
    
    Transformações aplicadas:
    1. Inversão da Recência (R_inv = -Recencia): Para que valores maiores indiquem "melhor" cliente, alinhando com Frequência e Receita.
    2. Logaritmo (log1p) em Frequência e Receita: Para reduzir a assimetria (skewness) comum em dados financeiros.
    3. Padronização (StandardScaler): Para colocar todas as variáveis na mesma escala (média 0, desvio padrão 1).

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
    Calcula a inércia para diferentes valores de k.
    Utilizado para gerar o gráfico do cotovelo.

    Args:
        Xs (np.ndarray): Matriz de features padronizadas.
        k_min (int): Valor mínimo de k a testar.
        k_max (int): Valor máximo de k a testar.
        random_state (int): Semente aleatória para reprodutibilidade.

    Returns:
        list[float]: Lista contendo os valores de inércia para cada k.
    """
    wcss = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=random_state)
        km.fit(Xs)
        wcss.append(float(km.inertia_))
    return wcss


def get_numero_otimo_clusters(wcss: list[float], k_min: int = 2, k_max: int = 10) -> int:
    """
    Calcula a maior distância perpendicular entre a curva WCSS e a reta que liga o primeiro e o último ponto (Knee/Elbow detection).

    Args:
        wcss (list[float]): Lista de inércias calculada por calcular_wcss.
        k_min (int): O k inicial correspondente ao primeiro elemento da lista wcss.
        k_max (int): O k final correspondente ao último elemento.

    Returns:
        int: O valor ótimo de k.
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
    3. Calcula um ScoreComposto para ranquear os clusters do pior para o melhor.
    4. Gera estatísticas descritivas para cada cluster.

    Args:
        rfm (pd.DataFrame): Tabela RFM base.
        n_clusters (int): Número de clusters desejado.
        random_state (int): Seed.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - enriched: DataFrame detalhado por cliente, com ClusterId e Rank.
            - prof_display: DataFrame agregado com médias/medianas por cluster.
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
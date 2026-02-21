import pandas as pd


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
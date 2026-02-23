import pandas as pd

def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 30) -> list[pd.DataFrame]:
    """
    Divide o DataFrame em uma lista de DataFrames menores (chunks).
    """
    return [df.iloc[i : i + chunk_size].copy() for i in range(0, len(df), chunk_size)]


def estimate_max_output_tokens(n_clusters: int) -> int:
    """
    Estima o limite de tokens de saída: ~1000 tokens por cluster + margem de segurança.
    Limite máximo fixado em 8192 tokens .
    """
    return min(10000, 200 * n_clusters + 200)
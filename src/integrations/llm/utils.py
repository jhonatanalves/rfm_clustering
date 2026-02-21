import pandas as pd

def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 30) -> list[pd.DataFrame]:
    """
    Divide o DataFrame em uma lista de DataFrames menores (chunks).
    """
    return [df.iloc[i : i + chunk_size].copy() for i in range(0, len(df), chunk_size)]


def estimate_max_output_tokens(n_clusters: int) -> int:
    """
    Estima o limite de tokens de saída: ~250 tokens por cluster + margem de segurança.
    Limite máximo fixado em 6000 tokens (seguro para maioria dos modelos modernos).
    """
    return min(6000, 250 * n_clusters + 200)
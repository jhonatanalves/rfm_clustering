import pandas as pd
from cleaning.auto_cleaner import AutoCleaner

def apply_autoclean(
    df: pd.DataFrame,
    tratar_duplicados: bool,
    imputacao: str | bool,    # "median" | "mean" | "most_frequent" | "knn" | "linreg" | "delete" | False
    tratar_outliers: str | bool, # "winz" | "delete" | "median" | False
) -> pd.DataFrame:
    """
    Aplica AutoClean em modo manual, configurando:
    - duplicates: True/False
    - missing_num: "median" | "mean" | "most_frequent" | "knn" | "linreg" | "delete" | False
    - missing_categ: "most_frequent" | "knn" | "logreg" | "delete" | False
    - outliers: "winz" | "delete" | "median" | False
    """

    # Mapeamento da estratégia geral para (numérico, categórico)
    if imputacao == "median":
        missing_num = "median"
        missing_categ = "most_frequent"
    elif imputacao == "mean":
        missing_num = "mean"
        missing_categ = "most_frequent"
    elif imputacao == "most_frequent":
        missing_num = "most_frequent"
        missing_categ = "most_frequent"
    elif imputacao == "knn":
        missing_num = "knn"
        missing_categ = "knn"
    elif imputacao == "linreg":
        missing_num = "linreg"
        missing_categ = "logreg"
    elif imputacao == "delete":
        missing_num = "delete"
        missing_categ = "delete"
    elif imputacao is False:
        missing_num = False
        missing_categ = False
    else:
        # Fallback seguro
        missing_num = "median"
        missing_categ = "most_frequent"

    cleaner = AutoCleaner(
        mode="manual",
        duplicates=True if tratar_duplicados else False,
        missing_num=missing_num,
        missing_categ=missing_categ,
        outliers=tratar_outliers,
    )
    
    return cleaner.clean(df)
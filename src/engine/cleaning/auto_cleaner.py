import pandas as pd
import numpy as np
from typing import Literal, Union, Optional
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.preprocessing import OrdinalEncoder

# Definição de Tipos para clareza e validação
OutlierOptions = Literal['auto', 'winz', 'delete', 'median', False]
MissingCategOptions = Literal['auto', 'logreg', 'knn', 'most_frequent', 'delete', False]
MissingNumOptions = Literal['auto', 'linreg', 'knn', 'mean', 'median', 'most_frequent', 'delete', False]
DuplicateOptions = Literal['auto', True, False]
ModeOptions = Literal['auto', 'manual']

class AutoCleaner:
    def __init__(
        self,
        outliers: OutlierOptions = 'auto',
        missing_categ: MissingCategOptions = 'auto',
        missing_num: MissingNumOptions = 'auto',
        duplicates: DuplicateOptions = 'auto',
        mode: ModeOptions = 'auto'
    ):
        self.outliers = outliers
        self.missing_categ = missing_categ
        self.missing_num = missing_num
        self.duplicates = duplicates
        self.mode = mode

        if self.mode == 'auto':
            self._set_auto_defaults()

    def _set_auto_defaults(self):
        """Define estratégias seguras para o modo totalmente automático."""
        if self.outliers == 'auto':
            self.outliers = 'winz'  # Winsorization é mais seguro que deletar
        if self.missing_num == 'auto':
            self.missing_num = 'median'
        if self.missing_categ == 'auto':
            self.missing_categ = 'most_frequent'
        if self.duplicates == 'auto':
            self.duplicates = True

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executa o pipeline de limpeza."""
        df = df.copy()
        
        # 1. Duplicatas
        if self.duplicates:
            df = df.drop_duplicates()

        # 2. Missing Values
        df = self._handle_missing(df)

        # 3. Outliers (apenas em colunas numéricas)
        if self.outliers:
            df = self._handle_outliers(df)

        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        # Identificar colunas
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns

        # --- Tratamento Numérico ---
        if self.missing_num and len(num_cols) > 0:
            if self.missing_num == 'delete':
                df = df.dropna(subset=num_cols)
            
            elif self.missing_num in ['mean', 'median', 'most_frequent']:
                imputer = SimpleImputer(strategy=self.missing_num)
                df[num_cols] = imputer.fit_transform(df[num_cols])
            
            elif self.missing_num == 'knn':
                imputer = KNNImputer(n_neighbors=5)
                df[num_cols] = imputer.fit_transform(df[num_cols])
            
            elif self.missing_num == 'linreg':
                # IterativeImputer com BayesianRidge é uma aproximação robusta de regressão linear
                imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42)
                df[num_cols] = imputer.fit_transform(df[num_cols])

        # --- Tratamento Categórico ---
        if self.missing_categ and len(cat_cols) > 0:
            if self.missing_categ == 'delete':
                df = df.dropna(subset=cat_cols)
            
            elif self.missing_categ == 'most_frequent':
                imputer = SimpleImputer(strategy='most_frequent')
                df[cat_cols] = imputer.fit_transform(df[cat_cols])
            
            elif self.missing_categ in ['knn', 'logreg']:
                # Para KNN/LogReg em categóricos, precisamos codificar -> imputar -> decodificar
                # Nota: Isso é computacionalmente custoso.
                
                # 1. Codificar (Ordinal) preservando NaNs
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
                
                # Precisamos tratar os NaNs manualmente no encoder pois o OrdinalEncoder padrão não ignora NaNs na entrada facilmente em versões antigas
                # Uma abordagem robusta: preencher temporariamente, fitar, e depois recolocar NaN onde era NaN original
                # Simplificação: Usar Pandas factorize ou map, mas OrdinalEncoder é melhor para pipeline.
                # Vamos usar uma abordagem híbrida segura:
                
                df_cat_encoded = df[cat_cols].copy()
                encoders = {}
                
                for col in cat_cols:
                    # Cria mapeamento apenas com valores existentes
                    non_nulls = df[col].dropna().unique()
                    mapping = {val: i for i, val in enumerate(non_nulls)}
                    encoders[col] = {i: val for val, i in mapping.items()} # reverso
                    df_cat_encoded[col] = df[col].map(mapping) # NaNs permanecem NaNs

                # 2. Imputar
                if self.missing_categ == 'knn':
                    imputer = KNNImputer(n_neighbors=5)
                else: # logreg
                    # IterativeImputer funciona para regressão, mas pode ser usado para classificação se arredondarmos
                    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
                
                imputed_data = imputer.fit_transform(df_cat_encoded)
                df_cat_encoded = pd.DataFrame(imputed_data, columns=cat_cols, index=df.index)

                # 3. Decodificar (arredondando para o inteiro mais próximo)
                for col in cat_cols:
                    df_cat_encoded[col] = df_cat_encoded[col].round().astype(int)
                    # Clip para garantir que está dentro dos limites do encoder
                    max_val = max(encoders[col].keys()) if encoders[col] else 0
                    df_cat_encoded[col] = df_cat_encoded[col].clip(0, max_val)
                    df[col] = df_cat_encoded[col].map(encoders[col])

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in num_cols:
            if self.outliers == 'delete':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            elif self.outliers == 'winz':
                # Winsorization: limita os extremos aos percentis 5% e 95%
                lower = df[col].quantile(0.05)
                upper = df[col].quantile(0.95)
                df[col] = df[col].clip(lower=lower, upper=upper)
            
            elif self.outliers == 'median':
                # Identifica via IQR e substitui pela mediana
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                median_val = df[col].median()
                mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                if mask.any():
                    df.loc[mask, col] = median_val
        
        return df
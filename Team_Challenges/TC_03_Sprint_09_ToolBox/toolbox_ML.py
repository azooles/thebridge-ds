#!/usr/bin/env python
# coding: utf-8

# toolbox_ML.py — mi navaja suiza para no repetir código como un mono

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# =============================================================================
# 1. describe_df
# =============================================================================

def describe_df(df):
    """
    Devuelve un resumen del dataframe con tipos, porcentaje de nulos,
    valores únicos y cardinalidad de cada columna.

    Argumentos:
    df (pd.DataFrame): El dataframe a describir.

    Retorna:
    pd.DataFrame: Dataframe con una fila por cada columna del original,
                  con las columnas: DATA_TYPE, MISSINGS (%), UNIQUE_VALUES, CARDIN (%).
    """
    # construyo el resumen columna a columna
    resumen = pd.DataFrame({
        "DATA_TYPE": df.dtypes,
        "MISSINGS (%)": (df.isnull().mean() * 100).round(2),
        "UNIQUE_VALUES": df.nunique(),
        "CARDIN (%)": (df.nunique() / len(df) * 100).round(2)
    })
    return resumen.T


# =============================================================================
# 2. tipifica_variables
# =============================================================================

def tipifica_variables(df, umbral_categoria, umbral_continua):
    """
    Sugiere el tipo de cada variable del dataframe según su cardinalidad.

    Argumentos:
    df (pd.DataFrame): El dataframe a analizar.
    umbral_categoria (int): Cardinalidad máxima para considerar una variable categórica.
    umbral_continua (float): Porcentaje mínimo de cardinalidad para considerar continua.

    Retorna:
    pd.DataFrame: Dataframe con columnas 'nombre_variable' y 'tipo_sugerido'.
    """
    resultados = []

    for col in df.columns:
        cardinalidad = df[col].nunique()
        pct_cardinalidad = cardinalidad / len(df) * 100

        # la lógica de clasificación, de más específica a más general
        if cardinalidad == 2:
            tipo = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo = "Categórica"
        elif pct_cardinalidad >= umbral_continua:
            tipo = "Numerica Continua"
        else:
            tipo = "Numerica Discreta"

        resultados.append({"nombre_variable": col, "tipo_sugerido": tipo})

    return pd.DataFrame(resultados)


# =============================================================================
# 3. get_features_num_regression
# =============================================================================

def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    """
    Devuelve las columnas numéricas cuya correlación con el target supera
    el umbral dado, con test de significación estadística opcional.

    Argumentos:
    df (pd.DataFrame): El dataframe.
    target_col (str): Nombre de la columna target (debe ser numérica continua).
    umbral_corr (float): Umbral mínimo de correlación en valor absoluto (0 a 1).
    pvalue (float o None): Si no es None, filtra también por significación estadística.

    Retorna:
    list o None: Lista de columnas que cumplen los criterios, o None si hay error.
    """
    # --- checks de entrada ---
    if target_col not in df.columns:
        print(f"Error: '{target_col}' no existe en el dataframe.")
        return None

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: '{target_col}' no es numérica.")
        return None

    if not (0 <= umbral_corr <= 1):
        print("Error: 'umbral_corr' debe estar entre 0 y 1.")
        return None

    if pvalue is not None and not (0 < pvalue < 1):
        print("Error: 'pvalue' debe estar entre 0 y 1 (o ser None).")
        return None

    # columnas numéricas excluyendo el target
    cols_numericas = df.select_dtypes(include=np.number).columns.tolist()
    cols_numericas = [c for c in cols_numericas if c != target_col]

    features_seleccionadas = []

    for col in cols_numericas:
        # elimino NaNs para el cálculo
        datos_limpios = df[[col, target_col]].dropna()
        corr, p_val = stats.pearsonr(datos_limpios[col], datos_limpios[target_col])

        # filtro por correlación
        if abs(corr) > umbral_corr:
            # filtro adicional por pvalue si procede
            if pvalue is None or p_val <= pvalue:
                features_seleccionadas.append(col)

    return features_seleccionadas


# =============================================================================
# 4. plot_features_num_regression
# =============================================================================

def plot_features_num_regression(df, target_col="", columns=None, umbral_corr=0, pvalue=None):
    """
    Pinta pairplots de las features numéricas correlacionadas con el target.
    Si columns está vacía, usa todas las numéricas del dataframe.

    Argumentos:
    df (pd.DataFrame): El dataframe.
    target_col (str): Nombre de la columna target.
    columns (list): Lista de columnas a evaluar. Si vacía, usa todas las numéricas.
    umbral_corr (float): Umbral mínimo de correlación en valor absoluto.
    pvalue (float o None): Nivel de significación estadística opcional.

    Retorna:
    list o None: Lista de columnas que cumplen los criterios, o None si hay error.
    """
    # --- checks de entrada ---
    if target_col not in df.columns:
        print(f"Error: '{target_col}' no existe en el dataframe.")
        return None

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: '{target_col}' no es numérica.")
        return None

    if not (0 <= umbral_corr <= 1):
        print("Error: 'umbral_corr' debe estar entre 0 y 1.")
        return None

    if pvalue is not None and not (0 < pvalue < 1):
        print("Error: 'pvalue' debe estar entre 0 y 1 (o ser None).")
        return None

    # si no me pasan columnas, uso todas las numéricas
    if columns is None:
        columns = []
    if len(columns) == 0:
        columns = df.select_dtypes(include=np.number).columns.tolist()
        columns = [c for c in columns if c != target_col]

    # reutilizo get_features_num_regression para filtrar
    features_validas = get_features_num_regression(df, target_col, umbral_corr, pvalue)

    if features_validas is None:
        return None

    # me quedo solo con las que están en columns Y pasan el filtro
    features_a_pintar = [c for c in columns if c in features_validas]

    if not features_a_pintar:
        print("No hay columnas que cumplan los criterios de correlación.")
        return []

    # pairplot en grupos de máximo 4 features + el target (5 columnas en total)
    chunk_size = 4
    for i in range(0, len(features_a_pintar), chunk_size):
        chunk = features_a_pintar[i:i + chunk_size]
        cols_plot = [target_col] + chunk
        sns.pairplot(df[cols_plot].dropna(), diag_kind="kde")
        plt.suptitle(f"Pairplot — grupo {i // chunk_size + 1}", y=1.02)
        plt.show()

    return features_a_pintar


# =============================================================================
# 5. get_features_cat_regression
# =============================================================================

def get_features_cat_regression(df, target_col, pvalue=0.05):
    """
    Devuelve las columnas categóricas cuya relación con el target numérico
    es estadísticamente significativa (ANOVA si hay más de 2 grupos, t-test si hay 2).

    Argumentos:
    df (pd.DataFrame): El dataframe.
    target_col (str): Nombre de la columna target (debe ser numérica).
    pvalue (float): Nivel de significación (por defecto 0.05).

    Retorna:
    list o None: Lista de columnas categóricas significativas, o None si hay error.
    """
    # --- checks de entrada ---
    if target_col not in df.columns:
        print(f"Error: '{target_col}' no existe en el dataframe.")
        return None

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: '{target_col}' no es numérica.")
        return None

    if not (0 < pvalue < 1):
        print("Error: 'pvalue' debe estar entre 0 y 1.")
        return None

    # columnas categóricas (object o category)
    cols_categoricas = df.select_dtypes(include=["object", "category"]).columns.tolist()

    features_seleccionadas = []

    for col in cols_categoricas:
        datos_limpios = df[[col, target_col]].dropna()
        grupos = [grupo[target_col].values for _, grupo in datos_limpios.groupby(col)]

        # necesito al menos 2 grupos con datos
        if len(grupos) < 2:
            continue

        # t-test para 2 grupos, ANOVA para más
        if len(grupos) == 2:
            _, p_val = stats.ttest_ind(grupos[0], grupos[1])
        else:
            _, p_val = stats.f_oneway(*grupos)

        if p_val < pvalue:
            features_seleccionadas.append(col)

    return features_seleccionadas


# =============================================================================
# 6. plot_features_cat_regression
# =============================================================================

def plot_features_cat_regression(df, target_col="", columns=None, pvalue=0.05, with_individual_plot=False):
    """
    Pinta histogramas agrupados del target para cada variable categórica
    que tenga relación estadísticamente significativa con él.

    Argumentos:
    df (pd.DataFrame): El dataframe.
    target_col (str): Nombre de la columna target.
    columns (list): Lista de columnas categóricas a evaluar. Si vacía, usa todas.
    pvalue (float): Nivel de significación (por defecto 0.05).
    with_individual_plot (bool): Si True, pinta un gráfico individual por cada categoría.

    Retorna:
    list o None: Lista de columnas que cumplen los criterios, o None si hay error.
    """
    # --- checks de entrada ---
    if target_col not in df.columns:
        print(f"Error: '{target_col}' no existe en el dataframe.")
        return None

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: '{target_col}' no es numérica.")
        return None

    if not (0 < pvalue < 1):
        print("Error: 'pvalue' debe estar entre 0 y 1.")
        return None

    # si no me pasan columnas, uso todas las categóricas
    if columns is None:
        columns = []
    if len(columns) == 0:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # filtro las que son estadísticamente significativas
    features_validas = get_features_cat_regression(df, target_col, pvalue)

    if features_validas is None:
        return None

    features_a_pintar = [c for c in columns if c in features_validas]

    if not features_a_pintar:
        print("No hay columnas categóricas significativas para pintar.")
        return []

    for col in features_a_pintar:
        datos_limpios = df[[col, target_col]].dropna()

        if with_individual_plot:
            # un histograma por cada valor único de la categórica
            categorias = datos_limpios[col].unique()
            fig, axes = plt.subplots(1, len(categorias), figsize=(5 * len(categorias), 4), sharey=True)
            if len(categorias) == 1:
                axes = [axes]
            for ax, cat in zip(axes, categorias):
                subset = datos_limpios[datos_limpios[col] == cat][target_col]
                ax.hist(subset, bins=20, edgecolor="black", alpha=0.7)
                ax.set_title(f"{col} = {cat}")
                ax.set_xlabel(target_col)
            plt.suptitle(f"Distribución de '{target_col}' por '{col}'")
            plt.tight_layout()
            plt.show()
        else:
            # histograma agrupado en un solo gráfico
            fig, ax = plt.subplots(figsize=(8, 4))
            for cat, grupo in datos_limpios.groupby(col):
                ax.hist(grupo[target_col], bins=20, alpha=0.6, label=str(cat), edgecolor="black")
            ax.set_title(f"Distribución de '{target_col}' por '{col}'")
            ax.set_xlabel(target_col)
            ax.legend(title=col)
            plt.tight_layout()
            plt.show()

    return features_a_pintar

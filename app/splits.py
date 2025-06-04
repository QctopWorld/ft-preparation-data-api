# splits.py

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime


def random_split(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split aléatoire simple.
    Renvoie (train_df, test_df) en prenant test_frac de manière aléatoire.
    """
    if not (0 < test_frac < 1):
        raise ValueError("test_frac doit être dans (0, 1)")
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n_test = int(len(df_shuffled) * test_frac)
    test_df  = df_shuffled.iloc[:n_test].reset_index(drop=True)
    train_df = df_shuffled.iloc[n_test:].reset_index(drop=True)
    return train_df, test_df


def stratified_split_univariate(
    df: pd.DataFrame,
    label_col: str,
    test_frac: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split stratifié univarié sur label_col.
    Garantit que chaque modalité de label_col est présente en proportion dans train/test.
    """
    if label_col not in df.columns:
        raise KeyError(f"Colonne '{label_col}' introuvable dans le DataFrame")
    if not (0 < test_frac < 1):
        raise ValueError("test_frac doit être dans (0, 1)")
    train_parts = []
    test_parts  = []
    # pour chaque modalité de label_col, tirer aléatoirement test_frac lignes
    for _, grp in df.groupby(label_col):
        grp_shuf = grp.sample(frac=1, random_state=random_state).reset_index(drop=True)
        n_test   = int(len(grp_shuf) * test_frac)
        test_parts.append(grp_shuf.iloc[:n_test])
        train_parts.append(grp_shuf.iloc[n_test:])
    train_df = pd.concat(train_parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df  = pd.concat(test_parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return train_df, test_df


def stratified_split_multivariate(
    df: pd.DataFrame,
    strat_cols: List[str],
    test_frac: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split stratifié multiconjoint : on crée une clé qui concatène les colonnes strat_cols,
    puis on stratifie sur cette clé.
    Exemple : strat_cols=['Groupe', 'Source'] → clé='Groupe|Source'.
    """
    for c in strat_cols:
        if c not in df.columns:
            raise KeyError(f"Colonne '{c}' introuvable dans le DataFrame")
    if not (0 < test_frac < 1):
        raise ValueError("test_frac doit être dans (0, 1)")

    # 1) Construire une colonne temporaire combinant toutes les strat_cols
    key = df[strat_cols].astype(str).agg("|".join, axis=1)
    df_with_key = df.copy()
    df_with_key["_strat_key"] = key

    train_parts = []
    test_parts  = []
    for _, grp in df_with_key.groupby("_strat_key"):
        grp_shuf = grp.sample(frac=1, random_state=random_state).reset_index(drop=True)
        n_test   = int(len(grp_shuf) * test_frac)
        test_parts.append(grp_shuf.iloc[:n_test])
        train_parts.append(grp_shuf.iloc[n_test:])
    train_df = pd.concat(train_parts).drop(columns=["_strat_key"])
    test_df  = pd.concat(test_parts).drop(columns=["_strat_key"])
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df  = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return train_df, test_df


def temporal_split(
    df: pd.DataFrame,
    date_col: str,
    cutoff: Optional[str] = None,
    test_frac: Optional[float] = None,
    date_format: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporel.
    - Si cutoff (str ou datetime) est fourni : toutes les lignes
      où date_col < cutoff → train, sinon → test.
    - Si cutoff est None et test_frac fourni : on ordonne par date_col ascendant,
      puis on prend les derniers test_frac pour test.
    date_format : format strptime si date_col est en string (ex: "%Y-%m-%d").
    """
    if date_col not in df.columns:
        raise KeyError(f"Colonne '{date_col}' introuvable dans le DataFrame")
    if cutoff is None and (test_frac is None or not (0 < test_frac < 1)):
        raise ValueError("Soit cutoff (date) doit être défini, soit test_frac dans (0,1).")

    df_copy = df.copy()
    # 1) S’assurer que date_col est bien datetime
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
        if date_format:
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], format=date_format, errors="coerce")
        else:
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
    if df_copy[date_col].isna().any():
        raise ValueError(f"Certaines valeurs de '{date_col}' n'ont pas pu être converties en datetime")

    # 2) Si cutoff donné
    if cutoff is not None:
        if isinstance(cutoff, str):
            cutoff_dt = pd.to_datetime(cutoff, format=date_format, errors="coerce")
            if pd.isna(cutoff_dt):
                raise ValueError(f"Impossible de parser cutoff='{cutoff}' en datetime")
        else:
            cutoff_dt = pd.to_datetime(cutoff)
        train_df = df_copy.loc[df_copy[date_col] < cutoff_dt].reset_index(drop=True)
        test_df  = df_copy.loc[df_copy[date_col] >= cutoff_dt].reset_index(drop=True)
        return train_df, test_df

    # 3) Sinon, on découpe par fraction en triant par date
    df_sorted = df_copy.sort_values(by=date_col).reset_index(drop=True)
    n_test    = int(len(df_sorted) * test_frac)
    test_df   = df_sorted.iloc[-n_test:].reset_index(drop=True)
    train_df  = df_sorted.iloc[:-n_test].reset_index(drop=True)
    return train_df, test_df


def group_aware_split(
    df: pd.DataFrame,
    group_col: str,
    test_frac: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split en veillant à ce qu’un même groupe (group_col) n’apparaisse
    qu’une seule fois dans train OU test. On tire d’abord un sous-ensemble
    de groupes aléatoires jusqu’à atteindre la proportion test_frac.
    """
    if group_col not in df.columns:
        raise KeyError(f"Colonne '{group_col}' introuvable dans le DataFrame")
    if not (0 < test_frac < 1):
        raise ValueError("test_frac doit être dans (0, 1)")

    # 1) Liste unique des groupes
    unique_groups = df[group_col].dropna().unique()
    rng = np.random.RandomState(random_state)
    # 2) Mélanger la liste des groupes
    shuffled = rng.permutation(unique_groups)
    n_total = len(unique_groups)
    n_test_grp = int(n_total * test_frac)
    test_groups = set(shuffled[:n_test_grp])
    # 3) Construire les DataFrames
    test_df  = df[df[group_col].isin(test_groups)].reset_index(drop=True)
    train_df = df[~df[group_col].isin(test_groups)].reset_index(drop=True)
    return train_df, test_df

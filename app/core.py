"""core.py – Fonctions de préparation de données
-------------------------------------------------
Réimplémente l'ancien `process_csv.py` sous forme de fonctions pures
(pandas), réutilisables hors du contexte FastAPI.
"""
from __future__ import annotations

import datetime, os, re, json, warnings
import time, sys, ctypes
from typing import Dict, List, Any
from pathlib import Path

import pandas as pd
import numpy as np

try:
    import yaml  # facultatif (parse_rules)
except ModuleNotFoundError:
    yaml = None

# ──────────────────────────────────────────
#  Helper JSON : convertit np.* vers types natifs
# ──────────────────────────────────────────

def json_serial(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime.datetime)):
        return obj.isoformat()
    raise TypeError(f"Type non sérialisable : {type(obj).__name__}")

# ──────────────────────────────────────────
#  Parsing des règles d’agrégation
# ──────────────────────────────────────────

def parse_rules(agg_inline: str | None = None, aggfile: str | None = None) -> Dict[str, dict]:
    rules: Dict[str, dict] = {}
    if aggfile:
        with open(aggfile, "r", encoding="utf-8") as f:
            if aggfile.lower().endswith((".yml", ".yaml")) and yaml:
                rules.update(yaml.safe_load(f))
            else:
                rules.update(json.load(f))
    if agg_inline:
        for item in agg_inline.split(","):
            if not item.strip():
                continue
            col, spec = item.split(":", 1)
            m = re.fullmatch(r"(\w+)(?:\(([^)]+)\))?", spec.strip())
            if not m:
                raise ValueError(f"Spécification invalide : {item}")
            op, weight = m.group(1), m.group(2)
            rules[col.strip()] = {"op": op, "weight": weight}
    return rules

# ──────────────────────────────────────────
#  Helpers détection type colonne
# ──────────────────────────────────────────

def _col_kind(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    sample = series.dropna().astype(str).head(100)
    if len(sample):
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.8:
            return "date"
    return "string"

def _ensure_datetime(df: pd.DataFrame, col: str):
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors="coerce")

# ──────────────────────────────────────────
#  Harmonisation déclarative des doublons
# ──────────────────────────────────────────

# On conserve _ALLOWED pour valider les règles, mais on passe par une version vectorisée ci-dessous
_ALLOWED = {
    "numeric": {"sum", "min", "max", "mean", "median", "wm", "first", "last"},
    "date":    {"min", "max", "first", "last"},
    "string":  {"first", "last", "mode"},
}

def harmoniser_doublons_vectorise(df: pd.DataFrame, rules: Dict[str, dict]) -> pd.DataFrame:
    """
    Version vectorisée de l’harmonisation déclarative :
    - Parcourt rules pour chaque colonne et applique groupby+agg au lieu de apply(lambda…)
    - Construit un DataFrame final à partir de l’index groupé
    """
    if not rules:
        return df

    # On récupère la liste de tous les "Titre" uniques pour reconstruire le df fusionné
    titres = df["Titre"].unique()
    # Création d’un DataFrame résultat qui contiendra la colonne "Titre" d’abord
    result = pd.DataFrame({"Titre": titres})

    for col, cfg in rules.items():
        if col not in df.columns:
            raise KeyError(f"Colonne inconnue : {col}")
        op = cfg.get("op", "copy")
        kind = _col_kind(df[col])

        # Si on demande min/max/first/last sur un champ string, on le convertit d’abord en datetime si possible.
        if op in {"min", "max", "first", "last"} and kind == "string":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                _ensure_datetime(df, col)
            kind = _col_kind(df[col])

        if op not in _ALLOWED[kind] and op != "copy":
            raise ValueError(f"Opération « {op} » interdite pour {col} ({kind})")
        if kind == "date":
            _ensure_datetime(df, col)

        # Exécution vectorisée selon l’opération
        if op == "wm":
            wcol = cfg.get("weight")
            if not wcol or wcol not in df.columns:
                raise KeyError(f"wm() : colonne de poids manquante pour {col}")
            if not pd.api.types.is_numeric_dtype(df[wcol]):
                raise TypeError(f"wm() : « {wcol} » n’est pas numérique")
            # On crée temporairement la colonne pondérée
            df["_numxwt"] = df[col] * df[wcol]
            grp_numxwt = df.groupby("Titre")["_numxwt"].sum()
            grp_wt     = df.groupby("Titre")[wcol].sum()
            # On ajoute la série résultat dans result, en s’assurant de l’index
            result[col] = (grp_numxwt / grp_wt).reindex(titres).values
            df.drop(columns=["_numxwt"], inplace=True)

        elif op == "sum":
            grp_sum = df.groupby("Titre")[col].sum()
            result[col] = grp_sum.reindex(titres).values

        elif op in {"min", "max", "first", "last", "median", "mean"}:
            # pandas agrège via agg directement
            grp_agg = df.groupby("Titre")[col].agg(op)
            result[col] = grp_agg.reindex(titres).values

        elif op == "mode":
            # Mode n’est pas natif en C, on utilise un petit apply sur chaque groupe
            # Mais on ne l’utilise que si vraiment nécessaire car coûteux.
            def _mode_or_first(s: pd.Series) -> Any:
                m = s.mode()
                return m.iat[0] if not m.empty else s.iat[0]
            grp_mode = df.groupby("Titre")[col].apply(_mode_or_first)
            result[col] = grp_mode.reindex(titres).values

        else:  # "copy" ou cas par défaut
            grp_first = df.groupby("Titre")[col].first()
            result[col] = grp_first.reindex(titres).values

    return result

# ──────────────────────────────────────────
#  Fusion « vectorisée » des doublons (remplace suppression_doublons_legacy)
# ──────────────────────────────────────────

def suppression_doublons_vectorisee(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne les lignes groupées par "Titre" en utilisant des agrégations vectorisées :
      - Date publication FB : min
      - Impressions : sum
      - Clics : sum
      - CTR (%) : somme(CTR*Impressions)/somme(Impressions)
    """
    # 1) Convertir en datetime si besoin
    _ensure_datetime(df, "Date publication FB")

    # 2) Calculer en mode vectorisé :
    #    – somme des Impressions
    grp_impr   = df.groupby("Titre")["Impressions"].sum()
    #    – somme des Clics
    grp_clics  = df.groupby("Titre")["Clics"].sum()
    #    – pondération CTR*Impr
    df["_CTRxImpr"]    = df["CTR (%)"] * df["Impressions"]
    grp_ctrximpr = df.groupby("Titre")["_CTRxImpr"].sum()
    #    – date min de publication FB
    grp_date_min = df.groupby("Titre")["Date publication FB"].min()

    # 3) Reconstruire le DataFrame résultat
    result = pd.DataFrame({
        "Titre":               grp_impr.index,
        "Date publication FB": grp_date_min.values,
        "CTR (%)":             (grp_ctrximpr / grp_impr).values,
        "Impressions":         grp_impr.values,
        "Clics":               grp_clics.values,
    })

    # 4) Nettoyer : supprimer la colonne temporaire
    df.drop(columns=["_CTRxImpr"], inplace=True)

    return result.reset_index(drop=True)

# ──────────────────────────────────────────
#  Fusion « legacy » des doublons (gardée à titre de référence, non utilisée)
# ──────────────────────────────────────────

def suppression_doublons_legacy(df: pd.DataFrame) -> pd.DataFrame:
    _ensure_datetime(df, "Date publication FB")
    return (
        df.groupby("Titre")
          .apply(lambda g: pd.Series({
              "Date publication FB": g["Date publication FB"].min(),
              "CTR (%)": (g["CTR (%)"] * g["Impressions"]).sum() / g["Impressions"].sum(),
              "Impressions": g["Impressions"].sum(),
              "Clics": g["Clics"].sum(),
          }))
          .reset_index()
    )

# ──────────────────────────────────────────
#  Catégorisation numérique
# ──────────────────────────────────────────

def ajouter_categorie(df: pd.DataFrame, var: str, bins: List[float], 
                cat_label0: str = "g0", cat_label1: str = "g1", 
                cat_label2: str = "g2", cat_label3: str = "g3",
                **cat_labels) -> pd.DataFrame:
    if var not in df.columns:
        raise KeyError(f"Colonne à catégoriser inconnue : {var}")
    if df[var].dtype == object:
        df[var] = (df[var].astype(str)
                            .str.replace(",", ".", regex=False)
                            .str.replace("%", "", regex=False))
    df[var] = pd.to_numeric(df[var], errors="coerce")
    if not pd.api.types.is_numeric_dtype(df[var]):
        raise TypeError(f"Colonne {var} n’est pas numérique")
    bins_sorted = sorted(set(float(b) for b in bins))
    if len(bins_sorted) < 2:
        raise ValueError("Au moins deux bornes nécessaires")

    # Utiliser les labels personnalisés au lieu des labels par défaut
    custom_labels = [cat_label0, cat_label1, cat_label2, cat_label3]

    # Ajouter les labels supplémentaires s'ils existent
    for i in range(4, 10):  # Support jusqu'à 10 catégories
        label_key = f"cat_label{i}"
        if label_key in cat_labels and cat_labels[label_key]:
            custom_labels.append(cat_labels[label_key])

    # S'assurer que nous avons suffisamment de labels pour le nombre de bins
    labels = custom_labels[:len(bins_sorted) - 1]

    df[f"{var}_cat"] = pd.cut(df[var], bins=bins_sorted, labels=labels, right=False)
    return df

# ──────────────────────────────────────────
#  Export CSV (réutilisable ou non)
# ──────────────────────────────────────────

def get_long_path_name(short_path: str) -> str:
    """
    Convertit un chemin DOS (8.3) en chemin complet lisible (Win32).
    Si l’appel échoue, on retourne le chemin tel quel (short ou absolu).
    """
    # 1) On normalise en chemin absolu
    full_buf = ctypes.create_unicode_buffer(260)
    get_full = ctypes.windll.kernel32.GetFullPathNameW
    ret_full = get_full(short_path, 260, full_buf, None)
    if ret_full == 0 or ret_full > 259:
        full_path = os.path.abspath(short_path)
    else:
        full_path = full_buf.value

    # 2) On tente de récupérer le nom long
    buf = ctypes.create_unicode_buffer(260)
    get_long = ctypes.windll.kernel32.GetLongPathNameW
    result = get_long(full_path, buf, 260)
    if result == 0 or result > 259:
        # Logging minimal pour debug
        err = ctypes.GetLastError()
        print(f"[get_long_path_name] échec Win32 0x{err:08X} sur : {full_path}", file=sys.stderr)
        return full_path
    # Si succès, buf.value contient déjà la version Unicode complète
    return buf.value

def export_cleaned_csv(df: pd.DataFrame, original_name: str, out_dir: str) -> str:
    """
    Enregistre df dans un CSV horodaté dans out_dir, puis renvoie le chemin Windows long.
    """
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = os.path.splitext(os.path.basename(original_name))[0]
    cleaned_filename = f"{base}_cleaned_{ts}.csv"
    cleaned_path = os.path.join(out_dir, cleaned_filename)

    # Écrire le CSV sur le disque
    df.to_csv(cleaned_path, index=False)

    # Chemin absolu (court) potentiel
    full_path = os.path.abspath(cleaned_path)

    # Tenter de récupérer le long path Windows
    try:
        long_path = get_long_path_name(full_path)
    except Exception:
        long_path = full_path

    return long_path

# ──────────────────────────────────────────
#  Fonction principale réutilisable
# ──────────────────────────────────────────

def process_dataframe(
    df: pd.DataFrame,
    nonnull: List[str],
    dup: str = "merge",
    agg: str | None = None,
    cat_var: str = "",
    cat_bins: str = "",
    cat_clean: str = "",
    buffer_radius: float = 0.0,
    balance_strategy: str = "",
    balance_sort_col: str = "",
    cat_label0: str = "g0",
    cat_label1: str = "g1",
    cat_label2: str = "g2",
    cat_label3: str = "g3",
    **additional_cat_labels
) -> Dict[str, Any]:
    """
    Transforme *df* et renvoie {operations, cleaned_df, debug}.
    Les étapes longues (gestion doublons) sont désormais en mode vectorisé.
    """

    debug: Dict[str, Any] = {}
    logs:  List[Dict[str, Any]] = []

    df = df.copy()
    df.rename(columns=lambda c: str(c).lstrip("\ufeff").strip(), inplace=True)

    start = time.time()
    # ── Colonnes non nulles ──────────────────────────────
    missing = [c for c in nonnull if c not in df.columns]
    if missing:
        raise KeyError(f"Colonnes obligatoires absentes : {', '.join(missing)}")
    nan_mask = df[nonnull].isna().any(axis=1)
    before_nan = len(df)
    df = df.loc[~nan_mask]
    logs.append({
        "name": f"Suppression NaN ({', '.join(nonnull)})",
        "before": before_nan,
        "after": len(df),
        "removed": before_nan - len(df),
        "pct_removed": round((before_nan - len(df)) * 100 / before_nan, 2) if before_nan else 0,
    })
    print(f"[TRACE] suppression NaN fait en {time.time() - start:.2f}s", file=sys.stderr)

    # ── Gestion des doublons ─────────────────────────────
    rules = parse_rules(agg_inline=agg)
    before_dup = len(df)
    if rules:
        # Harmonisation déclarative vectorisée
        df = harmoniser_doublons_vectorise(df, rules)
        label = "Harmonisation déclarative vectorisée + suppression doublons"
    else:
        # Pas de règles : on fait la fusion / suppression standard
        if dup == "merge":
            df = suppression_doublons_vectorisee(df)
            label = "Fusion doublons optimisée (merge)"
        else:
            df = df.loc[~df.duplicated(subset=["Titre"], keep="first")]
            label = "Suppression doublons (first)"
    logs.append({
        "name": label,
        "before": before_dup,
        "after": len(df),
        "removed": before_dup - len(df),
        "pct_removed": round((before_dup - len(df)) * 100 / before_dup, 2) if before_dup else 0,
    })
    print(f"[TRACE] Gestion des doublons fait en {time.time() - start:.2f}s", file=sys.stderr)

    # ── Boundary cleaning ───────────────────────────────
    if cat_clean:
        boundaries = [float(b) for b in cat_clean.split(",") if b.strip()]
        if cat_var not in df.columns:
            raise KeyError(f"--cat-var inconnu : {cat_var}")
        if df[cat_var].dtype == object:
            df[cat_var] = (df[cat_var].astype(str)
                                      .str.replace(",", ".", regex=False)
                                      .str.replace("%", "", regex=False))
        df[cat_var] = pd.to_numeric(df[cat_var], errors="coerce")
        mask_bc = df[cat_var].apply(lambda x: any((b - buffer_radius) <= x <= (b + buffer_radius) for b in boundaries))
        before_bc = len(df)
        df = df.loc[~mask_bc]
        logs.append({
            "name": f"Boundary cleaning ({cat_var})",
            "before": before_bc,
            "after": len(df),
            "removed": before_bc - len(df),
            "pct_removed": round((before_bc - len(df)) * 100 / before_bc, 2) if before_bc else 0,
            "details": {
                "boundaries": boundaries,
                "buffer_radius": buffer_radius,
            },
        })
        print(f"[TRACE] Boundary cleaning fait en {time.time() - start:.2f}s", file=sys.stderr)

    # ── Catégorisation ──────────────────────────────────
    if cat_var:
        bins_float = [float(b) for b in (cat_bins or "0,3,6,9,100").split(",") if b.strip()]
        avant = df.get(f"{cat_var}_cat", pd.Series(dtype="object")).notna().sum()
        df = ajouter_categorie(df, cat_var, bins_float, cat_label0, cat_label1, cat_label2, cat_label3, **additional_cat_labels)
        apres = df[f"{cat_var}_cat"].notna().sum()
        counts = df[f"{cat_var}_cat"].value_counts().sort_index().to_dict()
        logs.append({
            "name": f"Catégorisation ({cat_var})",
            "before": avant,
            "after": apres,
            "removed": 0,
            "pct_removed": 0,
            "details": {
                "bins": bins_float,
                "category_distribution": {str(k): int(v) for k, v in counts.items()}
            }
        })
        print(f"[TRACE] Catégorisation fait en {time.time() - start:.2f}s", file=sys.stderr)

    # ── Uniformisation des groupes ──────────────────────
    if balance_strategy and cat_var:
        cat_col = f"{cat_var}_cat"
        if cat_col not in df.columns:
            raise ValueError(f"La colonne '{cat_col}' n'existe pas")
        group_counts = df[cat_col].value_counts()
        min_group_size = group_counts.min()
        if balance_strategy == "undersample_random":
            df = df.groupby(cat_col, group_keys=False).apply(lambda g: g.sample(n=min_group_size, random_state=42))
        elif balance_strategy == "undersample_top":
            sort_col = balance_sort_col or "Impressions"
            if sort_col not in df.columns:
                raise KeyError(f"Colonne de tri manquante : {sort_col}")
            df = df.sort_values(sort_col, ascending=False)
            df = df.groupby(cat_col, group_keys=False).head(min_group_size)

        logs.append({
            "name": f"Uniformisation des groupes ({balance_strategy})",
            "before": int(group_counts.sum()),
            "after": len(df),
            "removed": int(group_counts.sum() - len(df)),
            "pct_removed": round((group_counts.sum() - len(df)) * 100 / group_counts.sum(), 2),
            "details": {
                "strategy": balance_strategy,
                "group_sizes_before": {str(k): int(v) for k, v in group_counts.items()},
                "group_rows_removed": {str(k): int(v - min_group_size) for k, v in group_counts.items()},
                "target_size": int(min_group_size),
                **({"sort_column": sort_col} if balance_strategy == "undersample_top" else {})
            }
        })
        print(f"[TRACE] Uniformisation des groupes fait en {time.time() - start:.2f}s", file=sys.stderr)

    debug["final_rows"] = len(df)
    return {"operations": logs, "cleaned_df": df, "debug": debug}

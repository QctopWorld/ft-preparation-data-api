# ─────────────────────────────────────────────────────────────────────────────
# 📦 Librairies standards
# ─────────────────────────────────────────────────────────────────────────────
import os
import contextlib
import sys
import json
import shutil
from collections import Counter
from pathlib import Path 
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# ─────────────────────────────────────────────────────────────────────────────
# 📦 Librairies tierces
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import openai
from openai import OpenAI

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    matthews_corrcoef,
    precision_recall_fscore_support,
    mean_squared_error
)



# ─── FONCTIONS ───────────────────────────────────────────────────────────────

def afficher_mse(y_true, y_pred, labels):
    """
    Calcule et affiche la Mean Squared Error (MSE) en interprétant les étiquettes comme des indices numériques.
    """
    y_true_enc = np.array([labels.index(t) for t in y_true])
    y_pred_enc = np.array([labels.index(p) for p in y_pred])
    mse = mean_squared_error(y_true_enc, y_pred_enc)
    print(f"\n=== MSE GLOBALE ===")
    print(f"Mean Squared Error (MSE) : {mse:.3f}")
    return mse

def afficher_mse_pondere_asym(y_true, y_pred, labels, W_mat):
    """
    Calcule et affiche une MSE pondérée selon la formule exacte :
      MSE = (1/N) * sum_{i,j} C[i][j] * W_mat[i][j] * (i - j)**2

    - C[i][j] est la matrice de confusion brute.
    - W_mat est une matrice de poids de forme (n_labels, n_labels).
    - N = somme des éléments de la matrice de confusion.
    """
    # 1) Matrice de confusion brute
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    N = cm.sum()
    if N == 0:
        print("Aucune prédiction à traiter pour la MSE pondérée.")
        return float('nan')

    # 2) Différences au carré
    i_idx, j_idx = np.indices(cm.shape)
    diff2 = (i_idx - j_idx) ** 2

    # 3) Calcul de la somme pondérée
    weighted = cm * W_mat * diff2
    mse_pond = weighted.sum() / N

    # 4) Affichage
    print("=== MSE PONDÉRÉE ASYMÉTRIQUE EXACTE ===")
    print(f"MSE pondérée : {mse_pond:.3f}")
    return mse_pond

def nom_rapport(model_id: str, job_id: str) -> str:
    # Vérifie que l'identifiant commence bien par "ft:"
    if not model_id.startswith("ft:"):
        raise ValueError("L'identifiant doit commencer par 'ft:'")

    # Supprime le préfixe spécifique
    
    prefixes_to_remove = [
    "ft:gpt-4.1-mini-2025-04-14:quebectop-inc:",
    "ft:gpt-4o-mini-2024-07-18:quebectop-inc:",
    "ft:gpt-4.1-mini-2025-04-14:quebectop-2007-inc:"
    ]

    res = model_id  # Valeur par défaut, au cas où aucun préfixe ne correspond

    for prefix in prefixes_to_remove:
        if model_id.startswith(prefix):
            res = model_id[len(prefix):]
            break  # On arrête à la première correspondance

    # Fusionne la dernière partie après le dernier ":" avec la partie précédente
    if ":" in res:
        parts = res.rsplit(":", 1)
        res = parts[0] + parts[1]

    # Appel API
    client = openai.OpenAI(api_key=openai.api_key)
    job = client.fine_tuning.jobs.retrieve(job_id)
    created_at = job.created_at

    # Format date
    from datetime import datetime
    date_creation = datetime.fromtimestamp(created_at).strftime('%m-%d-%Y')

    res = "eval_results_" + date_creation + "_" + res
    return res


def definition_label(fichier):
    LABELg = ["g0", "g1", "g2", "g3"]
    LABELn = ["0", "1", "2", "3"]
    LABELc = ["CTR_0_3", "CTR_3_6", "CTR_6_9", "CTR_9_plus"]
    
    with open(fichier, "r", encoding="utf-8") as file:
        for line in file:
            obj = json.loads(line)
            messages = obj.get("messages", [])
            
            for msg in messages:
                if msg.get("role") == "assistant":
                    if msg.get("content") in ["CTR_0_3", "CTR_3_6", "CTR_6_9", "CTR_9_plus"]:
                        return LABELc
                    elif msg.get("content") in ["g0", "g1", "g2", "g3"] :
                        return LABELg
                    else:
                        return LABELn
                                                
            break  # On sort après avoir traité la première ligne
            
def verifier_titres_communs(train_path: str, val_path: str):
    # Charger les titres du train pour role 'user'
    titres_train = []
    with open(train_path, 'r', encoding='utf-8') as file:
        for line in file:
            obj = json.loads(line)
            for msg in obj.get("messages", []):
                if msg.get("role") == "user":
                    titres_train.append(msg.get("content"))

    # Charger les titres de validation pour role 'user'
    titres_val = []
    with open(val_path, 'r', encoding='utf-8') as file:
        for line in file:
            obj = json.loads(line)
            for msg in obj.get("messages", []):
                if msg.get("role") == "user":
                    titres_val.append(msg.get("content"))

    # Déterminer les titres communs
    titres_communs = sorted(set(titres_val).intersection(titres_train))

    # Afficher les résultats
    #print(f"Titres communs (train vs validation): {len(titres_communs)} trouvés")
    #for titre in titres_communs:
    #    print(f"- {titre}")

    # Calculer la proportion
    total_val = len(titres_val)
    nombre_communs = len(titres_communs)
    proportion = (nombre_communs / total_val * 100) if total_val else 0

    print(f"\nTotal titres en validation : {total_val}")
    print(f"Nombre de titres communs avec le train : {nombre_communs}")
    print(f"Proportion de titres communs avec le train : {proportion:.2f}%")
    
    return proportion,titres_communs

def charger_jeu_de_test(fichier):
    test_messages, y_true, prompt_texts = [], [], []
    with open(fichier, "r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            msgs = obj["messages"]
            test_messages.append([m for m in msgs if m["role"] != "assistant"])
            y_true.append(msgs[-1]["content"].strip())
            prompt_texts.append(next(m["content"] for m in msgs if m["role"] == "user"))
    return test_messages, y_true, prompt_texts

def traiter_exemple(i, msgs, title, model_id: str, labels: list[str]):
    try:
        resp = openai.chat.completions.create(
            model=model_id,
            messages=msgs,
            temperature=0
        )
        pred = resp.choices[0].message.content.strip()
        if pred not in labels:
            print(f"\n⚠️  Label inattendu «{pred}» -> remplacé par 'unknown'")
            pred = "unknown"
        return i, pred, title
    except Exception as e:
        print(f"\n❌ Erreur à l’index {i} : {e}")
        return i, "unknown", title

def lancer_inference(test_messages, prompt_texts,
                     n_parallele: int,
                     model_id: str,
                     labels: list[str]):
    
    y_pred = [None] * len(test_messages)
    bar = tqdm(total=len(test_messages), unit="req", desc="Lancement en parallèle")
    with ThreadPoolExecutor(max_workers=n_parallele) as executor:
        futures = {
           executor.submit(traiter_exemple, i, msgs, title,
                           model_id, labels): i
            for i, (msgs, title) in enumerate(zip(test_messages, prompt_texts))
        }
        for future in as_completed(futures):
            i, pred, title = future.result()
            y_pred[i] = pred
            bar.set_description_str(f"{pred:<10} | {title[:60]}")
            bar.update(1)
    bar.close()
    return y_pred

def sauvegarder_resultats(prompt_texts, y_true, y_pred, fichier):
    pd.DataFrame({
        "prompt": prompt_texts,
        "true_label": y_true,
        "predicted_label": y_pred
    }).to_csv(fichier, index=False)
    print(f"\n✔ Résultats enregistrés dans {fichier}")

def afficher_matrices_et_stats(y_true, y_pred, labels):
    print("\nRapport de classification :")
    print(classification_report(y_true, y_pred, labels=labels, digits=3, zero_division=0))

    cm  = confusion_matrix(y_true, y_pred, labels=labels)
    cmn = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay(cm , display_labels=labels).plot(ax=axes[0], xticks_rotation=45)
    axes[0].set_title("Matrice de confusion (valeurs absolues)")
    ConfusionMatrixDisplay(cmn, display_labels=labels).plot(ax=axes[1], xticks_rotation=45, cmap="Blues")
    axes[1].set_title("Matrice de confusion (rappel normalisé)")
    plt.tight_layout()
    plt.show()

def afficher_performance_finetuning(job_id):
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id)
    for event in response.data:
        if event.type == "metrics" and "full_valid_loss" in event.data:
            print("Train loss:", round(event.data.get("train_loss", 0), 2))
            print("Validation loss:", round(event.data.get("valid_loss", 0), 2))
            print("Full validation loss:", round(event.data.get("full_valid_loss", 0), 2))
            break

def charger_donnees(depuis_csv):
    df = pd.read_csv(depuis_csv)
    y_true = df["true_label"].tolist()
    y_pred = df["predicted_label"].tolist()
    prompts = df["prompt"].tolist()
    return y_true, y_pred, prompts

def detecter_labels(y_true, y_pred):
    return sorted(set(y_true + y_pred))

def afficher_repartition_classes(y_true, labels):
    dist = Counter(y_true)
    total = sum(dist.values())
    print("\n=== RÉPARTITION DES CLASSES DANS LE JEU DE TEST ===")
    print("lbl        |    n   |  pct")
    for lbl in labels:
        n = dist.get(lbl, 0)
        pct = 100 * n / total if total else 0
        print(f"{lbl:>10} | {n:>6} | {pct:5.1f}%")
    print(f"TOTAL: {total}\n")

def calculer_metriques_globales(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=labels, digits=3,
        zero_division=0, output_dict=True
    )
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    return acc, bal_acc, report, mcc, kappa

def afficher_metriques_par_classe(y_true, y_pred, labels):
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    print("=== MÉTRIQUES PAR CLASSE ===")
    print(" lbl        | prec   | rec    | f1     | support")
    for lbl, p, r, f, s in zip(labels, prec, rec, f1, sup):
        print(f"{lbl:>10} | {p:6.3f} | {r:6.3f} | {f:6.3f} | {int(s):8}")

def afficher_matrices_confusion(y_true, y_pred, labels):
    def _joli_affichage(mat, titre):
        df_cm = pd.DataFrame(
            mat,
            index=[f"Réel   {l}" for l in labels],
            columns=[f"Prédit {l}" for l in labels]
        )
        print(f"\n--- {titre} ---")
        print(df_cm.round(3).to_string())

    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
    cm_rec = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    cm_prec = confusion_matrix(y_true, y_pred, labels=labels, normalize="pred")

    _joli_affichage(cm_raw, "Matrice brute (comptes)")
    _joli_affichage(cm_rec, "Matrice normalisée par ligne (rappel)")
    _joli_affichage(cm_prec, "Matrice normalisée par colonne (précision)")

def afficher_top_erreurs(y_true, y_pred, prompts, labels):
    # Calculer les erreurs avec leur distance d'index
    err_pairs = [
        (abs(labels.index(t) - labels.index(p)), t, p, txt)
        for t, p, txt in zip(y_true, y_pred, prompts)
        if t != p and t in labels and p in labels
    ]

    if not err_pairs:
        print("Aucune erreur à afficher.")
        return

    # Tri décroissant pour trouver la distance maximale
    err_pairs.sort(reverse=True)
    max_dist = err_pairs[0][0]

    # Ne garder que les erreurs ayant cette distance maximale
    erreurs_max = [e for e in err_pairs if e[0] == max_dist]
    erreurs_grandes = [e for e in err_pairs if e[0] == (max_dist - 1)]

    print(f"\n=== {len(erreurs_max)} ERREURS LES PLUS « EXTRÊMES » (distance d'index = 3) ===")
    for _, t, p, txt in erreurs_max:
        #snippet = txt[:80].replace("\n", " ")
        print(f"Réel : {t} → Prédit :{p} | {txt}")
        
    print(f"\n=== {len(erreurs_grandes) - 1} ERREURS « IMPORTANTES » (distance d'index = 2) ===")
    for _, t, p, txt in erreurs_grandes:
        #snippet = txt[:80].replace("\n", " ")
        print(f"Réel : {t} → Prédit :{p} | {txt}")

def executer_analyse(chemin_csv):
    y_true, y_pred, prompts = charger_donnees(chemin_csv)
    labels = detecter_labels(y_true, y_pred)
    print(f"→ LABELS détectés : {labels}")
    afficher_repartition_classes(y_true, labels)
    acc, bal_acc, report, mcc, kappa = calculer_metriques_globales(y_true, y_pred, labels)

    print("=== RÉSUMÉ GLOBAL ===")
    print(f"Accuracy              : {acc:.3%}")
    print(f"Balanced accuracy     : {bal_acc:.3%}")
    print(f"Macro F-score         : {report['macro avg']['f1-score']:.3f}")
    print(f"Weighted F-score      : {report['weighted avg']['f1-score']:.3f}")
    print(f"Matthews corrcoef     : {mcc:.3f}")
    print(f"Cohen kappa           : {kappa:.3f}\n")

    afficher_metriques_par_classe(y_true, y_pred, labels)
    afficher_matrices_confusion(y_true, y_pred, labels)
    afficher_top_erreurs(y_true, y_pred, prompts, labels)

def generer_rapport(
    y_true, y_pred,
    input_train: Path,        # ← Paths explicites
    input_val:   Path,
    job_id: str, model_id: str, train_file_id: str, val_file_id: str,
    weight_up: float, weight_down: float,
    output_txt: Path, output_csv: Path,
    labels: list[str]         # ← plus simple de passer déjà détecté
    
):
    labels = definition_label(input_val)

    with output_txt.open("w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f):
            print("Nom du rapport :", output_csv.stem)  # ex: eval_results_...
            print("\nID du job OpenAI :", job_id)
            print("ID du modèle OpenAI :", model_id)
            print("ID du fichier d'entraînement :", train_file_id)
            print("ID du fichier de validation :", val_file_id)

            prop, titres_communs = verifier_titres_communs(input_train, input_val)

            print("\nMETRICS MODEL")
            afficher_performance_finetuning(job_id)

            afficher_mse(y_true, y_pred, labels)

            # pondération
            n = len(labels)
            W = np.zeros((n, n))
            W[np.triu_indices(n, 1)] = weight_up   # au-dessus de la diagonale
            W[np.tril_indices(n, -1)] = weight_down

            afficher_mse_pondere_asym(y_true, y_pred, labels, W)
            afficher_matrices_et_stats(y_true, y_pred, labels)
            executer_analyse(output_csv)

def download_openai_file(file_id: str, api_key: str) -> bytes:
    
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"https://api.openai.com/v1/files/{file_id}/content"
    resp = requests.get(url, headers=headers, timeout=90)
    resp.raise_for_status()
    return resp.content

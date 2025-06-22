# app/reports.py  ────────────────────────────────────────────────
"""
Génère un rapport complet d'évaluation fine-tune
→ Renvoie un .zip encodé base64 contenant :
   - notebook renommé
   - train.jsonl / validation.jsonl
   - prédictions.csv
   - rapport.txt
"""
import io, os, shutil, base64, tempfile, json
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────────────────────
# 📦 Librairies standards
# ─────────────────────────────────────────────────────────────────────────────
import os
import contextlib
import sys
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────────────────────
# 📦 Librairies tierces
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
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
    precision_recall_fscore_support
)

# ─────────────── repliquez ici (ou importez) toutes vos fonctions utilitaires :
# ─── FONCTIONS ───────────────────────────────────────────────────────────────
def nom_rapport(model_id, job_id):
    # Vérifie que l'identifiant commence bien par "ft:"
    if not model_id.startswith("ft:"):
        raise ValueError("L'identifiant doit commencer par 'ft:'")

    # Supprime le préfixe spécifique
    
    prefixes_to_remove = [
    "ft:gpt-4.1-mini-2025-04-14:quebectop-inc:",
    "ft:gpt-4o-mini-2024-07-18:quebectop-inc:"
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

def traiter_exemple(i, msgs, title):
    try:
        resp = openai.chat.completions.create(
            model=MODEL_ID,
            messages=msgs,
            temperature=0
        )
        pred = resp.choices[0].message.content.strip()
        if pred not in LABELS:
            print(f"\n⚠️  Label inattendu «{pred}» -> remplacé par 'unknown'")
            pred = "unknown"
        return i, pred, title
    except Exception as e:
        print(f"\n❌ Erreur à l’index {i} : {e}")
        return i, "unknown", title

def lancer_inference(test_messages, prompt_texts,n_parallele):
    y_pred = [None] * len(test_messages)
    bar = tqdm(total=len(test_messages), unit="req", desc="Lancement en parallèle")
    with ThreadPoolExecutor(max_workers=n_parallele) as executor:
        futures = {
            executor.submit(traiter_exemple, i, msgs, title): i
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

def afficher_matrices_et_stats(y_true, y_pred):
    print("\nRapport de classification :")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0))

    cm  = confusion_matrix(y_true, y_pred, labels=LABELS)
    cmn = confusion_matrix(y_true, y_pred, labels=LABELS, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay(cm , display_labels=LABELS).plot(ax=axes[0], xticks_rotation=45)
    axes[0].set_title("Matrice de confusion (valeurs absolues)")
    ConfusionMatrixDisplay(cmn, display_labels=LABELS).plot(ax=axes[1], xticks_rotation=45, cmap="Blues")
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
          
          
def renommer_fichier_notebook(model_id,job_id):
    ancien_nom = "eval_results_CANEVA1.ipynb"
    nouveau_nom = nom_rapport(MODEL_ID,JOB_ID)+".ipynb"
    
    # Renommage du fichier
    os.rename(ancien_nom, nouveau_nom)
          
          
def get_notebook_path():
    """Retourne le chemin absolu du notebook en cours."""
    connection_file = os.path.basename(ipykernel.get_connection_file())
    kernel_id = connection_file.split('-', 1)[1].split('.')[0]

    for srv in notebookapp.list_running_servers():
        try:
            response = requests.get(f"{srv['url']}api/sessions", params={'token': srv.get('token', '')})
            for sess in json.loads(response.text):
                if sess['kernel']['id'] == kernel_id:
                    return os.path.join(srv['notebook_dir'], sess['notebook']['path'])
        except Exception as e:
            pass
    return None
                                    
def deplacement_fichiers():
    ancien_chemin = get_notebook_path()

    if ancien_chemin:
        dossier_parent, ancien_nom_notebook = os.path.split(ancien_chemin)
        nouveau_nom_notebook = nom_rapport(MODEL_ID,JOB_ID)+".ipynb"
        nouveau_chemin = os.path.join(dossier_parent, nouveau_nom_notebook)

        # Renommer le fichier
        os.rename(ancien_chemin, nouveau_chemin)
        #print(f"Notebook renommé : {ancien_nom_notebook} -> {nouveau_nom_notebook}")

        # Créer le dossier avec le nom (sans l'extension)
        nom_dossier = os.path.splitext(nouveau_nom_notebook)[0]
        chemin_dossier = os.path.join(dossier_parent, nom_dossier)
        os.makedirs(chemin_dossier, exist_ok=True)
                                    
        #Renommer les noms des fichiers train et test
        os.rename(INPUT_FILE_TRAIN, "train.jsonl")
        os.rename(INPUT_FILE_VALIDATION, "validation.jsonl")

        # Déplacement des fichiers dans le nouveau dossier
        fichiers = [nouveau_nom_notebook,"train.jsonl","validation.jsonl",OUTPUT_CSV,OUTPUT_TXT]
        
        print(f"\n")
        for fichier in fichiers:
            chemin_source = os.path.join(dossier_parent, fichier)
            chemin_final = os.path.join(chemin_dossier, fichier)

            if os.path.exists(chemin_source):
                shutil.move(chemin_source, chemin_final)
                print(f"✅ Fichier {fichier} bien déplacé vers → {chemin_dossier}/")
            else:
                print(f"⚠️ Fichier {fichier} introuvable dans {chemin_source}")
                pass

    else:
        print("❌ Impossible de localiser le notebook actuel.")
                  
# ------------------------------------------------------------------------------

TMP_PREFIX = "ft_report_"

def download_jsonl(client: openai.OpenAI, file_id: str, dst_path: str) -> None:
    content = client.files.content(file_id).text
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(content)

def make_zip(folder: str) -> bytes:
    zip_path = shutil.make_archive(folder, "zip", folder)
    with open(zip_path, "rb") as fh:
        return fh.read()

def generate_report(
    *,
    job_id: str,
    model_id: str,
    train_file_id: str,
    val_file_id: str,
    n_threads: int = 20
) -> bytes:
    client = openai.OpenAI()

    # 1. Préparer un dossier temporaire unique
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    work_dir = tempfile.mkdtemp(prefix=TMP_PREFIX + ts + "_")
    train_path = os.path.join(work_dir, "train.jsonl")
    val_path   = os.path.join(work_dir, "validation.jsonl")

    # 2. Télécharger les jeux JSONL depuis OpenAI
    download_jsonl(client, train_file_id, train_path)
    download_jsonl(client, val_file_id,   val_path)

    # 3. Lancer votre pipeline d’évaluation
    global MODEL_ID, JOB_ID, INPUT_FILE_TRAIN, INPUT_FILE_VALIDATION
    MODEL_ID, JOB_ID = model_id, job_id
    INPUT_FILE_TRAIN, INPUT_FILE_VALIDATION = train_path, val_path

    N0M_RAPPORT = nom_rapport(model_id, job_id)
    csv_out = os.path.join(work_dir, N0M_RAPPORT + ".csv")
    txt_out = os.path.join(work_dir, N0M_RAPPORT + ".txt")
    LABELS  = definition_label(val_path)

    # Exécution (simplifiée) – adaptez à vos fonctions :
    test_messages, y_true, prompts = charger_jeu_de_test(val_path)
    y_pred = lancer_inference(test_messages, prompts, n_threads)
    sauvegarder_resultats(prompts, y_true, y_pred, csv_out)

    # Capture du rapport TXT
    with open(txt_out, "w", encoding="utf-8") as f:
        # redirigez prints dans f (ou construisez un str)
        afficher_performance_finetuning(job_id, file=f)
        executer_analyse(csv_out, file=f)

    # 4. Créer le ZIP et renvoyer les octets
    zip_bytes = make_zip(work_dir)

    # Nettoyage (facultatif) : shutil.rmtree(work_dir)
    return zip_bytes


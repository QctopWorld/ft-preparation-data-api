import os, tempfile, base64, contextlib, json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import matplotlib

from app.settings import Settings
matplotlib.use("Agg")                      # backend non-GUI

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
styles = getSampleStyleSheet()

from .reports_function import * 


OPENAI_API_KEY = "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"  # last resort


# ------------------------------------------------------------------ #
#  Pipeline ré-emballé dans une fonction                              #
# ------------------------------------------------------------------ #
def run_full_pipeline(
    train_file: Path, val_file: Path,
    job_id: str, model_id: str, train_file_id: str,val_file_id: str,  # pour le rapport
    n_threads: int, weight_up: float, weight_down: float
) -> Path:

    nom      = nom_rapport(model_id, job_id)
    work_dir = train_file.parent
    txt      = work_dir / f"{nom}.txt"
    csv      = work_dir / f"{nom}.csv"

    # 1. Jeu de test + labels
    test_msgs, y_true, prompts = charger_jeu_de_test(val_file)
    labels = detecter_labels(y_true, y_true)  # on en déduira après inférence

    y_pred = lancer_inference(
        test_msgs, prompts,
        n_threads, model_id, labels
    )

    sauvegarder_resultats(prompts, y_true, y_pred, csv)

    # Génération du rapport TXT
    generer_rapport(
        y_true, y_pred,
        input_train=train_file,
        input_val=val_file,
        job_id=job_id, model_id=model_id, train_file_id=train_file_id, val_file_id=val_file_id,
        weight_up=weight_up, weight_down=weight_down,
        output_txt=txt, output_csv=csv,
        labels=labels
    )

    # On retourne directement le fichier TXT
    return txt

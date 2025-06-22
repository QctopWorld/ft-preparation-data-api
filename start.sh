#!/usr/bin/env bash
# active le venv situé à la racine du projet
source "$(dirname "$0")/venv/bin/activate"  # commenté pour éviter les erreurs de chemin si render est utilisé
# puis lance Uvicorn
exec uvicorn app:app \
     --host 127.0.0.1 \
     --port 9090 \
     --reload
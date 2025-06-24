# Fine‑Tuning CSV API (FastAPI)

Service HTTP qui reprend la logique du script `process_csv.py` :
– nettoyage NaN, gestion/fusion des doublons, *boundary cleaning*,
catégorisation numérique, undersampling des groupes, export CSV
nettoyé – et renvoie un rapport détaillé en JSON.

> **Swagger UI** : <http://127.0.0.1:9000/docs>  
> **OpenAPI JSON** : <http://127.0.0.1:9000/openapi.json>

---

## Sommaire

1. [Fonctionnalités](#fonctionnalités)  
2. [Prérequis](#prérequis)  
3. [Installation](#installation)  
4. [Activation du virtualenv](#activation-du-virtualenv)  
5. [Lancement du serveur](#lancement-du-serveur)  
6. [Exemple d’appel](#exemple-dappel)  
7. [Variables d’environnement utiles](#variables-denvironnement-utiles)  
8. [Licence](#licence)

---

## Fonctionnalités

| Étape | Description |
|-------|-------------|
| **Upload CSV** | Envoi en *multipart/form‑data* (`UploadFile`) |
| **Suppression NaN** | Colonnes obligatoires à renseigner (`nonnull`) |
| **Dedup** | `dup=merge` (legacy) ou `dup=first`, ou règles déclaratives (`agg`) |
| **Boundary cleaning** | Suppression des lignes dans un buffer ±`radius` autour des bornes |
| **Catégorisation** | `cat_var`, `cat_bins`, étiquette `g0…gn` |
| **Balancing** | `balance_strategy=undersample_random|undersample_top` |
| **Export** | Renvoie le chemin du CSV nettoyé et un rapport JSON (`operations[]`) |

---

## Prérequis

* Python 3.9 + (testé jusqu’à 3.12)  
* `git` (facultatif)  
* Accès réseau local au port 9000 (modifiable)

---

## Installation

```bash
git clone https://github.com/<vous>/ft-preparation-donnees.git
cd ft-preparation-donnees
python -m venv venv           # crée un environnement virtuel

# Windows (avec le venv activé)
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt  -v
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

#Linux
source venv/bin/activate

#Lancer le serveur
uvicorn app.api:app --reload --port 9090

# macOS / Linux (venv activé également)
pip install -r requirements.txt


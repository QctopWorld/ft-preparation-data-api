# api.py – Service FastAPI qui enveloppe la logique de core.py
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
import io, traceback, datetime, json
import pandas as pd
import base64

from .core import process_dataframe, json_serial
from .splits import (
    random_split,
    stratified_split_univariate,
    stratified_split_multivariate,
    temporal_split,
    group_aware_split
)

app = FastAPI(title="Fine-Tuning CSV API", version="1.0")

@app.post("/process")
async def process_csv(
    file: UploadFile,
    nonnull: str         = Form(...),
    dup: str             = Form("merge"),
    agg: str             = Form(""),
    cat_var: str         = Form(""),
    cat_bins: str        = Form(""),
    cat_clean: str       = Form(""),
    buffer_radius: float = Form(0.0),
    balance_strategy: str = Form(""),
    balance_sort_col: str = Form(""),
):
    try:
        print("/process est appelé")
        # Lecture du CSV en DataFrame
        df = pd.read_csv(io.BytesIO(await file.read()))

        # Traitement métier
        result = process_dataframe(
            df,
            nonnull.split(","),
            dup, agg,
            cat_var, cat_bins,
            cat_clean, buffer_radius,
            balance_strategy, balance_sort_col,
        )

        # 1) Générer le CSV nettoyé DANS UN BUFFER MÉMOIRE
        buffer = io.StringIO()
        result["cleaned_df"].to_csv(buffer, index=False)
        buffer.seek(0)
        csv_bytes = buffer.getvalue().encode("utf-8")

        # 2) Encoder en base64
        csv_b64 = base64.b64encode(csv_bytes).decode("ascii")

        # 3) Construire la réponse JSON : 
        #    on conserve "operations" et "debug", et on ajoute "csv_base64"
        response_content = {
            "operations": result["operations"],
            "debug":      result["debug"],
            "csv_base64": csv_b64
        }

        return JSONResponse(
            content=json.loads(json.dumps(response_content, default=json_serial, ensure_ascii=False)),
            media_type="application/json"
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": str(e),
                "trace": traceback.format_exc(),
            },
        )


@app.post("/split")
async def split_csv(
    file: UploadFile,
    strategy: str               = Form(...),           # "random", "strat_uni", "strat_multi", "temporal", "group"
    test_frac: Optional[float]  = Form(0.2),           # fraction pour les splits qui le demandent
    random_state: Optional[int] = Form(42),            # seed pour l’aléatoire
    label_col: Optional[str]    = Form(None),          # pour strat_uni
    strat_cols: Optional[str]   = Form(None),          # pour strat_multi, passé en CSV ("col1,col2,col3")
    date_col: Optional[str]     = Form(None),          # pour temporal
    cutoff: Optional[str]       = Form(None),          # pour temporal (format YYYY-MM-DD)
    date_format: Optional[str]  = Form(None),          # format pour parser cutoff ou date_col
    group_col: Optional[str]    = Form(None),          # pour group-aware
):

    try:
        print("/split est appelé")
        raw = await file.read()
        df  = pd.read_csv(io.BytesIO(raw))

        # Choix de la stratégie
        strat = strategy.strip().lower()
        if strat == "random":
            train_df, test_df = random_split(df, test_frac=test_frac, random_state=random_state)

        elif strat == "strat_uni":
            if not label_col:
                raise HTTPException(status_code=400, detail="label_col requis pour strat_uni")
            train_df, test_df = stratified_split_univariate(
                df, label_col=label_col,
                test_frac=test_frac, random_state=random_state
            )

        elif strat == "strat_multi":
            if not strat_cols:
                raise HTTPException(status_code=400, detail="strat_cols requis pour strat_multi")
            cols_list = [c.strip() for c in strat_cols.split(",") if c.strip()]
            train_df, test_df = stratified_split_multivariate(
                df, strat_cols=cols_list,
                test_frac=test_frac, random_state=random_state
            )

        elif strat == "temporal":
            if not date_col:
                raise HTTPException(status_code=400, detail="date_col requis pour temporal")
            # Soit cutoff fourni, soit on se base sur test_frac
            train_df, test_df = temporal_split(
                df, date_col=date_col,
                cutoff=cutoff, test_frac=test_frac,
                date_format=date_format
            )

        elif strat == "group":
            if not group_col:
                raise HTTPException(status_code=400, detail="group_col requis pour group")
            train_df, test_df = group_aware_split(
                df, group_col=group_col,
                test_frac=test_frac, random_state=random_state
            )

        else:
            raise HTTPException(status_code=400, detail=f"strategy inconnue : {strategy}")

        # Encoder les deux DataFrames en base64 CSV
        def df_to_b64(_df: pd.DataFrame) -> str:
            buf = io.StringIO()
            _df.to_csv(buf, index=False)
            return base64.b64encode(buf.getvalue().encode("utf-8")).decode("ascii")

        response_content = {
            "train_csv_base64": df_to_b64(train_df),
            "validation_csv_base64":  df_to_b64(test_df),
            "train_rows":       len(train_df),
            "validation_rows":        len(test_df),
        }

        return JSONResponse(
            content=json.loads(json.dumps(response_content, default=json_serial, ensure_ascii=False)),
            media_type="application/json"
        )

    except HTTPException:
        # déjà levé plus haut
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"error": str(e), "trace": traceback.format_exc()},
        )
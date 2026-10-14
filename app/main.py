import io

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from .classifier import classify_dataframe
from .presets import PRESETS

load_dotenv()

app = FastAPI(title="CSV Classifier", description="Classify CSV rows with OpenAI and add the result as a new column.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/presets")
def presets():
    return PRESETS


@app.post("/classify")
async def classify(
    file: UploadFile = File(..., description="CSV file to classify"),
    preset: str | None = Form(None, description=f"One of: {', '.join(PRESETS)}. Overrides labels/instruction/output_column defaults."),
    labels: str | None = Form(None, description="Comma-separated allowed labels, e.g. 'spam,not_spam'"),
    instruction: str | None = Form(None, description="What the classification is about"),
    output_column: str | None = Form(None, description="Name of the new column"),
    input_columns: str | None = Form(None, description="Comma-separated columns to show the model (default: all)"),
    model: str | None = Form(None),
):
    cfg = {}
    if preset:
        if preset not in PRESETS:
            raise HTTPException(400, f"Unknown preset '{preset}'. Options: {list(PRESETS)}")
        cfg = dict(PRESETS[preset])
    if labels:
        cfg["labels"] = [x.strip() for x in labels.split(",") if x.strip()]
    if instruction:
        cfg["instruction"] = instruction
    if output_column:
        cfg["output_column"] = output_column
    if "labels" not in cfg:
        raise HTTPException(400, "Provide 'labels' or a 'preset'")
    cfg.setdefault("instruction", "Classify the record.")
    cfg.setdefault("output_column", "classification")

    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    cols = [c.strip() for c in input_columns.split(",")] if input_columns else None
    try:
        out = classify_dataframe(df, input_columns=cols, model=model, **cfg)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    name = (file.filename or "data.csv").rsplit(".", 1)[0]
    return Response(
        out.to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}_classified.csv"'},
    )

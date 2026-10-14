import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd

DEFAULT_MODEL = "gpt-6-luna"  # only model the Decisions API supports (per OpenAI docs)
DECISIONS_URL = "https://api.openai.com/v1/decisions"


def get_client() -> httpx.Client:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set (put it in .env)")
    return httpx.Client(
        headers={"Authorization": f"Bearer {key}"},
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        timeout=60,
    )


def _row_to_text(row: pd.Series, columns: list[str]) -> str:
    return "\n".join(f"{c}: {row[c]}" for c in columns)


def decide(client: httpx.Client, text: str, labels: list[str], instruction: str, model: str) -> tuple[str, float | None]:
    """One call to POST /v1/decisions with a single 'choice' question. Returns (label, confidence)."""
    body = {
        "model": model,
        "input": text,
        "questions": [
            {
                "type": "choice",
                "name": "classification",
                "instructions": instruction,
                "choices": [{"value": l, "description": l} for l in labels],
            }
        ],
    }
    for attempt in range(4):
        r = client.post("/decisions", json=body)
        if r.status_code in (429, 500, 502, 503) and attempt < 3:
            time.sleep(2**attempt)
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        break
    answer = r.json()["answers"][0]
    if answer.get("type") == "refusal":
        return "REFUSED", None
    return answer["choice"], answer.get("confidence")


def classify_dataframe(
    df: pd.DataFrame,
    labels: list[str],
    instruction: str,
    output_column: str = "classification",
    input_columns: list[str] | None = None,
    model: str | None = None,
    max_workers: int = 8,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Return a copy of df with `output_column` and `<output_column>_confidence` appended.
    A failed row gets 'ERROR: ...' instead of aborting the whole file."""
    input_columns = input_columns or list(df.columns)
    missing = [c for c in input_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in CSV: {missing}")
    conf_column = f"{output_column}_confidence"
    for c in (output_column, conf_column):
        if c in df.columns:
            raise ValueError(f"Output column '{c}' already exists in CSV")
    if len(labels) < 2:
        raise ValueError("Provide at least two labels")

    client = client or get_client()
    model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    def work(row: pd.Series):
        try:
            return decide(client, _row_to_text(row, input_columns), labels, instruction, model)
        except Exception as e:
            return f"ERROR: {e}"[:200], None

    rows = [r for _, r in df.iterrows()]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(work, rows))

    out = df.copy()
    out[output_column] = [r[0] for r in results]
    out[conf_column] = [r[1] for r in results]
    return out

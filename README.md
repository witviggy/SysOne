# CSV Classifier (OpenAI)

Classifies each CSV row with the OpenAI **Decisions API** (`POST /v1/decisions`, "choice" question, model `gpt-6-luna`) and appends the result as a new column, plus a `<column>_confidence` column. Runs entirely in Docker; nothing is installed locally.

## Setup
```
copy .env.example .env      # then put your OPENAI_API_KEY in .env
docker compose up --build
```

## Use the API
Open http://localhost:8000/docs, use `POST /classify`, upload a file from `sample_data/`, and set `preset` (e.g. `support_tickets`), or give your own `labels`, `instruction`, `output_column`. The response is the CSV with the new column.

```
curl -F file=@sample_data/reviews.csv -F preset=reviews http://localhost:8000/classify -o reviews_classified.csv
```

## Use the CLI (in the container)
```
docker compose run --rm api python cli.py sample_data/transactions.csv --preset transactions -o output/transactions_classified.csv
docker compose run --rm api python cli.py sample_data/reviews.csv --labels "positive,negative" --instruction "Sentiment?" --output-column mood -o output/r.csv
```
Results appear in `./output`.

## Samples
| File | Preset | New column |
|---|---|---|
| sample_data/support_tickets.csv | support_tickets | category |
| sample_data/reviews.csv | reviews | sentiment |
| sample_data/transactions.csv | transactions | spend_category |

## Tests (no API key needed, OpenAI is mocked)
```
docker compose run --rm api python -m pytest -q
```
Model defaults to `gpt-6-luna` (currently the only one the Decisions API supports); override with `OPENAI_MODEL` in `.env` or the `model` field.

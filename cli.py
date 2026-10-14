"""Classify a CSV from the command line.

Examples:
  python cli.py sample_data/support_tickets.csv --preset support_tickets
  python cli.py data.csv --labels "spam,not_spam" --instruction "Is this spam?" --output-column is_spam
"""
import argparse
import sys

import pandas as pd
from dotenv import load_dotenv

from app.classifier import classify_dataframe
from app.presets import PRESETS

load_dotenv()

p = argparse.ArgumentParser()
p.add_argument("csv")
p.add_argument("--preset", choices=list(PRESETS))
p.add_argument("--labels", help="comma-separated labels")
p.add_argument("--instruction")
p.add_argument("--output-column")
p.add_argument("--input-columns", help="comma-separated columns to send to the model")
p.add_argument("--model")
p.add_argument("-o", "--output", help="default: <input>_classified.csv")
a = p.parse_args()

cfg = dict(PRESETS.get(a.preset, {}))
if a.labels:
    cfg["labels"] = [x.strip() for x in a.labels.split(",")]
if a.instruction:
    cfg["instruction"] = a.instruction
if a.output_column:
    cfg["output_column"] = a.output_column
if "labels" not in cfg:
    sys.exit("Provide --labels or --preset")
cfg.setdefault("instruction", "Classify the record.")

df = pd.read_csv(a.csv)
cols = a.input_columns.split(",") if a.input_columns else None
out = classify_dataframe(df, input_columns=cols, model=a.model, **cfg)
dest = a.output or a.csv.rsplit(".", 1)[0] + "_classified.csv"
out.to_csv(dest, index=False)
print(f"Wrote {len(out)} rows to {dest}")

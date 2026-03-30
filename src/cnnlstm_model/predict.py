import argparse
import pandas as pd
from joblib import load

from .data import load_csv, save_csv


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/model.joblib")
    ap.add_argument("--input", required=True, help="CSV with feature columns")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--id-col", default="id", help="ID column for submission output")
    return ap.parse_args()


def main():
    args = parse_args()

    model = load(args.model)
    X = load_csv(args.input)

    proba = model.predict_proba(X)[:, 1]

    if args.id_col in X.columns:
        out = pd.DataFrame({args.id_col: X[args.id_col], "click": proba})
    else:
        out = pd.DataFrame({"click": proba})

    save_csv(out, args.output)
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

from .data import load_csv
from .model import train_eval_save

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.gz"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=str(DEFAULT_DATA_PATH),
        help="Path to training CSV",
    )

    ap.add_argument(
        "--valcsv",
        default=str(DEFAULT_DATA_PATH),
        help="Path to validation CSV",
    )
    
    ap.add_argument(
        "--model-out",
        default=str(DEFAULT_MODEL_PATH),
        help="Saved model path",
    )
    ap.add_argument("--test-size", type=float, default=0.2, help="Validation fraction")
    ap.add_argument("--random-state", type=int, default=42)
    
    return ap.parse_args()


def main():
    args = parse_args()

    train_ds = HighwayVideoClips(ROOT, T=T, size=SIZE, train=True)
    val_ds   = HighwayVideoClips(ROOT, T=T, size=SIZE, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    metrics = train_eval_save(
        df=df,
        label=args.label,
        model_path=model_path,
        random_state=args.random_state,
        test_size=args.test_size,
    )

    print(f"Saved model to: {model_path}")
    
if __name__ == "__main__":
    main()


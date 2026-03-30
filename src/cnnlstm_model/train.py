import argparse
from pathlib import Path

from .data import load_csv
from .model import train_eval_save

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.gz"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward(); opt.step()
        loss_sum += float(loss) * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, *, return_cm: bool=False, num_classes: int=None):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    cm = None

    for b, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        loss_sum += float(loss) * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)

        if return_cm:
            # lazily init CM using first batch's num_classes if not provided
            if cm is None:
                K = num_classes if num_classes is not None else logits.size(1)
                cm = torch.zeros(K, K, dtype=torch.long, device='cpu')
            # update confusion matrix on CPU
            for t, p in zip(y.view(-1).cpu(), pred.view(-1).cpu()):
                cm[t.long(), p.long()] += 1

    avg_loss = loss_sum / total
    acc = correct / total
    if return_cm:
        return avg_loss, acc, cm.numpy()
    return avg_loss, acc


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=str(DEFAULT_DATA_PATH),
        help="Path to training CSV",
    )
    ap.add_argument("--label", default="click", help="Target column")
    ap.add_argument(
        "--model-out",
        default=str(DEFAULT_MODEL_PATH),
        help="Saved model path",
    )
    ap.add_argument("--test-size", type=float, default=0.2, help="Validation fraction")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--nrows", type=int, default=200000, help="Rows to load for training")
    ap.add_argument(
        "--model",
        choices=["logreg", "xgb"],
        default="logreg",
        help="Which model to train.",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    model_path = Path(args.model_out)

    df = load_csv(csv_path, nrows=args.nrows)

    if args.label not in df.columns:
        raise ValueError(f"Label column '{args.label}' not found in {csv_path}")

    metrics = train_eval_save(
        df=df,
        label=args.label,
        model_path=model_path,
        random_state=args.random_state,
        test_size=args.test_size,
    )

    print(f"Saved model to: {model_path}")
    print(f"log_loss={metrics['log_loss']:.6f}")
    if "auc" in metrics:
        print(f"auc={metrics['auc']:.6f}")


if __name__ == "__main__":
    main()


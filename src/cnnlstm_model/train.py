from .data import load_csv
from .model import train_eval_save
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader

from cnnlstm_model.data import HighwayVideoClips
from cnnlstm_model.model import Simple3DCNN, train_one_epoch, evaluate



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-out", default="models/model.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=112)
    
    return ap.parse_args()


def main():
    args = parse_args()

    train_ds = HighwayVideoClips(ROOT, T=T, size=SIZE, train=True)
    val_ds   = HighwayVideoClips(ROOT, T=T, size=SIZE, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Simple3DCNN(num_classes=2).to(device)

    metrics = train_eval_save(
        df=df,
        label=args.label,
        epochs = args.epochs,
        model_path=model_path,
        random_state=args.random_state,
        test_size=args.test_size,
    )

    print(f"Saved model to: {model_path}")
    
if __name__ == "__main__":
    main()


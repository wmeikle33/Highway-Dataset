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

    train_ds = HighwayVideoClips(ROOT, T=args.frames, size=args.size, train=True)
    val_ds   = HighwayVideoClips(ROOT, T=args.frames, size=args.size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Simple3DCNN(num_classes=2).to(device)

     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.model_out)
    
if __name__ == "__main__":
    main()


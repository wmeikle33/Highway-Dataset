class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=2, in_ch=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_ch, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d((1,2,2)),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.MaxPool3d((2,2,2)),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.MaxPool3d((2,2,2)),
        )
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):           # x: (B,C,T,H,W)
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

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

def train_eval_save():
    ROOT =  ""
    T, SIZE, BATCH = 16, 112, 4

    train_ds = HighwayVideoClips(ROOT, T=T, size=SIZE, train=True)
    val_ds   = HighwayVideoClips(ROOT, T=T, size=SIZE, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    train_ds.summarize_counts(train_ds.labels)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Simple3DCNN(num_classes=2).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, loss_fn, device)
        va_loss, va_acc, cm = evaluate(model, val_loader, loss_fn, device, return_cm=True, num_classes=2)

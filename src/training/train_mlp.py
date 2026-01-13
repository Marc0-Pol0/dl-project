from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_utils import (
    ensure_sorted_datetime,
    eval_multiclass,
    get_xy,
    pick_feature_columns,
    print_split_sizes,
    read_table,
    time_split,
    time_val_split,
)


DATA = Path("data/trainable/event_table_500.parquet")
OUT = Path("networks/mlp_earnings_model.joblib")

SPLIT_DATE = "2025-05-01"
VAL_TAIL_FRAC = 0.15

RANDOM_STATE = 0
CLASS_ORDER = ["heavy_down", "down", "neutral", "up", "heavy_up"]

HIDDEN_DIMS = (256, 128)
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 1000
PATIENCE = 100

USE_CLASS_WEIGHTS = True


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = resolve_device()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
        ]
    )


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...], dropout: float, out_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = int(in_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(d, int(h)))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            d = int(h)
        layers.append(nn.Linear(d, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_loader_multiclass(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.int64)),
    )
    return DataLoader(ds, batch_size=int(batch_size), shuffle=bool(shuffle), drop_last=False)


@torch.no_grad()
def predict_proba_multiclass(model: nn.Module, x: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    loader = DataLoader(x_t, batch_size=int(batch_size), shuffle=False, drop_last=False)

    probs: list[np.ndarray] = []
    for xb in loader:
        xb = xb.to(device)
        logits = model(xb)  # (B,K)
        p = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        probs.append(p)

    if not probs:
        # shape (0, K) is handled by caller based on x.shape[0]
        return np.zeros((0, 0), dtype=np.float32)

    return np.concatenate(probs, axis=0)


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Inverse-frequency weights: w_c = N / (K * count_c)
    Gives equal total weight per class (roughly).
    """
    y = np.asarray(y_train).astype(int)
    counts = np.bincount(y, minlength=int(num_classes)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    n = float(len(y))
    k = float(num_classes)
    w = n / (k * counts)
    return w.astype(np.float32)


def train_one_epoch(
    model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, loss_fn: nn.Module, device: str
) -> float:
    model.train()
    total = 0.0
    n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()

        bs = int(xb.shape[0])
        total += float(loss.item()) * bs
        n += bs

    return total / max(1, n)


def run_split_multiclass(name: str, model: nn.Module, x: np.ndarray, y: np.ndarray, class_names: list[str]) -> None:
    if len(y) == 0:
        print(f"\n[{name}] empty")
        return
    print(f"\n[{name}]")
    proba = predict_proba_multiclass(model, x, BATCH_SIZE, DEVICE)
    eval_multiclass(y, proba, class_names=class_names)


def main() -> None:
    set_seeds(RANDOM_STATE)

    df = read_table(DATA)
    df = ensure_sorted_datetime(df, "earnings_day")

    feature_cols = pick_feature_columns(df, ("sent_", "f_"))
    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found with prefixes ('sent_', 'f_').")

    train_all, test = time_split(df, SPLIT_DATE, date_col="earnings_day")
    train, val = time_val_split(train_all, VAL_TAIL_FRAC, date_col="earnings_day")

    print_split_sizes(df, train, val, test)
    print("feature_cols:", len(feature_cols))
    print("device:", DEVICE)
    print("label_col:", "label")
    print("class_order:", CLASS_ORDER)

    x_tr, y_tr, class_names = get_xy(train, feature_cols, "label", class_order=CLASS_ORDER)
    if list(class_names) != list(CLASS_ORDER):
        raise RuntimeError(f"class_names={class_names} != CLASS_ORDER={CLASS_ORDER}")

    x_te, y_te, _ = get_xy(test, feature_cols, "label", class_order=CLASS_ORDER)

    if len(val) > 0:
        x_va, y_va, _ = get_xy(val, feature_cols, "label", class_order=CLASS_ORDER)
    else:
        x_va = np.zeros((0, x_tr.shape[1]), dtype=np.float32)
        y_va = np.zeros((0,), dtype=np.int64)

    pre = build_preprocessor()
    x_tr = pre.fit_transform(x_tr)
    x_te = pre.transform(x_te)
    x_va = pre.transform(x_va) if len(val) > 0 else x_va

    x_tr = np.asarray(x_tr, dtype=np.float32)
    x_va = np.asarray(x_va, dtype=np.float32)
    x_te = np.asarray(x_te, dtype=np.float32)

    out_dim = len(CLASS_ORDER)
    model = MLP(in_dim=int(x_tr.shape[1]), hidden_dims=HIDDEN_DIMS, dropout=DROPOUT, out_dim=out_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_loader = make_loader_multiclass(x_tr, y_tr, BATCH_SIZE, shuffle=True)

    if USE_CLASS_WEIGHTS:
        w = compute_class_weights(y_tr, num_classes=len(CLASS_ORDER))
        print("class_weights:", {class_names[i]: float(w[i]) for i in range(len(w))})
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))
    else:
        loss_fn = nn.CrossEntropyLoss()

    best_state: dict[str, torch.Tensor] | None = None
    best_val_acc = -1.0
    patience_left = int(PATIENCE)

    for epoch in range(1, int(MAX_EPOCHS) + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, DEVICE)

        if len(val) == 0:
            print(f"epoch {epoch:03d} | train_loss={tr_loss:.5f}")
            continue

        val_proba = predict_proba_multiclass(model, x_va, BATCH_SIZE, DEVICE)
        val_pred = val_proba.argmax(axis=1)
        val_acc = float((val_pred == y_va).mean()) if len(y_va) > 0 else float("nan")
        print(f"epoch {epoch:03d} | train_loss={tr_loss:.5f} | val_acc={val_acc:.4f}")

        if np.isfinite(val_acc) and val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(PATIENCE)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[INFO] early stopping at epoch {epoch} (best_val_acc={best_val_acc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    run_split_multiclass("train", model, x_tr, y_tr, class_names)
    if len(val) > 0:
        run_split_multiclass("val", model, x_va, y_va, class_names)
    run_split_multiclass("test", model, x_te, y_te, class_names)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "state_dict": model.state_dict(),
        "preprocessor": pre,
        "feature_cols": feature_cols,
        "class_names": class_names,
        "class_order": CLASS_ORDER,
        "date_col": "earnings_day",
        "split_date": SPLIT_DATE,
        "val_tail_frac": float(VAL_TAIL_FRAC),
        "data_path": str(DATA),
        "device_hint": DEVICE,
        "hyperparams": {
            "hidden_dims": tuple(int(x) for x in HIDDEN_DIMS),
            "dropout": float(DROPOUT),
            "lr": float(LR),
            "weight_decay": float(WEIGHT_DECAY),
            "batch_size": int(BATCH_SIZE),
            "max_epochs": int(MAX_EPOCHS),
            "patience": int(PATIENCE),
            "use_class_weights": bool(USE_CLASS_WEIGHTS),
            "random_state": int(RANDOM_STATE),
        },
    }
    joblib.dump(bundle, OUT)
    print(f"\nSaved model bundle to: {OUT}")


if __name__ == "__main__":
    main()

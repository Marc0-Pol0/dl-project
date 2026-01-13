"""
Train and evaluate an MLP baseline on the earnings-event dataset.

Supports:
- Binary target (label_up) with BCEWithLogitsLoss
- 3-class target (label: down/neutral/up) with CrossEntropyLoss

Pipeline:
1. Load the event-level earnings dataset
2. Sort by time
3. Select numerical features by prefix ("sent_", "f_")
4. Fit preprocessing on train only: median imputation + standardization
5. Train an MLP (binary: BCEWithLogitsLoss; multiclass: CrossEntropyLoss with optional class weights)
6. Evaluate performance on train / val / test splits
7. Save a model bundle (weights + preprocessing + config) for reuse
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_utils import (
    SplitConfig,
    ensure_sorted_datetime,
    eval_binary,
    eval_multiclass,
    get_xy,
    pick_feature_columns,
    print_split_sizes,
    read_table,
    time_split,
    time_val_split,
)

# -------------------------- paths / split --------------------------

DATA = Path("data/trainable/event_table_500.parquet")
OUT = Path("networks/mlp_earnings_model.joblib")

SPLIT_DATE = "2025-05-01"
VAL_TAIL_FRAC = 0.15

# -------------------------- task selection --------------------------
# Set True for 3-class ("label"), False for binary ("label_up")
MULTICLASS = True

# -------------------------- runtime --------------------------

RANDOM_STATE = 0
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# -------------------------- model hyperparams --------------------------

HIDDEN_DIMS = (256, 128)
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 1000
PATIENCE = 100

# imbalance handling:
USE_POS_WEIGHT = True          # binary only
USE_CLASS_WEIGHTS = True       # multiclass only


@dataclass(frozen=True)
class MLPConfig:
    hidden_dims: tuple[int, ...] = (256, 128)
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 1000
    patience: int = 100

    multiclass: bool = True
    num_classes: int = 3
    use_pos_weight: bool = True           # binary
    use_class_weights: bool = True        # multiclass

    random_state: int = 0
    device: str = "cpu"
    class_order: tuple[str, ...] = ("down", "neutral", "up")


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
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
        return self.net(x)  # (B, out_dim)


def make_loader_binary(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.float32)),  # BCE expects float targets
    )
    return DataLoader(ds, batch_size=int(batch_size), shuffle=bool(shuffle), drop_last=False)


def make_loader_multiclass(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.int64)),    # CE expects int64 class indices
    )
    return DataLoader(ds, batch_size=int(batch_size), shuffle=bool(shuffle), drop_last=False)


@torch.no_grad()
def predict_proba_binary(model: nn.Module, x: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    loader = DataLoader(x_t, batch_size=int(batch_size), shuffle=False, drop_last=False)
    probs: list[np.ndarray] = []
    for xb in loader:
        xb = xb.to(device)
        logits = model(xb).squeeze(-1)  # (B,)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


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
    return np.concatenate(probs, axis=0)


def compute_pos_weight(y_train: np.ndarray) -> float:
    y = np.asarray(y_train).astype(int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos <= 0:
        return 1.0
    return float(neg / pos)


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Inverse-frequency weights: w_c = N / (K * count_c)
    Gives equal total weight per class (roughly).
    """
    y = np.asarray(y_train).astype(int)
    counts = np.bincount(y, minlength=int(num_classes)).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # avoid div-by-zero if a class is missing
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


def run_split_binary(name: str, model: nn.Module, x: np.ndarray, y: np.ndarray, cfg: MLPConfig) -> dict[str, float]:
    print(f"\n[{name}]")
    prob = predict_proba_binary(model, x, cfg.batch_size, cfg.device)
    return eval_binary(y, prob, thresh=0.5)


def run_split_multiclass(
    name: str, model: nn.Module, x: np.ndarray, y: np.ndarray, cfg: MLPConfig, class_names: list[str]
) -> dict[str, float]:
    print(f"\n[{name}]")
    proba = predict_proba_multiclass(model, x, cfg.batch_size, cfg.device)
    return eval_multiclass(y, proba, class_names=class_names)


def main() -> None:
    set_seeds(RANDOM_STATE)

    split_cfg = SplitConfig(
        label_col="label" if MULTICLASS else "label_up",
        split_date=SPLIT_DATE,
        val_tail_frac=VAL_TAIL_FRAC,
        class_order=("down", "neutral", "up"),
    )

    df = read_table(DATA)
    if split_cfg.label_col not in df.columns:
        raise KeyError(f"Missing label column: {split_cfg.label_col}")
    if split_cfg.date_col not in df.columns:
        raise KeyError(f"Missing date column: {split_cfg.date_col}")

    df = ensure_sorted_datetime(df, split_cfg.date_col)
    feature_cols = pick_feature_columns(df, split_cfg.feature_prefixes)
    if not feature_cols:
        raise ValueError(f"No feature columns found with prefixes: {split_cfg.feature_prefixes}")

    train_all, test = time_split(df, split_cfg.date_col, split_cfg.split_date)
    train, val = time_val_split(train_all, split_cfg.date_col, split_cfg.val_tail_frac)

    print_split_sizes(df, train, val, test)
    print("feature_cols:", len(feature_cols))
    print("device:", DEVICE)
    print("label_col:", split_cfg.label_col)

    x_tr, y_tr, class_names = get_xy(train, feature_cols, split_cfg.label_col, class_order=split_cfg.class_order)
    x_te, y_te, _ = get_xy(test, feature_cols, split_cfg.label_col, class_order=split_cfg.class_order)

    if len(val) > 0:
        x_va, y_va, _ = get_xy(val, feature_cols, split_cfg.label_col, class_order=split_cfg.class_order)
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

    cfg = MLPConfig(
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        multiclass=MULTICLASS,
        num_classes=len(class_names) if MULTICLASS else 2,
        use_pos_weight=USE_POS_WEIGHT,
        use_class_weights=USE_CLASS_WEIGHTS,
        random_state=RANDOM_STATE,
        device=DEVICE,
        class_order=tuple(split_cfg.class_order),
    )

    out_dim = cfg.num_classes if cfg.multiclass else 1
    model = MLP(in_dim=int(x_tr.shape[1]), hidden_dims=cfg.hidden_dims, dropout=cfg.dropout, out_dim=out_dim).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    # Loss + loader selection
    if cfg.multiclass:
        train_loader = make_loader_multiclass(x_tr, y_tr, cfg.batch_size, shuffle=True)

        if cfg.use_class_weights:
            w = compute_class_weights(y_tr, num_classes=cfg.num_classes)
            print("class_weights:", {class_names[i]: float(w[i]) for i in range(len(w))})
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=cfg.device))
        else:
            loss_fn = nn.CrossEntropyLoss()

        best_state: dict[str, torch.Tensor] | None = None
        best_val_acc = -1.0
        patience_left = int(cfg.patience)

        for epoch in range(1, int(cfg.max_epochs) + 1):
            tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, cfg.device)

            if len(val) == 0:
                print(f"epoch {epoch:03d} | train_loss={tr_loss:.5f}")
                continue

            val_proba = predict_proba_multiclass(model, x_va, cfg.batch_size, cfg.device)
            val_pred = val_proba.argmax(axis=1)
            val_acc = float((val_pred == y_va).mean()) if len(y_va) > 0 else float("nan")
            print(f"epoch {epoch:03d} | train_loss={tr_loss:.5f} | val_acc={val_acc:.4f}")

            if np.isfinite(val_acc) and val_acc > best_val_acc + 1e-6:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = int(cfg.patience)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[INFO] early stopping at epoch {epoch} (best_val_acc={best_val_acc:.4f})")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        run_split_multiclass("train", model, x_tr, y_tr, cfg, class_names)
        if len(val) > 0:
            run_split_multiclass("val", model, x_va, y_va, cfg, class_names)
        run_split_multiclass("test", model, x_te, y_te, cfg, class_names)

    else:
        # binary
        train_loader = make_loader_binary(x_tr, y_tr, cfg.batch_size, shuffle=True)

        pos_w = compute_pos_weight(y_tr) if cfg.use_pos_weight else 1.0
        if cfg.use_pos_weight:
            print(f"pos_weight (neg/pos) on train: {pos_w:.4f}")
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32, device=cfg.device))
        else:
            loss_fn = nn.BCEWithLogitsLoss()

        best_state: dict[str, torch.Tensor] | None = None
        best_val_auc = -1.0
        patience_left = int(cfg.patience)

        for epoch in range(1, int(cfg.max_epochs) + 1):
            tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, cfg.device)

            if len(val) == 0:
                print(f"epoch {epoch:03d} | train_loss={tr_loss:.5f}")
                continue

            val_prob = predict_proba_binary(model, x_va, cfg.batch_size, cfg.device)
            val_auc = float("nan") if len(np.unique(y_va)) <= 1 else float(roc_auc_score(y_va, val_prob))
            print(f"epoch {epoch:03d} | train_loss={tr_loss:.5f} | val_auc={val_auc:.4f}")

            if np.isfinite(val_auc) and val_auc > best_val_auc + 1e-6:
                best_val_auc = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = int(cfg.patience)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[INFO] early stopping at epoch {epoch} (best_val_auc={best_val_auc:.4f})")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        run_split_binary("train", model, x_tr, y_tr, cfg)
        if len(val) > 0:
            run_split_binary("val", model, x_va, y_va, cfg)
        run_split_binary("test", model, x_te, y_te, cfg)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "state_dict": model.state_dict(),
        "preprocessor": pre,
        "feature_cols": feature_cols,
        "split_cfg": split_cfg,
        "mlp_cfg": cfg,
        "class_names": class_names,
        "data_path": str(DATA),
    }
    joblib.dump(bundle, OUT)
    print(f"\nSaved model bundle to: {OUT}")


if __name__ == "__main__":
    main()

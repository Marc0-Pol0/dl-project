"""
Train and evaluate an MLP baseline on the earnings-event dataset.

Pipeline:
1. Load the event-level earnings dataset
2. Sort by time
3. Select numerical features by prefix ("sent_", "f_")
4. Fit preprocessing on train only: median imputation + standardization
5. Train an MLP with BCEWithLogitsLoss (optionally pos_weight for imbalance)
6. Evaluate performance on train / val / test splits
7. Save a model bundle (weights + preprocessing + config) for reuse

Notes:
- Time-based splits only (no random shuffling across time)
- Uses early stopping on validation AUC (if a validation split exists)
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

# -------------------------- runtime --------------------------

RANDOM_STATE = 0
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():# -------------------------- model hyperparams --------------------------
    DEVICE = "mps"
else:
    DEVICE = "cpu"



HIDDEN_DIMS = (256, 128)
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
USE_POS_WEIGHT = True


@dataclass(frozen=True)
class MLPConfig:
    hidden_dims: tuple[int, ...] = (256, 128)
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
    use_pos_weight: bool = True
    random_state: int = 0
    device: str = "cpu"


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # may affect speed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_preprocessor() -> Pipeline:
    """
    Preprocessor for numeric arrays:
    - median imputation
    - standardization
    """
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
        ]
    )


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = int(in_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(d, int(h)))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            d = int(h)
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.float32)),
    )
    return DataLoader(ds, batch_size=int(batch_size), shuffle=bool(shuffle), drop_last=False)


@torch.no_grad()
def predict_proba(model: nn.Module, x: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    loader = DataLoader(x_t, batch_size=int(batch_size), shuffle=False, drop_last=False)
    probs: list[np.ndarray] = []
    for xb in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


def compute_pos_weight(y_train: np.ndarray) -> float:
    y = np.asarray(y_train).astype(int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos <= 0:
        return 1.0
    return float(neg / pos)


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


def run_split(name: str, model: nn.Module, x: np.ndarray, y: np.ndarray, cfg: MLPConfig) -> dict[str, float]:
    print(f"\n[{name}]")
    prob = predict_proba(model, x, cfg.batch_size, cfg.device)
    return eval_binary(y, prob, thresh=0.5)


def main() -> None:
    set_seeds(RANDOM_STATE)

    split_cfg = SplitConfig(split_date=SPLIT_DATE, val_tail_frac=VAL_TAIL_FRAC)

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

    x_tr, y_tr = get_xy(train, feature_cols, split_cfg.label_col)
    x_te, y_te = get_xy(test, feature_cols, split_cfg.label_col)

    if len(val) > 0:
        x_va, y_va = get_xy(val, feature_cols, split_cfg.label_col)
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
        use_pos_weight=USE_POS_WEIGHT,
        random_state=RANDOM_STATE,
        device=DEVICE,
    )

    model = MLP(in_dim=int(x_tr.shape[1]), hidden_dims=cfg.hidden_dims, dropout=cfg.dropout).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    pos_w = compute_pos_weight(y_tr) if cfg.use_pos_weight else 1.0
    if cfg.use_pos_weight:
        print(f"pos_weight (neg/pos) on train: {pos_w:.4f}")

    # Only pass pos_weight if requested (and always as a tensor on device)
    if cfg.use_pos_weight:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32, device=cfg.device))
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    train_loader = make_loader(x_tr, y_tr, cfg.batch_size, shuffle=True)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_auc = -1.0
    patience_left = int(cfg.patience)

    for epoch in range(1, int(cfg.max_epochs) + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, cfg.device)

        if len(val) == 0:
            print(f"epoch {epoch:03d} | train_loss={tr_loss:.5f}")
            continue

        val_prob = predict_proba(model, x_va, cfg.batch_size, cfg.device)
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

    run_split("train", model, x_tr, y_tr, cfg)
    if len(val) > 0:
        run_split("val", model, x_va, y_va, cfg)
    run_split("test", model, x_te, y_te, cfg)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "state_dict": model.state_dict(),
        "preprocessor": pre,
        "feature_cols": feature_cols,
        "split_cfg": split_cfg,
        "mlp_cfg": cfg,
        "data_path": str(DATA),
    }
    joblib.dump(bundle, OUT)
    print(f"\nSaved model bundle to: {OUT}")


if __name__ == "__main__":
    main()

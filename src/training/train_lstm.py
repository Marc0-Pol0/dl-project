"""
Train and evaluate an LSTM model on the earnings-event dataset.

Unlike logreg/xgboost/mlp (event-level tabular features), LSTM consumes a *sequence* of per-day features leading up to 
the earnings event.

Inputs:
- Event table (.parquet/.csv) produced by build_event_table.py (one row per earnings event), provides:
    * label_up (0/1)
    * feature_day (the last day whose features are allowed)
    * earnings_day (used for time splits)
    * ticker (used to index into final_data)
    * f_* columns (define which per-day features to extract)
- final_data_*.pkl produced earlier:
    dict[ticker] -> DataFrame indexed by trading-day DatetimeIndex with daily features (same keys as f_* without "f_")

Sequence construction:
- For each event row, build a fixed-length window of SEQ_LEN trading days ending at feature_day (inclusive).
- If history is shorter than SEQ_LEN, left-pad with NaNs (then impute).
- Per-time-step feature set is the base feature names derived from event table f_* columns.

Preprocessing (fit on train only):
- Median imputation (per feature) + standardization, applied to all time steps.

Model:
- LSTM over sequences, using last hidden state -> linear -> logit
- BCEWithLogitsLoss, optional pos_weight for imbalance
- Early stopping on validation AUC (if validation split exists)

Outputs:
- Printed metrics on train/val/test
- Saved model bundle (.joblib): state_dict, preprocessing, feature list, and configs
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_utils import (
    SplitConfig,
    ensure_sorted_datetime,
    eval_binary,
    pick_feature_columns,
    print_split_sizes,
    read_table,
    time_split,
    time_val_split,
)

# ----------------------------- paths -----------------------------

EVENT_TABLE = Path("data/trainable/event_table_500.parquet")
FINAL_DATA = Path("data/trainable/final_data_500.pkl")
OUT = Path("networks/lstm_earnings_model.joblib")

# ----------------------------- split -----------------------------

SPLIT_DATE = "2025-05-01"
VAL_TAIL_FRAC = 0.15

# ----------------------------- device -----------------------------

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ----------------------------- model/training defaults -----------------------------

RANDOM_STATE = 0
SEQ_LEN = 30

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
MAX_EPOCHS = 60
PATIENCE = 8
USE_POS_WEIGHT = True


@dataclass(frozen=True)
class LSTMConfig:
    seq_len: int = 30
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 60
    patience: int = 8
    use_pos_weight: bool = True
    random_state: int = 0
    device: str = "cpu"


# ----------------------------- utils -----------------------------

def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pickle(path: Path):
    """Load a pickle or joblib file from path."""
    with open(path, "rb") as f:
        return joblib.load(f) if path.suffix.lower() == ".joblib" else __import__("pickle").load(f)


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame index is a sorted, timezone-naive DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Expected DatetimeIndex, got {type(df.index)}")
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert(None)
    return df.sort_index()


def compute_pos_weight(y_train: np.ndarray) -> float:
    """
    Compute the positive class weight (neg/pos) for BCEWithLogitsLoss.
    Assumes y_train contains binary labels {0,1}.
    """
    y = y_train.astype(int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0:
        return 1.0
    return float(neg / pos)


def base_feature_names_from_event_table(event_df: pd.DataFrame) -> list[str]:
    """
    Derive the base per-day feature names from the event table f_* columns.
    Strips the "f_" prefix from the column names.
    """
    f_cols = pick_feature_columns(event_df, prefixes=("f_",))
    return [c[len("f_") :] for c in f_cols]


def build_sequence(
    per_ticker_df: pd.DataFrame, feature_day: pd.Timestamp, base_cols: list[str], seq_len: int
) -> np.ndarray:
    """
    Build a sequence of per-day features for a given ticker up to feature_day (inclusive).
    Left-pad with NaNs if history is shorter than seq_len.
    """
    idx = per_ticker_df.index
    if feature_day not in idx:
        # If feature_day is not a trading day for this ticker, try nearest previous trading day.
        pos = int(idx.searchsorted(feature_day, side="right")) - 1
        if pos < 0:
            arr = np.full((seq_len, len(base_cols)), np.nan, dtype=np.float32)
            return arr
        feature_day = idx[pos]

    end_pos = int(idx.get_loc(feature_day))
    start_pos = max(0, end_pos - (seq_len - 1))
    window = per_ticker_df.iloc[start_pos : end_pos + 1]

    x = window.reindex(columns=base_cols).to_numpy(dtype=np.float32)
    if x.shape[0] < seq_len:
        pad = np.full((seq_len - x.shape[0], x.shape[1]), np.nan, dtype=np.float32)
        x = np.vstack([pad, x])
    return x


def fit_preprocessing(x_train: np.ndarray) -> tuple[SimpleImputer, StandardScaler]:
    """Fit imputer and scaler on the training sequences."""
    n, t, f = x_train.shape
    flat = x_train.reshape(n * t, f)
    imp = SimpleImputer(strategy="median")
    flat_imp = imp.fit_transform(flat)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(flat_imp)
    return imp, scaler


def apply_preprocessing(x: np.ndarray, imp: SimpleImputer, scaler: StandardScaler) -> np.ndarray:
    """
    Apply fitted imputer and scaler to sequences.
    x: [N, T, F] -> impute and scale per feature using all time steps
    """
    n, t, f = x.shape
    flat = x.reshape(n * t, f)
    flat = imp.transform(flat)
    flat = scaler.transform(flat)
    return flat.reshape(n, t, f).astype(np.float32)


class EventSequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        do = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=do,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """Predict probabilities using the LSTM model on data from loader."""
    model.eval()
    probs: list[np.ndarray] = []
    for xb, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


def train_one_epoch(
    model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, loss_fn: nn.Module, device: str
) -> float:
    """
    Train the model for one epoch over the data in loader.
    Returns the average loss over the epoch.
    """
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


def eval_split(name: str, model: nn.Module, x: np.ndarray, y: np.ndarray, cfg: LSTMConfig) -> dict[str, float]:
    """Run evaluation on a data split and print metrics."""
    if len(y) == 0:
        print(f"\n[{name}] empty")
        return {"auc": float("nan")}

    ds = EventSequenceDataset(x, y)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    prob = predict_proba(model, loader, cfg.device)
    print(f"\n[{name}]")
    return eval_binary(y.astype(int), prob, thresh=0.5)


# ----------------------------- main -----------------------------

def main() -> None:
    set_seeds(RANDOM_STATE)

    split_cfg = SplitConfig(split_date=SPLIT_DATE, val_tail_frac=VAL_TAIL_FRAC)
    event_df = read_table(EVENT_TABLE)
    if split_cfg.label_col not in event_df.columns:
        raise KeyError(f"Missing label column: {split_cfg.label_col}")
    if split_cfg.date_col not in event_df.columns:
        raise KeyError(f"Missing date column: {split_cfg.date_col}")
    if "ticker" not in event_df.columns or "feature_day" not in event_df.columns:
        raise KeyError("Event table must contain 'ticker' and 'feature_day'")

    event_df = ensure_sorted_datetime(event_df, split_cfg.date_col)
    event_df["feature_day"] = pd.to_datetime(event_df["feature_day"])

    base_cols = base_feature_names_from_event_table(event_df)
    if not base_cols:
        raise ValueError("No per-day base features found. Expected f_* columns in event table.")

    final_data: dict[str, pd.DataFrame] = load_pickle(FINAL_DATA)
    final_data = {k: ensure_datetime_index(v) for k, v in final_data.items()}

    train_all, test = time_split(event_df, split_cfg.date_col, split_cfg.split_date)
    train, val = time_val_split(train_all, split_cfg.date_col, split_cfg.val_tail_frac)

    print_split_sizes(event_df, train, val, test)
    print("seq_len:", SEQ_LEN, "n_features:", len(base_cols))
    print("device:", DEVICE)

    def build_xy(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        xs: list[np.ndarray] = []
        ys: list[int] = []
        for _, row in part.iterrows():
            t = str(row["ticker"])
            if t not in final_data:
                continue
            df_t = final_data[t]
            x = build_sequence(df_t, pd.Timestamp(row["feature_day"]), base_cols, SEQ_LEN)
            xs.append(x)
            ys.append(int(row[split_cfg.label_col]))
        if not xs:
            return np.zeros((0, SEQ_LEN, len(base_cols)), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64)

    x_tr_raw, y_tr = build_xy(train)
    x_va_raw, y_va = build_xy(val) if len(val) else (np.zeros((0, SEQ_LEN, len(base_cols)), dtype=np.float32), np.zeros(0))
    x_te_raw, y_te = build_xy(test)

    imp, scaler = fit_preprocessing(x_tr_raw)
    x_tr = apply_preprocessing(x_tr_raw, imp, scaler)
    x_va = apply_preprocessing(x_va_raw, imp, scaler) if len(y_va) else x_va_raw
    x_te = apply_preprocessing(x_te_raw, imp, scaler)

    cfg = LSTMConfig(
        seq_len=SEQ_LEN,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
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

    model = LSTMClassifier(in_dim=len(base_cols), hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, dropout=cfg.dropout)
    model = model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    pos_w = compute_pos_weight(y_tr) if cfg.use_pos_weight else 1.0
    if cfg.use_pos_weight:
        print(f"pos_weight (neg/pos) on train: {pos_w:.4f}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32, device=cfg.device))

    train_ds = EventSequenceDataset(x_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    best_state = None
    best_val_auc = -1.0
    patience_left = cfg.patience

    for epoch in range(1, cfg.max_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, cfg.device)

        if len(y_va) == 0:
            print(f"epoch {epoch:03d} | train_loss={tr_loss:.5f}")
            continue

        metrics = eval_split("val_tmp", model, x_va, y_va, cfg)
        val_auc = float(metrics.get("auc", float("nan")))
        print(f"epoch {epoch:03d} | train_loss={tr_loss:.5f} | val_auc={val_auc:.4f}")

        if np.isfinite(val_auc) and val_auc > best_val_auc + 1e-6:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[INFO] early stopping at epoch {epoch} (best_val_auc={best_val_auc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    eval_split("train", model, x_tr, y_tr, cfg)
    if len(y_va):
        eval_split("val", model, x_va, y_va, cfg)
    eval_split("test", model, x_te, y_te, cfg)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "state_dict": model.state_dict(),
        "imputer": imp,
        "scaler": scaler,
        "base_feature_cols": base_cols,
        "split_cfg": split_cfg,
        "lstm_cfg": cfg,
    }
    joblib.dump(bundle, OUT)
    print(f"\nSaved model bundle to: {OUT}")


if __name__ == "__main__":
    main()

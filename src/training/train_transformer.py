"""
Train and evaluate a Transformer model on the earnings-event dataset.

This mirrors train_lstm.py but replaces the recurrent encoder with a Transformer encoder over a fixed-length sequence of
daily features leading up to each earnings event.

Inputs:
- Event table (.parquet/.csv) produced by build_event_table.py (one row per earnings event), provides:
    * label_up (0/1)
    * feature_day (last day whose features are allowed)
    * earnings_day (used for time splits)
    * ticker (used to index into final_data)
    * f_* columns (define which per-day features to extract)
- final_data_*.pkl:
    dict[ticker] -> DataFrame indexed by trading-day DatetimeIndex with daily features

Sequence construction:
- For each event row, build a fixed-length window of SEQ_LEN trading days ending at feature_day (inclusive).
- If history is shorter than SEQ_LEN, left-pad with NaNs (then impute).
- Features per step are base feature names derived from event table f_* columns.

Preprocessing (fit on train only):
- Median imputation (per feature) + standardization, applied to all time steps.
- A padding mask is built from NaNs before imputation (True = padded positions).

Model:
- Input projection: Linear(F -> D_MODEL)
- Positional encoding: sinusoidal (added to projected inputs)
- TransformerEncoder (batch_first=True) with src_key_padding_mask
- Pooling: masked mean pooling over time (ignoring padded steps)
- Head: Linear(D_MODEL -> 1) producing a logit
- Loss: BCEWithLogitsLoss, optional pos_weight for imbalance
- Early stopping on validation AUC (if validation split exists)

Outputs:
- Printed metrics on train/val/test
- Saved bundle (.joblib): state_dict, preprocessing, feature list, and configs
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
OUT = Path("networks/transformer_earnings_model.joblib")

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

D_MODEL = 128
N_HEAD = 4
NUM_LAYERS = 3
FF_DIM = 256
DROPOUT = 0.2

LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
MAX_EPOCHS = 60
PATIENCE = 8
USE_POS_WEIGHT = True


@dataclass(frozen=True)
class TransformerConfig:
    seq_len: int = 30
    d_model: int = 128
    n_head: int = 4
    num_layers: int = 3
    ff_dim: int = 256
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
        return __import__("pickle").load(f)


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


def build_sequence_and_padmask(
    per_ticker_df: pd.DataFrame, feature_day: pd.Timestamp, base_cols: list[str], seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build a fixed-length sequence and padding mask for a given ticker up to feature_day."""
    idx = per_ticker_df.index
    if feature_day not in idx:
        pos = int(idx.searchsorted(feature_day, side="right")) - 1
        if pos < 0:
            x = np.full((seq_len, len(base_cols)), np.nan, dtype=np.float32)
            pad = np.ones((seq_len,), dtype=bool)
            return x, pad
        feature_day = idx[pos]

    end_pos = int(idx.get_loc(feature_day))
    start_pos = max(0, end_pos - (seq_len - 1))
    window = per_ticker_df.iloc[start_pos : end_pos + 1]

    x = window.reindex(columns=base_cols).to_numpy(dtype=np.float32)
    if x.shape[0] < seq_len:
        pad_rows = seq_len - x.shape[0]
        pad_x = np.full((pad_rows, x.shape[1]), np.nan, dtype=np.float32)
        x = np.vstack([pad_x, x])

    pad_mask = np.isnan(x).all(axis=1)
    return x, pad_mask


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


class EventSeqMaskDataset(Dataset):
    def __init__(self, x: np.ndarray, pad_mask: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.m = torch.from_numpy(pad_mask.astype(bool))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.m[idx], self.y[idx]


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        t = x.shape[1]
        return x + self.pe[:t].unsqueeze(0)


class TransformerClassifier(nn.Module):
    def __init__(self, in_dim: int, cfg: TransformerConfig) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, cfg.d_model)
        self.pos = SinusoidalPositionalEncoding(cfg.d_model, cfg.seq_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_head,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F], pad_mask: [B, T] True where padded
        h = self.in_proj(x)
        h = self.pos(h)
        h = self.enc(h, src_key_padding_mask=pad_mask)

        # masked mean pooling
        valid = (~pad_mask).unsqueeze(-1).to(h.dtype)
        denom = valid.sum(dim=1).clamp_min(1.0)
        pooled = (h * valid).sum(dim=1) / denom
        return self.head(pooled).squeeze(-1)


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """Predict probabilities using the model and data loader."""
    model.eval()
    probs: list[np.ndarray] = []
    for xb, mb, _ in loader:
        xb = xb.to(device)
        mb = mb.to(device)
        logits = model(xb, mb)
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
    for xb, mb, yb in loader:
        xb = xb.to(device)
        mb = mb.to(device)
        yb = yb.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(xb, mb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        bs = int(xb.shape[0])
        total += float(loss.item()) * bs
        n += bs
    return total / max(1, n)


def eval_split(
    name: str, model: nn.Module, x: np.ndarray, pad_mask: np.ndarray, y: np.ndarray, cfg: TransformerConfig
) -> dict[str, float]:
    """Run evaluation on a data split and print metrics."""
    if len(y) == 0:
        print(f"\n[{name}] empty")
        return {"auc": float("nan")}

    ds = EventSeqMaskDataset(x, pad_mask, y)
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

    def build_xy(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs: list[np.ndarray] = []
        ms: list[np.ndarray] = []
        ys: list[int] = []
        for _, row in part.iterrows():
            t = str(row["ticker"])
            if t not in final_data:
                continue
            df_t = final_data[t]
            x, m = build_sequence_and_padmask(df_t, pd.Timestamp(row["feature_day"]), base_cols, SEQ_LEN)
            xs.append(x)
            ms.append(m)
            ys.append(int(row[split_cfg.label_col]))
        if not xs:
            x0 = np.zeros((0, SEQ_LEN, len(base_cols)), dtype=np.float32)
            m0 = np.ones((0, SEQ_LEN), dtype=bool)
            y0 = np.zeros((0,), dtype=np.int64)
            return x0, m0, y0
        return np.stack(xs, axis=0), np.stack(ms, axis=0), np.asarray(ys, dtype=np.int64)

    x_tr_raw, m_tr, y_tr = build_xy(train)
    x_va_raw, m_va, y_va = build_xy(val) if len(val) else (
        np.zeros((0, SEQ_LEN, len(base_cols)), dtype=np.float32),
        np.ones((0, SEQ_LEN), dtype=bool),
        np.zeros((0,), dtype=np.int64),
    )
    x_te_raw, m_te, y_te = build_xy(test)

    imp, scaler = fit_preprocessing(x_tr_raw)
    x_tr = apply_preprocessing(x_tr_raw, imp, scaler)
    x_va = apply_preprocessing(x_va_raw, imp, scaler) if len(y_va) else x_va_raw
    x_te = apply_preprocessing(x_te_raw, imp, scaler)

    cfg = TransformerConfig(
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        n_head=N_HEAD,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
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

    model = TransformerClassifier(in_dim=len(base_cols), cfg=cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    pos_w = compute_pos_weight(y_tr) if cfg.use_pos_weight else 1.0
    if cfg.use_pos_weight:
        print(f"pos_weight (neg/pos) on train: {pos_w:.4f}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32, device=cfg.device))

    train_ds = EventSeqMaskDataset(x_tr, m_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    best_state = None
    best_val_auc = -1.0
    patience_left = cfg.patience

    for epoch in range(1, cfg.max_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, cfg.device)

        if len(y_va) == 0:
            print(f"epoch {epoch:03d} | train_loss={tr_loss:.5f}")
            continue

        metrics = eval_split("val_tmp", model, x_va, m_va, y_va, cfg)
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

    eval_split("train", model, x_tr, m_tr, y_tr, cfg)
    if len(y_va):
        eval_split("val", model, x_va, m_va, y_va, cfg)
    eval_split("test", model, x_te, m_te, y_te, cfg)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "state_dict": model.state_dict(),
        "imputer": imp,
        "scaler": scaler,
        "base_feature_cols": base_cols,
        "split_cfg": split_cfg,
        "transformer_cfg": cfg,
    }
    joblib.dump(bundle, OUT)
    print(f"\nSaved model bundle to: {OUT}")


if __name__ == "__main__":
    main()

"""
Train and evaluate a Transformer model on the earnings-event dataset.

This mirrors train_transformer.py but supports:
- Binary target (label_up) with BCEWithLogitsLoss (+ optional pos_weight)
- 3-class target (label: down/neutral/up) with CrossEntropyLoss (+ optional class weights)

Inputs:
- Event table (.parquet/.csv) from build_event_table.py (one row per earnings event), provides:
    * label (down/neutral/up) OR label_up (0/1)
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
- Positional encoding: sinusoidal
- TransformerEncoder (batch_first=True) with src_key_padding_mask
- Pooling: masked mean pooling over time (ignoring padded steps)
- Head: Linear(D_MODEL -> out_dim) producing logits
- Loss:
    * binary: BCEWithLogitsLoss, optional pos_weight
    * multiclass: CrossEntropyLoss, optional class weights
- Early stopping:
    * binary: validation AUC
    * multiclass: validation accuracy

Outputs:
- Printed metrics on train/val/test
- Saved bundle (.joblib): state_dict, preprocessing, feature list, configs, class_names (and binary threshold=None)
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_utils import (
    SplitConfig,
    ensure_sorted_datetime,
    eval_binary,
    eval_multiclass,
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

# ----------------------------- task selection -----------------------------
# True: 3-class label ("label" with down/neutral/up); False: binary ("label_up")
MULTICLASS = True

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
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10

# imbalance handling:
USE_POS_WEIGHT = True      # binary only
USE_CLASS_WEIGHTS = True   # multiclass only


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
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10

    multiclass: bool = True
    num_classes: int = 3
    use_pos_weight: bool = True
    use_class_weights: bool = True

    random_state: int = 0
    device: str = "cpu"
    class_order: tuple[str, ...] = ("down", "neutral", "up")


# ----------------------------- utils -----------------------------

def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return __import__("pickle").load(f)


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Expected DatetimeIndex, got {type(df.index)}")
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert(None)
    return df.sort_index()


def compute_pos_weight(y_train: np.ndarray) -> float:
    y = np.asarray(y_train).astype(int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0:
        return 1.0
    return float(neg / pos)


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Inverse-frequency weights: w_c = N / (K * count_c)
    """
    y = np.asarray(y_train).astype(int)
    counts = np.bincount(y, minlength=int(num_classes)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    n = float(len(y))
    k = float(num_classes)
    w = n / (k * counts)
    return w.astype(np.float32)


def base_feature_names_from_event_table(event_df: pd.DataFrame) -> list[str]:
    f_cols = pick_feature_columns(event_df, prefixes=("f_",))
    f_cols.sort()
    return [c[len("f_") :] for c in f_cols]


def build_sequence_and_padmask(
    per_ticker_df: pd.DataFrame, feature_day: pd.Timestamp, base_cols: list[str], seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
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
    n, t, f = x_train.shape
    flat = x_train.reshape(n * t, f)
    imp = SimpleImputer(strategy="median")
    flat_imp = imp.fit_transform(flat)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(flat_imp)
    return imp, scaler


def apply_preprocessing(x: np.ndarray, imp: SimpleImputer, scaler: StandardScaler) -> np.ndarray:
    n, t, f = x.shape
    flat = x.reshape(n * t, f)
    flat = imp.transform(flat)
    flat = scaler.transform(flat)
    return flat.reshape(n, t, f).astype(np.float32)


# ----------------------------- datasets -----------------------------

class EventSeqMaskDatasetBinary(Dataset):
    def __init__(self, x: np.ndarray, pad_mask: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.m = torch.from_numpy(np.asarray(pad_mask, dtype=bool))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))  # BCE expects float

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.m[idx], self.y[idx]


class EventSeqMaskDatasetMC(Dataset):
    def __init__(self, x: np.ndarray, pad_mask: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.m = torch.from_numpy(np.asarray(pad_mask, dtype=bool))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))  # CE expects int64

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.m[idx], self.y[idx]


# ----------------------------- model -----------------------------

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
        out_dim = int(cfg.num_classes) if cfg.multiclass else 1
        self.head = nn.Linear(cfg.d_model, out_dim)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.pos(h)
        h = self.enc(h, src_key_padding_mask=pad_mask)

        valid = (~pad_mask).unsqueeze(-1).to(h.dtype)
        denom = valid.sum(dim=1).clamp_min(1.0)
        pooled = (h * valid).sum(dim=1) / denom
        out = self.head(pooled)
        return out.squeeze(-1) if out.shape[-1] == 1 else out  # (B,) or (B,K)


# ----------------------------- train/eval helpers -----------------------------

@torch.no_grad()
def predict_proba_binary(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    for xb, mb, _ in loader:
        xb = xb.to(device)
        mb = mb.to(device)
        logits = model(xb, mb)  # (B,)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


@torch.no_grad()
def predict_proba_multiclass(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    for xb, mb, _ in loader:
        xb = xb.to(device)
        mb = mb.to(device)
        logits = model(xb, mb)  # (B,K)
        p = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


def train_one_epoch(
    model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, loss_fn: nn.Module, device: str
) -> float:
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


def eval_split_binary(
    name: str, model: nn.Module, x: np.ndarray, pad_mask: np.ndarray, y: np.ndarray, cfg: TransformerConfig
) -> dict[str, float]:
    if len(y) == 0:
        print(f"\n[{name}] empty")
        return {"auc": float("nan")}

    ds = EventSeqMaskDatasetBinary(x, pad_mask, y)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    prob = predict_proba_binary(model, loader, cfg.device)
    print(f"\n[{name}]")
    return eval_binary(np.asarray(y, dtype=int), prob, thresh=0.5)


def eval_split_multiclass(
    name: str, model: nn.Module, x: np.ndarray, pad_mask: np.ndarray, y: np.ndarray, cfg: TransformerConfig, class_names: list[str]
) -> dict[str, float]:
    if len(y) == 0:
        print(f"\n[{name}] empty")
        return {"accuracy": float("nan")}

    ds = EventSeqMaskDatasetMC(x, pad_mask, y)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    proba = predict_proba_multiclass(model, loader, cfg.device)
    print(f"\n[{name}]")
    return eval_multiclass(np.asarray(y, dtype=int), proba, class_names=class_names)


# ----------------------------- main -----------------------------

def main() -> None:
    set_seeds(RANDOM_STATE)

    split_cfg = SplitConfig(
        label_col="label" if MULTICLASS else "label_up",
        split_date=SPLIT_DATE,
        val_tail_frac=VAL_TAIL_FRAC,
        class_order=("down", "neutral", "up"),
    )

    event_df = read_table(EVENT_TABLE)
    if split_cfg.label_col not in event_df.columns:
        raise KeyError(f"Missing label column: {split_cfg.label_col}")
    if split_cfg.date_col not in event_df.columns:
        raise KeyError(f"Missing date column: {split_cfg.date_col}")
    if "ticker" not in event_df.columns or "feature_day" not in event_df.columns:
        raise KeyError("Event table must contain 'ticker' and 'feature_day'")

    event_df = ensure_sorted_datetime(event_df, split_cfg.date_col)
    event_df["feature_day"] = pd.to_datetime(event_df["feature_day"], errors="coerce")
    event_df = event_df.dropna(subset=["feature_day"])

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
    print("label_col:", split_cfg.label_col)

    def encode_label(v: object) -> int:
        if MULTICLASS:
            s = str(v)
            order = list(split_cfg.class_order)
            if s not in order:
                raise ValueError(f"Unknown label '{s}' (expected one of {order})")
            return int(order.index(s))
        return int(v)

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
            ys.append(encode_label(row[split_cfg.label_col]))

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

    if len(y_tr) == 0 or len(y_te) == 0:
        raise RuntimeError("No train/test sequences built. Check tickers and feature_day alignment.")
    if len(np.unique(y_tr)) < 2:
        raise ValueError(f"Training split has <2 classes: {np.unique(y_tr)}")

    imp, scaler = fit_preprocessing(x_tr_raw)
    x_tr = apply_preprocessing(x_tr_raw, imp, scaler)
    x_va = apply_preprocessing(x_va_raw, imp, scaler) if len(y_va) else x_va_raw
    x_te = apply_preprocessing(x_te_raw, imp, scaler)

    class_names = list(split_cfg.class_order) if MULTICLASS else ["0", "1"]

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
        multiclass=MULTICLASS,
        num_classes=len(class_names) if MULTICLASS else 1,
        use_pos_weight=USE_POS_WEIGHT,
        use_class_weights=USE_CLASS_WEIGHTS,
        random_state=RANDOM_STATE,
        device=DEVICE,
        class_order=tuple(split_cfg.class_order),
    )

    model = TransformerClassifier(in_dim=len(base_cols), cfg=cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    # loaders
    if cfg.multiclass:
        train_ds = EventSeqMaskDatasetMC(x_tr, m_tr, y_tr)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

        if cfg.use_class_weights:
            w = compute_class_weights(y_tr, num_classes=cfg.num_classes)
            print("class_weights:", {class_names[i]: float(w[i]) for i in range(len(w))})
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=cfg.device))
        else:
            loss_fn = nn.CrossEntropyLoss()

        best_state = None
        best_val_acc = -1.0
        patience_left = int(cfg.patience)

        for epoch in range(1, int(cfg.max_epochs) + 1):
            tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, cfg.device)

            if len(y_va) == 0:
                print(f"epoch {epoch:03d} | train_loss={tr_loss:.5f}")
                continue

            va_ds = EventSeqMaskDatasetMC(x_va, m_va, y_va)
            va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
            va_proba = predict_proba_multiclass(model, va_loader, cfg.device)
            va_pred = va_proba.argmax(axis=1)
            va_acc = float((va_pred == y_va).mean()) if len(y_va) else float("nan")
            print(f"epoch {epoch:03d} | train_loss={tr_loss:.5f} | val_acc={va_acc:.4f}")

            if np.isfinite(va_acc) and va_acc > best_val_acc + 1e-6:
                best_val_acc = va_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = int(cfg.patience)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[INFO] early stopping at epoch {epoch} (best_val_acc={best_val_acc:.4f})")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        eval_split_multiclass("train", model, x_tr, m_tr, y_tr, cfg, class_names)
        if len(y_va):
            eval_split_multiclass("val", model, x_va, m_va, y_va, cfg, class_names)
        eval_split_multiclass("test", model, x_te, m_te, y_te, cfg, class_names)

    else:
        train_ds = EventSeqMaskDatasetBinary(x_tr, m_tr, y_tr)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

        pos_w = compute_pos_weight(y_tr) if cfg.use_pos_weight else 1.0
        if cfg.use_pos_weight:
            print(f"pos_weight (neg/pos) on train: {pos_w:.4f}")
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32, device=cfg.device))
        else:
            loss_fn = nn.BCEWithLogitsLoss()

        best_state = None
        best_val_auc = -1.0
        patience_left = int(cfg.patience)

        for epoch in range(1, int(cfg.max_epochs) + 1):
            tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, cfg.device)

            if len(y_va) == 0:
                print(f"epoch {epoch:03d} | train_loss={tr_loss:.5f}")
                continue

            va_ds = EventSeqMaskDatasetBinary(x_va, m_va, y_va)
            va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
            va_prob = predict_proba_binary(model, va_loader, cfg.device)
            va_auc = float("nan") if len(np.unique(y_va)) <= 1 else float(roc_auc_score(y_va, va_prob))
            print(f"epoch {epoch:03d} | train_loss={tr_loss:.5f} | val_auc={va_auc:.4f}")

            if np.isfinite(va_auc) and va_auc > best_val_auc + 1e-6:
                best_val_auc = va_auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = int(cfg.patience)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[INFO] early stopping at epoch {epoch} (best_val_auc={best_val_auc:.4f})")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        eval_split_binary("train", model, x_tr, m_tr, y_tr, cfg)
        if len(y_va):
            eval_split_binary("val", model, x_va, m_va, y_va, cfg)
        eval_split_binary("test", model, x_te, m_te, y_te, cfg)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "state_dict": model.state_dict(),
        "imputer": imp,
        "scaler": scaler,
        "base_feature_cols": base_cols,
        "split_cfg": split_cfg,
        "transformer_cfg": cfg,
        "class_names": class_names,
        "data_path_event_table": str(EVENT_TABLE),
        "data_path_final_data": str(FINAL_DATA),
    }
    joblib.dump(bundle, OUT)
    print(f"\nSaved model bundle to: {OUT}")


if __name__ == "__main__":
    main()

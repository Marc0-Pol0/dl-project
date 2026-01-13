from collections import Counter
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
    ensure_sorted_datetime,
    eval_multiclass,
    pick_feature_columns,
    print_split_sizes,
    read_table,
    time_split,
    time_val_split,
)

EVENT_TABLE = Path("data/trainable/event_table_500.parquet")
FINAL_DATA = Path("data/trainable/final_data_500.pkl")
OUT = Path("networks/transformer_earnings_model.joblib")

DATE_COL = "earnings_day"
SPLIT_DATE = "2025-05-01"
VAL_TAIL_FRAC = 0.15

RANDOM_STATE = 0
CLASS_ORDER = ["heavy_down", "down", "neutral", "up", "heavy_up"]

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


def load_pickle(path: Path):
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Sorted, timezone-naive, normalized DatetimeIndex (day resolution)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Expected DatetimeIndex, got {type(df.index)}")

    out = df.sort_index()
    if out.index.tz is not None:
        out = out.copy()
        out.index = out.index.tz_convert(None)

    out = out.copy()
    out.index = out.index.normalize()
    return out


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> np.ndarray:
    """Inverse-frequency weights: w_c = N / (K * count_c)"""
    y = np.asarray(y_train).astype(int)
    counts = np.bincount(y, minlength=int(num_classes)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    n = float(len(y))
    k = float(num_classes)
    w = n / (k * counts)
    return w.astype(np.float32)


def base_feature_names_from_event_table(event_df: pd.DataFrame) -> list[str]:
    """Derive base per-day feature names from event-table f_* columns (strip 'f_')."""
    f_cols = pick_feature_columns(event_df, prefixes=("f_",))
    f_cols.sort()
    return [c[len("f_") :] for c in f_cols]


def build_sequence_and_padmask(
    per_ticker_df: pd.DataFrame,
    feature_day: pd.Timestamp,
    base_cols: list[str],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a [T,F] sequence ending at feature_day (inclusive).
    Left-pad with NaNs if history is shorter than seq_len.
    pad_mask is True for padded steps (and also True if all features NaN in that row).
    """
    idx = per_ticker_df.index
    d = pd.Timestamp(feature_day).normalize()

    if d not in idx:
        pos = int(idx.searchsorted(d, side="right")) - 1
        if pos < 0:
            x = np.full((seq_len, len(base_cols)), np.nan, dtype=np.float32)
            pad = np.ones((seq_len,), dtype=bool)
            return x, pad
        d = idx[pos]

    end_pos = int(idx.get_loc(d))
    start_pos = max(0, end_pos - (seq_len - 1))
    window = per_ticker_df.iloc[start_pos : end_pos + 1]

    x = window.reindex(columns=base_cols).to_numpy(dtype=np.float32)
    if x.shape[0] < seq_len:
        pad_rows = seq_len - x.shape[0]
        pad_x = np.full((pad_rows, x.shape[1]), np.nan, dtype=np.float32)
        x = np.vstack([pad_x, x])

    pad_mask = np.isnan(x).all(axis=1)
    return x, pad_mask


def fit_preprocessing(x_train: np.ndarray, pad_mask_train: np.ndarray) -> tuple[SimpleImputer, StandardScaler]:
    """
    Fit imputer + scaler on training sequences using ONLY non-padded timesteps.
    x_train: [N,T,F]
    pad_mask_train: [N,T] True for padded/invalid rows
    """
    n, t, f = x_train.shape
    flat = x_train.reshape(n * t, f)
    valid = (~pad_mask_train.reshape(n * t)).astype(bool)

    if valid.sum() <= 0:
        raise ValueError("No valid (non-padded) timesteps in training data; cannot fit preprocessing.")

    flat_valid = flat[valid]

    imp = SimpleImputer(strategy="median")
    flat_imp = imp.fit_transform(flat_valid)

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(flat_imp)

    return imp, scaler


def apply_preprocessing(x: np.ndarray, imp: SimpleImputer, scaler: StandardScaler) -> np.ndarray:
    """Apply imputer + scaler to sequences x: [N,T,F] (includes padded rows)."""
    n, t, f = x.shape
    flat = x.reshape(n * t, f)
    flat = imp.transform(flat)
    flat = scaler.transform(flat)
    return flat.reshape(n, t, f).astype(np.float32)


class EventSeqMaskDatasetMC(Dataset):
    def __init__(self, x: np.ndarray, pad_mask: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.m = torch.from_numpy(np.asarray(pad_mask, dtype=bool))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))

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
        t = x.shape[1]
        return x + self.pe[:t].unsqueeze(0)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int,
        seq_len: int,
        n_head: int,
        ff_dim: int,
        dropout: float,
        num_layers: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(int(in_dim), int(d_model))
        self.pos = SinusoidalPositionalEncoding(int(d_model), int(seq_len))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(n_head),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.head = nn.Linear(int(d_model), int(num_classes))

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.pos(h)
        h = self.enc(h, src_key_padding_mask=pad_mask)

        valid = (~pad_mask).unsqueeze(-1).to(h.dtype)
        denom = valid.sum(dim=1).clamp_min(1.0)
        pooled = (h * valid).sum(dim=1) / denom
        return self.head(pooled)  # (B,K)


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
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, 0), dtype=np.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
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


def eval_split_multiclass(
    name: str,
    model: nn.Module,
    loader: DataLoader,
    y: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    if len(y) == 0:
        print(f"\n[{name}] empty")
        return {"accuracy": float("nan")}

    proba = predict_proba_multiclass(model, loader, DEVICE)
    print(f"\n[{name}]")
    return eval_multiclass(np.asarray(y, dtype=int), proba, class_names=class_names)


def main() -> None:
    set_seeds(RANDOM_STATE)

    event_df = read_table(EVENT_TABLE)
    event_df = ensure_sorted_datetime(event_df, DATE_COL)
    event_df["feature_day"] = pd.to_datetime(event_df["feature_day"], errors="coerce").dt.normalize()
    event_df = event_df.dropna(subset=["feature_day"])

    base_cols = base_feature_names_from_event_table(event_df)
    if not base_cols:
        raise ValueError("No per-day base features found. Expected f_* columns in event table.")

    final_data: dict[str, pd.DataFrame] = load_pickle(FINAL_DATA)
    final_data = {k: ensure_datetime_index(v) for k, v in final_data.items()}

    train_all, test = time_split(event_df, SPLIT_DATE, date_col=DATE_COL)
    train, val = time_val_split(train_all, VAL_TAIL_FRAC, date_col=DATE_COL)

    print_split_sizes(event_df, train, val, test, date_col=DATE_COL)
    print("seq_len:", SEQ_LEN, "| n_features:", len(base_cols))
    print("device:", DEVICE)
    print("label_col:", "label")
    print("class_order:", CLASS_ORDER)

    def encode_labels(series: pd.Series) -> np.ndarray:
        y_str = series.astype(str)
        cat = pd.Categorical(y_str, categories=list(CLASS_ORDER), ordered=True)
        if (cat.codes < 0).any():
            bad = sorted(set(y_str[cat.codes < 0]))
            raise ValueError(f"Found labels not in class_order: {bad}")
        return cat.codes.astype(np.int64)

    def build_xy(part: pd.DataFrame, split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs: list[np.ndarray] = []
        ms: list[np.ndarray] = []
        ys_raw: list[str] = []

        skipped_no_ticker = 0
        missing_col_counter: Counter[str] = Counter()

        for row in part.itertuples(index=False):
            t = str(getattr(row, "ticker"))
            if t not in final_data:
                skipped_no_ticker += 1
                continue

            df_t = final_data[t]

            missing = [c for c in base_cols if c not in df_t.columns]
            if missing:
                missing_col_counter.update(missing)

            fd = pd.Timestamp(getattr(row, "feature_day")).normalize()
            x, m = build_sequence_and_padmask(df_t, fd, base_cols, SEQ_LEN)
            xs.append(x)
            ms.append(m)
            ys_raw.append(str(getattr(row, "label")))

        print(f"[INFO] {split_name}: built={len(xs)} | skipped_missing_ticker={skipped_no_ticker}")
        if missing_col_counter:
            print(f"[INFO] {split_name}: top missing base columns (count over events): {missing_col_counter.most_common(10)}")

        if not xs:
            x0 = np.zeros((0, SEQ_LEN, len(base_cols)), dtype=np.float32)
            m0 = np.ones((0, SEQ_LEN), dtype=bool)
            y0 = np.zeros((0,), dtype=np.int64)
            return x0, m0, y0

        y = encode_labels(pd.Series(ys_raw))
        return np.stack(xs, axis=0), np.stack(ms, axis=0), y

    x_tr_raw, m_tr, y_tr = build_xy(train, "train")
    x_va_raw, m_va, y_va = build_xy(val, "val") if len(val) else (
        np.zeros((0, SEQ_LEN, len(base_cols)), dtype=np.float32),
        np.ones((0, SEQ_LEN), dtype=bool),
        np.zeros((0,), dtype=np.int64),
    )
    x_te_raw, m_te, y_te = build_xy(test, "test")

    if len(y_tr) == 0 or len(y_te) == 0:
        raise RuntimeError("No train/test sequences built. Check tickers and feature_day alignment.")

    imp, scaler = fit_preprocessing(x_tr_raw, m_tr)
    x_tr = apply_preprocessing(x_tr_raw, imp, scaler)
    x_va = apply_preprocessing(x_va_raw, imp, scaler) if len(y_va) else x_va_raw.astype(np.float32)
    x_te = apply_preprocessing(x_te_raw, imp, scaler)

    class_names = list(CLASS_ORDER)

    model = TransformerClassifier(
        in_dim=len(base_cols),
        d_model=D_MODEL,
        seq_len=SEQ_LEN,
        n_head=N_HEAD,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        num_layers=NUM_LAYERS,
        num_classes=len(class_names),
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=float(LR), weight_decay=float(WEIGHT_DECAY))

    train_ds = EventSeqMaskDatasetMC(x_tr, m_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    val_loader = None
    if len(y_va):
        va_ds = EventSeqMaskDatasetMC(x_va, m_va, y_va)
        val_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    test_ds = EventSeqMaskDatasetMC(x_te, m_te, y_te)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    if USE_CLASS_WEIGHTS:
        w = compute_class_weights(y_tr, num_classes=len(class_names))
        print("class_weights:", {class_names[i]: float(w[i]) for i in range(len(w))})
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))
    else:
        loss_fn = nn.CrossEntropyLoss()

    best_state: dict[str, torch.Tensor] | None = None
    best_val_acc = -1.0
    patience_left = int(PATIENCE)

    for epoch in range(1, int(MAX_EPOCHS) + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, DEVICE)

        if val_loader is None:
            print(f"epoch {epoch:03d} | train_loss={tr_loss:.5f}")
            continue

        va_proba = predict_proba_multiclass(model, val_loader, DEVICE)
        va_pred = va_proba.argmax(axis=1)
        va_acc = float((va_pred == y_va).mean()) if len(y_va) else float("nan")
        print(f"epoch {epoch:03d} | train_loss={tr_loss:.5f} | val_acc={va_acc:.4f}")

        if np.isfinite(va_acc) and va_acc > best_val_acc + 1e-6:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(PATIENCE)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[INFO] early stopping at epoch {epoch} (best_val_acc={best_val_acc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_eval_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    eval_split_multiclass("train", model, train_eval_loader, y_tr, class_names)
    if val_loader is not None:
        eval_split_multiclass("val", model, val_loader, y_va, class_names)
    eval_split_multiclass("test", model, test_loader, y_te, class_names)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "state_dict": model.state_dict(),
        "imputer": imp,
        "scaler": scaler,
        "base_feature_cols": base_cols,
        "class_names": class_names,
        "class_order": CLASS_ORDER,
        "date_col": DATE_COL,
        "split_date": SPLIT_DATE,
        "val_tail_frac": float(VAL_TAIL_FRAC),
        "data_path_event_table": str(EVENT_TABLE),
        "data_path_final_data": str(FINAL_DATA),
        "seq_len": int(SEQ_LEN),
        "device_hint": DEVICE,
        "hyperparams": {
            "d_model": int(D_MODEL),
            "n_head": int(N_HEAD),
            "num_layers": int(NUM_LAYERS),
            "ff_dim": int(FF_DIM),
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

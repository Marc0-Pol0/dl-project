from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
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
OUT = Path("networks/lstm_earnings_model.joblib")

DATE_COL = "earnings_day"
SPLIT_DATE = "2025-05-01"
VAL_TAIL_FRAC = 0.15

RANDOM_STATE = 0
CLASS_ORDER = ["heavy_down", "down", "neutral", "up", "heavy_up"]

SEQ_LEN = 30

HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.5

LR = 5e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 1000
PATIENCE = 100

USE_CLASS_WEIGHTS = True

RAW_SENTIMENT_COLS = ("positive", "negative", "neutral")
SENT_FEATURES = ("pos_frac", "neg_frac", "neu_frac", "sent_net")
SENT_EPS = 1e-6


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
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Expected DatetimeIndex, got {type(df.index)}")

    out = df.sort_index()
    if out.index.tz is not None:
        out = out.copy()
        out.index = out.index.tz_convert(None)

    out = out.copy()
    out.index = out.index.normalize()
    return out


def base_feature_names_from_event_table(event_df: pd.DataFrame) -> list[str]:
    """Derive base per-day feature names from event-table f_* columns (strip 'f_')."""
    f_cols = pick_feature_columns(event_df, prefixes=("f_",))
    f_cols.sort()
    return [c[len("f_") :] for c in f_cols]


def build_feature_list(event_df: pd.DataFrame) -> list[str]:
    """
    Per-day feature list for sequences:
      - base daily features from f_* (strip 'f_')
      - engineered sentiment fractions + net (derived from raw daily sentiment cols)
    """
    base_cols = base_feature_names_from_event_table(event_df)
    cols = set(base_cols)
    cols.update(SENT_FEATURES)
    return sorted(cols)


def _add_engineered_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure engineered sentiment columns exist: pos_frac, neg_frac, neu_frac, sent_net.
    If raw sentiment cols are missing, engineered cols will be NaN.
    """
    out = df.copy()

    if all(c in out.columns for c in RAW_SENTIMENT_COLS):
        pos = out["positive"].astype(float)
        neg = out["negative"].astype(float)
        neu = out["neutral"].astype(float)

        tot = pos + neg + neu
        denom = tot + float(SENT_EPS)

        out["pos_frac"] = pos / denom
        out["neg_frac"] = neg / denom
        out["neu_frac"] = neu / denom
        out["sent_net"] = (pos - neg) / denom
    else:
        for c in SENT_FEATURES:
            if c not in out.columns:
                out[c] = np.nan

    return out


def build_sequence_raw(
    per_ticker_df: pd.DataFrame,
    feature_day: pd.Timestamp,
    cols: list[str],
    max_len: int,
) -> tuple[np.ndarray, int]:
    """
    Build a variable-length [L, F] sequence ending at feature_day (inclusive), with L <= max_len.
    If feature_day is not present, use nearest previous index day.
    Returns (sequence, length). If no history exists, returns (empty, 0).
    """
    df = _add_engineered_sentiment(per_ticker_df)
    idx = df.index

    d = pd.Timestamp(feature_day).normalize()
    if d not in idx:
        pos = int(idx.searchsorted(d, side="right")) - 1
        if pos < 0:
            return np.zeros((0, len(cols)), dtype=np.float32), 0
        d = idx[pos]

    end_pos = int(idx.get_loc(d))
    start_pos = max(0, end_pos - (max_len - 1))
    window = df.iloc[start_pos : end_pos + 1]

    x = window.reindex(columns=cols).to_numpy(dtype=np.float32)
    return x, int(x.shape[0])


def fit_preprocessing(seqs: list[np.ndarray], lengths: list[int]) -> tuple[SimpleImputer, StandardScaler]:
    """
    Fit imputer + scaler on training sequences using ONLY real time steps (no padding).
    seqs: list of [L_i, F]
    lengths: list of L_i
    """
    if not seqs:
        raise ValueError("No training sequences to fit preprocessing.")

    feats = int(seqs[0].shape[1])
    total_steps = int(sum(lengths))
    if total_steps <= 0:
        raise ValueError("All training sequences are empty; cannot fit preprocessing.")

    flat = np.zeros((total_steps, feats), dtype=np.float32)
    cur = 0
    for x, L in zip(seqs, lengths, strict=True):
        if L <= 0:
            continue
        flat[cur : cur + L] = x[:L]
        cur += L

    imp = SimpleImputer(strategy="median")
    flat_imp = imp.fit_transform(flat)

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(flat_imp)

    return imp, scaler


def apply_preprocessing_list(seqs: list[np.ndarray], imp: SimpleImputer, scaler: StandardScaler) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for x in seqs:
        if x.size == 0:
            out.append(x.astype(np.float32, copy=False))
            continue
        x2 = imp.transform(x)
        x2 = scaler.transform(x2)
        out.append(np.asarray(x2, dtype=np.float32))
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


class EventSequenceDatasetMC(Dataset):
    def __init__(self, seqs: list[np.ndarray], lengths: list[int], y: np.ndarray) -> None:
        if len(seqs) != len(lengths) or len(seqs) != len(y):
            raise ValueError("Length mismatch between seqs, lengths, and y.")

        self.seqs = seqs
        self.lengths = lengths
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int):
        return self.seqs[idx], int(self.lengths[idx]), int(self.y[idx])


def collate_padded(batch: list[tuple[np.ndarray, int, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pads sequences on the RIGHT to max length in batch.
    Returns:
      x_pad: (B, T_max, F) float32
      lengths: (B,) int64
      y: (B,) int64
    """
    seqs, lens, ys = zip(*batch)
    lens_arr = np.asarray(lens, dtype=np.int64)
    ys_arr = np.asarray(ys, dtype=np.int64)

    b = len(seqs)
    f = int(seqs[0].shape[1]) if b > 0 else 0
    t_max = int(max(lens_arr)) if b > 0 else 0

    x_pad = np.zeros((b, t_max, f), dtype=np.float32)
    for i, (x, L) in enumerate(zip(seqs, lens_arr, strict=True)):
        if L > 0:
            x_pad[i, :L, :] = x[:L, :]

    return (
        torch.from_numpy(x_pad),
        torch.from_numpy(lens_arr),
        torch.from_numpy(ys_arr),
    )


class LSTMClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int, num_layers: int, dropout: float, out_dim: int) -> None:
        super().__init__()
        do = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=int(in_dim),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=do,
            batch_first=True,
        )
        self.head = nn.Linear(int(hidden_size), int(out_dim))

    def forward(self, x_pad: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x_pad: (B, T, F) padded on the right
        lengths: (B,) lengths of each sequence (>=1)
        """
        lengths_cpu = lengths.detach().cpu()
        packed = pack_padded_sequence(x_pad, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers, B, hidden_size) -> take last layer
        last_h = h_n[-1]
        return self.head(last_h)  # (B, out_dim)


@torch.no_grad()
def predict_proba_multiclass(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    for x_pad, lengths, _ in loader:
        x_pad = x_pad.to(device)
        lengths = lengths.to(device)
        logits = model(x_pad, lengths)  # (B,K)
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
    for x_pad, lengths, yb in loader:
        x_pad = x_pad.to(device)
        lengths = lengths.to(device)
        yb = yb.to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(x_pad, lengths)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()

        bs = int(x_pad.shape[0])
        total += float(loss.item()) * bs
        n += bs

    return total / max(1, n)


def eval_split_multiclass(
    name: str,
    model: nn.Module,
    loader: DataLoader,
    y_true: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    if len(y_true) == 0:
        print(f"\n[{name}] empty")
        return {"accuracy": float("nan")}

    proba = predict_proba_multiclass(model, loader, device=DEVICE)
    print(f"\n[{name}]")
    return eval_multiclass(np.asarray(y_true, dtype=int), proba, class_names=class_names)


def main() -> None:
    set_seeds(RANDOM_STATE)

    event_df = read_table(EVENT_TABLE)
    event_df = ensure_sorted_datetime(event_df, DATE_COL)

    event_df["feature_day"] = pd.to_datetime(event_df["feature_day"], errors="coerce")
    event_df = event_df.dropna(subset=["feature_day"])

    seq_cols = build_feature_list(event_df)
    if not seq_cols:
        raise ValueError("No per-day base features found (expected f_* columns).")

    final_data: dict[str, pd.DataFrame] = load_pickle(FINAL_DATA)
    final_data = {k: ensure_datetime_index(v) for k, v in final_data.items()}

    train_all, test = time_split(event_df, SPLIT_DATE, date_col=DATE_COL)
    train, val = time_val_split(train_all, VAL_TAIL_FRAC, date_col=DATE_COL)

    print_split_sizes(event_df, train, val, test, date_col=DATE_COL)
    print("seq_len:", SEQ_LEN, "| n_features:", len(seq_cols))
    print("device:", DEVICE)
    print("label_col:", "label")
    print("class_order:", CLASS_ORDER)

    def build_xy(part: pd.DataFrame, split_name: str) -> tuple[list[np.ndarray], list[int], np.ndarray, list[str]]:
        seqs: list[np.ndarray] = []
        lengths: list[int] = []
        ys_raw: list[str] = []

        skipped_no_ticker = 0
        skipped_empty_seq = 0
        missing_col_counter: Counter[str] = Counter()

        for row in part.itertuples(index=False):
            t = str(getattr(row, "ticker"))
            if t not in final_data:
                skipped_no_ticker += 1
                continue

            df_t = final_data[t]

            base_cols = [c for c in seq_cols if c not in SENT_FEATURES]
            missing = [c for c in base_cols if c not in df_t.columns]
            if missing:
                missing_col_counter.update(missing)

            fd = pd.Timestamp(getattr(row, "feature_day")).normalize()
            x, L = build_sequence_raw(df_t, fd, seq_cols, SEQ_LEN)
            if L <= 0:
                skipped_empty_seq += 1
                continue

            seqs.append(x)
            lengths.append(L)
            ys_raw.append(str(getattr(row, "label")))

        print(
            f"[INFO] {split_name}: built={len(seqs)} | skipped_missing_ticker={skipped_no_ticker} | skipped_empty_seq={skipped_empty_seq}"
        )

        if missing_col_counter:
            top = missing_col_counter.most_common(10)
            print(f"[INFO] {split_name}: top missing base columns (count over events): {top}")

        if not seqs:
            return [], [], np.zeros((0,), dtype=np.int64), list(CLASS_ORDER)

        y_ser = pd.Series(ys_raw, dtype="string")
        cat = pd.Categorical(y_ser.astype(str), categories=list(CLASS_ORDER), ordered=True)
        if (cat.codes < 0).any():
            bad = sorted(set(y_ser.astype(str)[cat.codes < 0]))
            raise ValueError(f"Found labels not in class_order: {bad}")

        y = cat.codes.astype(np.int64)
        class_names = list(CLASS_ORDER)
        return seqs, lengths, y, class_names

    tr_seqs_raw, tr_lengths, y_tr, class_names = build_xy(train, "train")
    va_seqs_raw, va_lengths, y_va, _ = build_xy(val, "val") if len(val) else ([], [], np.zeros((0,), dtype=np.int64), class_names)
    te_seqs_raw, te_lengths, y_te, _ = build_xy(test, "test")

    if class_names != CLASS_ORDER:
        raise RuntimeError(f"class_names={class_names} != CLASS_ORDER={CLASS_ORDER}")

    imp, scaler = fit_preprocessing(tr_seqs_raw, tr_lengths)
    tr_seqs = apply_preprocessing_list(tr_seqs_raw, imp, scaler)
    va_seqs = apply_preprocessing_list(va_seqs_raw, imp, scaler) if len(y_va) else []
    te_seqs = apply_preprocessing_list(te_seqs_raw, imp, scaler)

    out_dim = len(CLASS_ORDER)
    model = LSTMClassifier(
        in_dim=len(seq_cols),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        out_dim=out_dim,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if USE_CLASS_WEIGHTS:
        w = compute_class_weights(y_tr, num_classes=len(CLASS_ORDER))
        print("class_weights:", {class_names[i]: float(w[i]) for i in range(len(w))})
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))
    else:
        loss_fn = nn.CrossEntropyLoss()

    train_ds = EventSequenceDatasetMC(tr_seqs, tr_lengths, y_tr)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_padded,
    )

    val_loader = None
    if len(y_va):
        val_ds = EventSequenceDatasetMC(va_seqs, va_lengths, y_va)
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_padded,
        )

    test_ds = EventSequenceDatasetMC(te_seqs, te_lengths, y_te)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_padded,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_acc = -1.0
    patience_left = int(PATIENCE)

    for epoch in range(1, int(MAX_EPOCHS) + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, DEVICE)

        if val_loader is None:
            print(f"epoch {epoch:03d} | train_loss={tr_loss:.5f}")
            continue

        proba_va = predict_proba_multiclass(model, val_loader, device=DEVICE)
        pred_va = proba_va.argmax(axis=1)
        va_acc = float((pred_va == y_va).mean()) if len(y_va) else float("nan")
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

    train_eval_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_padded,
    )

    eval_split_multiclass("train", model, train_eval_loader, y_tr, class_names)
    if val_loader is not None:
        eval_split_multiclass("val", model, val_loader, y_va, class_names)
    eval_split_multiclass("test", model, test_loader, y_te, class_names)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "state_dict": model.state_dict(),
        "imputer": imp,
        "scaler": scaler,
        "seq_cols": seq_cols,
        "class_names": class_names,
        "class_order": CLASS_ORDER,
        "date_col": DATE_COL,
        "split_date": SPLIT_DATE,
        "val_tail_frac": float(VAL_TAIL_FRAC),
        "data_path_event_table": str(EVENT_TABLE),
        "data_path_final_data": str(FINAL_DATA),
        "raw_sentiment_cols": RAW_SENTIMENT_COLS,
        "engineered_sentiment_cols": SENT_FEATURES,
        "seq_len": int(SEQ_LEN),
        "device_hint": DEVICE,
        "hyperparams": {
            "hidden_size": int(HIDDEN_SIZE),
            "num_layers": int(NUM_LAYERS),
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

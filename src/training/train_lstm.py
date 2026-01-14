from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
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
CLASS_ORDER = ["down", "neutral", "up"]

SEQ_LEN = 30

HIDDEN_SIZE = 32
NUM_LAYERS = 1
LSTM_DROPOUT = 0.2

STATIC_EMBED_DIM = 32
STATIC_DROPOUT = 0.4

HEAD_DROPOUT = 0.5

LR = 3e-4
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 64
MAX_EPOCHS = 400
PATIENCE = 30
GRAD_CLIP_NORM = 1.0

USE_CLASS_WEIGHTS = True

RAW_SENTIMENT_COLS = ("positive", "negative", "neutral")
SENT_FEATURES = ("pos_frac", "neg_frac", "neu_frac", "sent_net")
SENT_EPS = 1e-6

PROBA_EPS = 1e-7


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
    f_cols = pick_feature_columns(event_df, prefixes=("f_",))
    f_cols.sort()
    return [c[len("f_") :] for c in f_cols]


def build_seq_feature_list(event_df: pd.DataFrame) -> list[str]:
    base_cols = base_feature_names_from_event_table(event_df)
    cols = set(base_cols)
    cols.update(SENT_FEATURES)
    return sorted(cols)


def _add_engineered_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if all(c in out.columns for c in RAW_SENTIMENT_COLS):
        pos = pd.to_numeric(out["positive"], errors="coerce")
        neg = pd.to_numeric(out["negative"], errors="coerce")
        neu = pd.to_numeric(out["neutral"], errors="coerce")
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
    w = window.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
    x = w.to_numpy(dtype=np.float32, copy=True)
    return x, int(x.shape[0])


def fit_seq_preprocessing(seqs: list[np.ndarray], lengths: list[int]) -> tuple[SimpleImputer, StandardScaler]:
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


def apply_seq_preprocessing_list(seqs: list[np.ndarray], imp: SimpleImputer, scaler: StandardScaler) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for x in seqs:
        if x.size == 0:
            out.append(x.astype(np.float32, copy=False))
            continue
        x2 = imp.transform(x)
        x2 = scaler.transform(x2)
        x2 = np.nan_to_num(x2, nan=0.0, posinf=0.0, neginf=0.0)
        out.append(np.asarray(x2, dtype=np.float32))
    return out


def fit_static_preprocessing(X: np.ndarray) -> tuple[SimpleImputer, StandardScaler]:
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_imp)
    return imp, scaler


def apply_static_preprocessing(X: np.ndarray, imp: SimpleImputer, scaler: StandardScaler) -> np.ndarray:
    X2 = imp.transform(X)
    X2 = scaler.transform(X2)
    X2 = np.nan_to_num(X2, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(X2, dtype=np.float32)


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.asarray(y_train).astype(int)
    counts = np.bincount(y, minlength=int(num_classes)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    n = float(len(y))
    k = float(num_classes)
    w = n / (k * counts)
    return w.astype(np.float32)


def sanitize_proba(p: np.ndarray, eps: float = PROBA_EPS) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, eps, 1.0 - eps)
    s = p.sum(axis=1, keepdims=True)
    s = np.where(s <= 0.0, 1.0, s)
    return p / s


def show_label_dist(name: str, y: np.ndarray) -> None:
    c = Counter(np.asarray(y, dtype=int).tolist())
    print(name, {CLASS_ORDER[i]: int(c.get(i, 0)) for i in range(len(CLASS_ORDER))})


def get_static_feature_columns(event_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Static per-event features from event table:
      - f_* snapshot columns
      - sent_* aggregate columns (exclude sent_seq_*)
      - event_type one-hot (handled separately but we need levels)
    """
    f_cols = pick_feature_columns(event_df, prefixes=("f_",), exclude_prefixes=())
    sent_cols = pick_feature_columns(event_df, prefixes=("sent_",), exclude_prefixes=("sent_seq_",))
    static_numeric_cols = sorted(set(f_cols + sent_cols))
    return static_numeric_cols, ["event_type"]


def build_static_matrix(event_df_part: pd.DataFrame, static_num_cols: list[str]) -> np.ndarray:
    X_parts: list[np.ndarray] = []
    for c in static_num_cols:
        s = pd.to_numeric(event_df_part[c], errors="coerce") if c in event_df_part.columns else pd.Series([np.nan] * len(event_df_part))
        X_parts.append(s.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1))
    if not X_parts:
        return np.zeros((len(event_df_part), 0), dtype=np.float32)
    return np.concatenate(X_parts, axis=1)


def build_event_type_onehot(event_df_part: pd.DataFrame, categories: list[str]) -> np.ndarray:
    if "event_type" not in event_df_part.columns:
        return np.zeros((len(event_df_part), len(categories)), dtype=np.float32)

    ser = event_df_part["event_type"].astype(str)
    cat = pd.Categorical(ser, categories=categories, ordered=True)

    codes = np.asarray(cat.codes, dtype=np.int64)

    oh = np.zeros((len(event_df_part), len(categories)), dtype=np.float32)
    ok = codes >= 0
    oh[np.arange(len(event_df_part))[ok], codes[ok]] = 1.0
    return oh


def add_position_feature(x_pad: torch.Tensor) -> torch.Tensor:
    """
    Append a simple position feature t/T in [0,1] for each timestep.
    x_pad: (B,T,F) -> (B,T,F+1)
    """
    b, t, _ = x_pad.shape
    pos = torch.linspace(0.0, 1.0, steps=t, device=x_pad.device).view(1, t, 1).expand(b, t, 1)
    return torch.cat([x_pad, pos], dim=-1)


class EventHybridDataset(Dataset):
    def __init__(self, seqs: list[np.ndarray], lengths: list[int], x_static: np.ndarray, y: np.ndarray) -> None:
        if len(seqs) != len(lengths) or len(seqs) != len(y) or len(seqs) != len(x_static):
            raise ValueError("Length mismatch between seqs, lengths, x_static, and y.")
        self.seqs = seqs
        self.lengths = lengths
        self.x_static = np.asarray(x_static, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int):
        return self.seqs[idx], int(self.lengths[idx]), self.x_static[idx], int(self.y[idx])


def collate_hybrid(batch: list[tuple[np.ndarray, int, np.ndarray, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      x_pad: (B, T_max, F_seq)
      lengths: (B,)
      x_static: (B, F_static)
      y: (B,)
    """
    seqs, lens, xstat, ys = zip(*batch)
    lens_arr = np.asarray(lens, dtype=np.int64)
    ys_arr = np.asarray(ys, dtype=np.int64)

    b = len(seqs)
    f = int(seqs[0].shape[1]) if b > 0 else 0
    t_max = int(max(lens_arr)) if b > 0 else 0

    x_pad = np.zeros((b, t_max, f), dtype=np.float32)
    for i, (x, L) in enumerate(zip(seqs, lens_arr, strict=True)):
        if L > 0:
            x_pad[i, :L, :] = x[:L, :]

    x_static = np.stack(xstat, axis=0).astype(np.float32, copy=False)

    return (
        torch.from_numpy(x_pad),
        torch.from_numpy(lens_arr),
        torch.from_numpy(x_static),
        torch.from_numpy(ys_arr),
    )


class StaticEmbed(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMHybridClassifier(nn.Module):
    def __init__(
        self,
        seq_in_dim: int,
        hidden_size: int,
        num_layers: int,
        lstm_dropout: float,
        static_in_dim: int,
        static_embed_dim: int,
        static_dropout: float,
        head_dropout: float,
        out_dim: int,
    ) -> None:
        super().__init__()
        do = float(lstm_dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=int(seq_in_dim),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=do,
            batch_first=True,
        )
        self.static_embed = StaticEmbed(static_in_dim, static_embed_dim, dropout=static_dropout)
        self.dropout = nn.Dropout(float(head_dropout))
        self.head = MLPHead(hidden_size + static_embed_dim, out_dim, dropout=head_dropout)

    def forward(self, x_pad: torch.Tensor, lengths: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        lengths_cpu = lengths.detach().cpu()
        packed = pack_padded_sequence(x_pad, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        last_h = h_n[-1]  # (B,H)

        s = self.static_embed(x_static)  # (B,S)

        h = torch.cat([last_h, s], dim=-1)
        h = self.dropout(h)
        return self.head(h)


@torch.no_grad()
def predict_proba_multiclass(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    for x_pad, lengths, x_static, _ in loader:
        x_pad = x_pad.to(device)
        lengths = lengths.to(device)
        x_static = x_static.to(device)

        x_pad = add_position_feature(x_pad)
        logits = model(x_pad, lengths, x_static)
        p = torch.softmax(logits, dim=-1)
        probs.append(p.detach().cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, 0), dtype=np.float32)


def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, loss_fn: nn.Module, device: str) -> float:
    model.train()
    total = 0.0
    n = 0
    for x_pad, lengths, x_static, yb in loader:
        x_pad = x_pad.to(device)
        lengths = lengths.to(device)
        x_static = x_static.to(device)
        yb = yb.to(device)

        x_pad = add_position_feature(x_pad)

        opt.zero_grad(set_to_none=True)
        logits = model(x_pad, lengths, x_static)
        loss = loss_fn(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        opt.step()

        bs = int(x_pad.shape[0])
        total += float(loss.item()) * bs
        n += bs

    return total / max(1, n)


def eval_split(name: str, model: nn.Module, loader: DataLoader, y_true: np.ndarray, class_names: list[str], model_name: str) -> dict[str, float]:
    if len(y_true) == 0:
        print(f"\n[{name}] empty")
        return {"accuracy": float("nan")}
    proba = predict_proba_multiclass(model, loader, device=DEVICE)
    print(f"\n[{name}]")
    return eval_multiclass(np.asarray(y_true, dtype=int), proba, class_names=class_names, model_name=model_name)


def main() -> None:
    set_seeds(RANDOM_STATE)

    event_df = read_table(EVENT_TABLE)
    event_df = ensure_sorted_datetime(event_df, DATE_COL)

    event_df["feature_day"] = pd.to_datetime(event_df["feature_day"], errors="coerce")
    event_df = event_df.dropna(subset=["feature_day"])

    seq_cols = build_seq_feature_list(event_df)
    if not seq_cols:
        raise ValueError("No per-day base features found (expected f_* columns).")

    static_num_cols, _ = get_static_feature_columns(event_df)

    # categories for event_type one-hot (must be fixed from TRAIN ONLY ideally; using full is ok since it's only categories)
    event_type_categories = sorted(event_df["event_type"].astype(str).unique().tolist()) if "event_type" in event_df.columns else ["BMO", "AMC", "INTRADAY"]

    final_data: dict[str, pd.DataFrame] = load_pickle(FINAL_DATA)
    final_data = {k: ensure_datetime_index(v) for k, v in final_data.items()}

    train_all, test = time_split(event_df, SPLIT_DATE, date_col=DATE_COL)
    train, val = time_val_split(train_all, VAL_TAIL_FRAC, date_col=DATE_COL)

    print_split_sizes(event_df, train, val, test, date_col=DATE_COL)
    print("seq_len:", SEQ_LEN, "| n_seq_features:", len(seq_cols), "(+1 pos feature)")
    print("n_static_numeric:", len(static_num_cols), "| event_type categories:", event_type_categories)
    print("device:", DEVICE)
    print("class_order:", CLASS_ORDER)

    def encode_labels_from_part(part: pd.DataFrame) -> np.ndarray:
        y_ser = part["label"].astype(str)
        cat = pd.Categorical(y_ser, categories=list(CLASS_ORDER), ordered=True)
        if (cat.codes < 0).any():
            bad = sorted(set(y_ser[cat.codes < 0]))
            raise ValueError(f"Found labels not in class_order: {bad}")
        return cat.codes.astype(np.int64)

    def build_sequences_for_part(part: pd.DataFrame) -> tuple[list[np.ndarray], list[int]]:
        seqs: list[np.ndarray] = []
        lengths: list[int] = []
        for row in part.itertuples(index=False):
            t = str(getattr(row, "ticker"))
            df_t = final_data[t]
            fd = pd.Timestamp(getattr(row, "feature_day")).normalize()
            x, L = build_sequence_raw(df_t, fd, seq_cols, SEQ_LEN)
            seqs.append(x)
            lengths.append(L)
        return seqs, lengths

    # Build raw sequences (aligned 1:1 with the part rows)
    tr_seqs_raw, tr_lengths = build_sequences_for_part(train)
    va_seqs_raw, va_lengths = build_sequences_for_part(val) if len(val) else ([], [])
    te_seqs_raw, te_lengths = build_sequences_for_part(test)

    # Filter out any empty sequences (rare with your current setup, but keep safe)
    def filter_nonempty(part: pd.DataFrame, seqs: list[np.ndarray], lengths: list[int]) -> tuple[pd.DataFrame, list[np.ndarray], list[int]]:
        keep = np.array(lengths, dtype=np.int64) > 0
        part2 = part.iloc[np.where(keep)[0]].copy()
        seqs2 = [s for s, k in zip(seqs, keep, strict=True) if bool(k)]
        lengths2 = [int(L) for L, k in zip(lengths, keep, strict=True) if bool(k)]
        return part2, seqs2, lengths2

    train, tr_seqs_raw, tr_lengths = filter_nonempty(train, tr_seqs_raw, tr_lengths)
    if len(val):
        val, va_seqs_raw, va_lengths = filter_nonempty(val, va_seqs_raw, va_lengths)
    test, te_seqs_raw, te_lengths = filter_nonempty(test, te_seqs_raw, te_lengths)

    y_tr = encode_labels_from_part(train)
    y_va = encode_labels_from_part(val) if len(val) else np.zeros((0,), dtype=np.int64)
    y_te = encode_labels_from_part(test)

    show_label_dist("train_dist", y_tr)
    if len(y_va):
        show_label_dist("val_dist", y_va)
    show_label_dist("test_dist", y_te)

    # ----- STATIC FEATURES (D) -----
    Xs_tr_num = build_static_matrix(train, static_num_cols)
    Xs_va_num = build_static_matrix(val, static_num_cols) if len(val) else np.zeros((0, Xs_tr_num.shape[1]), dtype=np.float32)
    Xs_te_num = build_static_matrix(test, static_num_cols)

    Xs_tr_type = build_event_type_onehot(train, event_type_categories)
    Xs_va_type = build_event_type_onehot(val, event_type_categories) if len(val) else np.zeros((0, len(event_type_categories)), dtype=np.float32)
    Xs_te_type = build_event_type_onehot(test, event_type_categories)

    Xs_tr_raw = np.concatenate([Xs_tr_num, Xs_tr_type], axis=1).astype(np.float32, copy=False)
    Xs_va_raw = np.concatenate([Xs_va_num, Xs_va_type], axis=1).astype(np.float32, copy=False) if len(val) else np.zeros((0, Xs_tr_raw.shape[1]), dtype=np.float32)
    Xs_te_raw = np.concatenate([Xs_te_num, Xs_te_type], axis=1).astype(np.float32, copy=False)

    static_imp, static_scaler = fit_static_preprocessing(Xs_tr_raw)
    Xs_tr = apply_static_preprocessing(Xs_tr_raw, static_imp, static_scaler)
    Xs_va = apply_static_preprocessing(Xs_va_raw, static_imp, static_scaler) if len(val) else Xs_va_raw.astype(np.float32, copy=False)
    Xs_te = apply_static_preprocessing(Xs_te_raw, static_imp, static_scaler)

    # ----- SEQUENCE PREPROCESSING -----
    seq_imp, seq_scaler = fit_seq_preprocessing(tr_seqs_raw, tr_lengths)
    tr_seqs = apply_seq_preprocessing_list(tr_seqs_raw, seq_imp, seq_scaler)
    va_seqs = apply_seq_preprocessing_list(va_seqs_raw, seq_imp, seq_scaler) if len(val) else []
    te_seqs = apply_seq_preprocessing_list(te_seqs_raw, seq_imp, seq_scaler)

    # ----- MODEL -----
    out_dim = len(CLASS_ORDER)
    seq_in_dim = len(seq_cols) + 1  # + position feature
    static_in_dim = int(Xs_tr.shape[1])

    model = LSTMHybridClassifier(
        seq_in_dim=seq_in_dim,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        lstm_dropout=LSTM_DROPOUT,
        static_in_dim=static_in_dim,
        static_embed_dim=STATIC_EMBED_DIM,
        static_dropout=STATIC_DROPOUT,
        head_dropout=HEAD_DROPOUT,
        out_dim=out_dim,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    weight_t = None
    if USE_CLASS_WEIGHTS:
        w = compute_class_weights(y_tr, num_classes=len(CLASS_ORDER))
        print("class_weights:", {CLASS_ORDER[i]: float(w[i]) for i in range(len(w))})
        weight_t = torch.tensor(w, dtype=torch.float32, device=DEVICE)

    loss_fn = nn.CrossEntropyLoss(weight=weight_t)

    train_ds = EventHybridDataset(tr_seqs, tr_lengths, Xs_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, collate_fn=collate_hybrid)

    val_loader = None
    if len(y_va):
        val_ds = EventHybridDataset(va_seqs, va_lengths, Xs_va, y_va)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_hybrid)

    test_ds = EventHybridDataset(te_seqs, te_lengths, Xs_te, y_te)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_hybrid)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_ll = float("inf")
    patience_left = int(PATIENCE)
    labels = list(range(len(CLASS_ORDER)))

    for epoch in range(1, int(MAX_EPOCHS) + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, DEVICE)

        if val_loader is None:
            print(f"epoch {epoch:03d} | train_loss={tr_loss:.5f}")
            continue

        proba_va = predict_proba_multiclass(model, val_loader, device=DEVICE)
        proba_va_s = sanitize_proba(proba_va)
        pred_va = proba_va_s.argmax(axis=1)

        va_acc = float((pred_va == y_va).mean())
        va_ll = float(log_loss(y_va.astype(int), proba_va_s, labels=labels))
        print(f"epoch {epoch:03d} | train_loss={tr_loss:.5f} | val_acc={va_acc:.4f} | val_logloss={va_ll:.4f}")

        if np.isfinite(va_ll) and va_ll < best_val_ll - 1e-6:
            best_val_ll = va_ll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(PATIENCE)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[INFO] early stopping at epoch {epoch} (best_val_logloss={best_val_ll:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_eval_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_hybrid)

    eval_split("train", model, train_eval_loader, y_tr, list(CLASS_ORDER), model_name="lstm_train")
    if val_loader is not None:
        eval_split("val", model, val_loader, y_va, list(CLASS_ORDER), model_name="lstm_val")
    eval_split("test", model, test_loader, y_te, list(CLASS_ORDER), model_name="lstm_test")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "state_dict": model.state_dict(),
        "seq_imputer": seq_imp,
        "seq_scaler": seq_scaler,
        "static_imputer": static_imp,
        "static_scaler": static_scaler,
        "seq_cols": seq_cols,
        "static_num_cols": static_num_cols,
        "event_type_categories": event_type_categories,
        "class_order": CLASS_ORDER,
        "date_col": DATE_COL,
        "split_date": SPLIT_DATE,
        "val_tail_frac": float(VAL_TAIL_FRAC),
        "data_path_event_table": str(EVENT_TABLE),
        "data_path_final_data": str(FINAL_DATA),
        "seq_len": int(SEQ_LEN),
        "device_hint": DEVICE,
        "hyperparams": {
            "hidden_size": int(HIDDEN_SIZE),
            "num_layers": int(NUM_LAYERS),
            "lstm_dropout": float(LSTM_DROPOUT),
            "static_embed_dim": int(STATIC_EMBED_DIM),
            "static_dropout": float(STATIC_DROPOUT),
            "head_dropout": float(HEAD_DROPOUT),
            "lr": float(LR),
            "weight_decay": float(WEIGHT_DECAY),
            "batch_size": int(BATCH_SIZE),
            "max_epochs": int(MAX_EPOCHS),
            "patience": int(PATIENCE),
            "grad_clip_norm": float(GRAD_CLIP_NORM),
            "use_class_weights": bool(USE_CLASS_WEIGHTS),
            "pos_feature": True,
        },
    }
    joblib.dump(bundle, OUT)
    print(f"\nSaved model bundle to: {OUT}")


if __name__ == "__main__":
    main()

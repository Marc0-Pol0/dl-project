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
    dict[ticker] -> DataFrame indexed by trading-day DatetimeIndex with daily features

Sequence construction:
- For each event row, build a fixed-length window of SEQ_LEN trading days ending at feature_day (inclusive).
- If history is shorter than SEQ_LEN, left-pad with NaNs (then impute).
- Per-time-step features:
    * base features from event table f_* columns (strip "f_")
    * daily sentiment columns: positive/negative/neutral
    * derived daily sentiment feature: pos_minus_neg = positive - negative

Preprocessing (fit on train only):
- Median imputation (per feature) + standardization, applied to all time steps.

Model:
- LSTM over sequences, using last hidden state -> linear -> logit
- BCEWithLogitsLoss (no pos_weight by default; mild imbalance + better calibration)
- Early stopping on validation AUC (if validation split exists)

Evaluation:
- Threshold is selected on validation set to maximize balanced accuracy (if validation exists).
  The selected threshold is then used for train/val/test reporting.
  If no validation split exists, threshold defaults to 0.5.

Outputs:
- Printed metrics on train/val/test
- Saved model bundle (.joblib): state_dict, preprocessing, feature list, configs, chosen threshold
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

# Reduced capacity for better generalization
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.3

LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10

# No pos_weight by default (calibration + mild imbalance)
USE_POS_WEIGHT = False

# ----------------------------- sentiment features -----------------------------

SENTIMENT_COLS = ("positive", "negative", "neutral")
ADD_DERIVED_POS_MINUS_NEG = True

# Threshold selection grid (on validation)
THRESH_GRID = np.linspace(0.05, 0.95, 91)


@dataclass(frozen=True)
class LSTMConfig:
    seq_len: int = 30
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
    use_pos_weight: bool = False
    random_state: int = 0
    device: str = "cpu"


# ----------------------------- utils -----------------------------

def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return __import__("pickle").load(f)


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame index is a sorted, timezone-naive DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Expected DatetimeIndex, got {type(df.index)}")
    out = df.sort_index()
    if out.index.tz is not None:
        out = out.copy()
        out.index = out.index.tz_convert(None)
    return out


def base_feature_names_from_event_table(event_df: pd.DataFrame) -> list[str]:
    """Derive base per-day feature names from event table f_* columns (strip 'f_')."""
    f_cols = pick_feature_columns(event_df, prefixes=("f_",))
    f_cols.sort()
    return [c[len("f_") :] for c in f_cols]


def build_feature_list(event_df: pd.DataFrame) -> list[str]:
    """
    Build the per-day feature list for sequences:
      - base (from f_* columns)
      - sentiment cols
      - optional derived pos_minus_neg
    """
    base_cols = base_feature_names_from_event_table(event_df)
    cols = set(base_cols)
    cols.update(SENTIMENT_COLS)
    if ADD_DERIVED_POS_MINUS_NEG:
        cols.add("pos_minus_neg")
    out = sorted(cols)
    return out


def build_sequence(
    per_ticker_df: pd.DataFrame, feature_day: pd.Timestamp, base_cols: list[str], seq_len: int
) -> np.ndarray:
    """
    Build a [T, F] sequence ending at feature_day (inclusive).
    If feature_day is not present, use nearest previous trading day.
    Left-pad with NaNs to reach seq_len.

    Adds derived sentiment column pos_minus_neg if requested and possible.
    """
    df = per_ticker_df

    # Derived daily sentiment feature
    if ADD_DERIVED_POS_MINUS_NEG and ("pos_minus_neg" in base_cols):
        if "pos_minus_neg" not in df.columns:
            if ("positive" in df.columns) and ("negative" in df.columns):
                df = df.copy()
                df["pos_minus_neg"] = df["positive"].astype(float) - df["negative"].astype(float)

    idx = df.index
    if feature_day not in idx:
        pos = int(idx.searchsorted(feature_day, side="right")) - 1
        if pos < 0:
            return np.full((seq_len, len(base_cols)), np.nan, dtype=np.float32)
        feature_day = idx[pos]

    end_pos = int(idx.get_loc(feature_day))
    start_pos = max(0, end_pos - (seq_len - 1))
    window = df.iloc[start_pos : end_pos + 1]

    x = window.reindex(columns=base_cols).to_numpy(dtype=np.float32)
    if x.shape[0] < seq_len:
        pad = np.full((seq_len - x.shape[0], x.shape[1]), np.nan, dtype=np.float32)
        x = np.vstack([pad, x])

    return x


def fit_preprocessing(x_train: np.ndarray) -> tuple[SimpleImputer, StandardScaler]:
    """Fit imputer + scaler on training sequences using all time steps: x_train [N,T,F]."""
    n, t, f = x_train.shape
    flat = x_train.reshape(n * t, f)
    imp = SimpleImputer(strategy="median")
    flat_imp = imp.fit_transform(flat)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(flat_imp)
    return imp, scaler


def apply_preprocessing(x: np.ndarray, imp: SimpleImputer, scaler: StandardScaler) -> np.ndarray:
    """Apply imputer + scaler to sequences x: [N,T,F]."""
    n, t, f = x.shape
    flat = x.reshape(n * t, f)
    flat = imp.transform(flat)
    flat = scaler.transform(flat)
    return flat.reshape(n, t, f).astype(np.float32)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy = (TPR + TNR) / 2."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    return float(0.5 * (tpr + tnr))


def choose_threshold_on_val(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Pick threshold maximizing balanced accuracy on validation."""
    best_t = 0.5
    best = -1.0
    for t in THRESH_GRID:
        pred = (p_val >= float(t)).astype(int)
        bacc = balanced_accuracy(y_val, pred)
        if bacc > best + 1e-12:
            best = bacc
            best_t = float(t)
    return float(best_t)


class EventSequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        do = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=int(in_dim),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=do,
            batch_first=True,
        )
        self.head = nn.Linear(int(hidden_size), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


@torch.no_grad()
def predict_proba(model: nn.Module, x: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    ds = torch.from_numpy(np.asarray(x, dtype=np.float32))
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, drop_last=False)
    probs: list[np.ndarray] = []
    for xb in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


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


def eval_split(name: str, model: nn.Module, x: np.ndarray, y: np.ndarray, cfg: LSTMConfig, thresh: float) -> dict[str, float]:
    if len(y) == 0:
        print(f"\n[{name}] empty")
        return {"auc": float("nan")}

    prob = predict_proba(model, x, cfg.batch_size, cfg.device)
    print(f"\n[{name}]")
    return eval_binary(np.asarray(y, dtype=int), prob, thresh=float(thresh))


# ----------------------------- main -----------------------------

def main() -> None:
    set_seeds(RANDOM_STATE)

    split_cfg = SplitConfig(split_date=SPLIT_DATE, val_tail_frac=VAL_TAIL_FRAC)

    event_df = read_table(EVENT_TABLE)
    for col in (split_cfg.label_col, split_cfg.date_col, "ticker", "feature_day"):
        if col not in event_df.columns:
            raise KeyError(f"Event table must contain '{col}'")

    event_df = ensure_sorted_datetime(event_df, split_cfg.date_col)
    event_df["feature_day"] = pd.to_datetime(event_df["feature_day"], errors="coerce")
    event_df = event_df.dropna(subset=["feature_day"])

    base_cols = build_feature_list(event_df)
    if not base_cols:
        raise ValueError("No per-day base features found (expected f_* columns).")

    final_data: dict[str, pd.DataFrame] = load_pickle(FINAL_DATA)
    final_data = {k: ensure_datetime_index(v) for k, v in final_data.items()}

    train_all, test = time_split(event_df, split_cfg.date_col, split_cfg.split_date)
    train, val = time_val_split(train_all, split_cfg.date_col, split_cfg.val_tail_frac)

    print_split_sizes(event_df, train, val, test)
    print("seq_len:", SEQ_LEN, "| n_features:", len(base_cols))
    print("device:", DEVICE)

    def build_xy(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        xs: list[np.ndarray] = []
        ys: list[int] = []
        for row in part.itertuples(index=False):
            t = str(getattr(row, "ticker"))
            if t not in final_data:
                continue
            df_t = final_data[t]
            fd = pd.Timestamp(getattr(row, "feature_day"))
            x = build_sequence(df_t, fd, base_cols, SEQ_LEN)
            xs.append(x)
            ys.append(int(getattr(row, split_cfg.label_col)))
        if not xs:
            return (
                np.zeros((0, SEQ_LEN, len(base_cols)), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64)

    x_tr_raw, y_tr = build_xy(train)
    x_va_raw, y_va = build_xy(val) if len(val) else (
        np.zeros((0, SEQ_LEN, len(base_cols)), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
    )
    x_te_raw, y_te = build_xy(test)

    if len(y_tr) == 0 or len(y_te) == 0:
        raise RuntimeError("No train/test sequences built. Check tickers and feature_day alignment.")

    imp, scaler = fit_preprocessing(x_tr_raw)
    x_tr = apply_preprocessing(x_tr_raw, imp, scaler)
    x_va = apply_preprocessing(x_va_raw, imp, scaler) if len(y_va) else x_va_raw.astype(np.float32)
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

    model = LSTMClassifier(
        in_dim=len(base_cols),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    # Loss: default no pos_weight; if enabled, compute weight on train
    if cfg.use_pos_weight:
        pos = int((y_tr == 1).sum())
        neg = int((y_tr == 0).sum())
        pos_w = float(neg / max(1, pos))
        print(f"pos_weight (neg/pos) on train: {pos_w:.4f}")
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32, device=cfg.device))
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    train_ds = EventSequenceDataset(x_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_auc = -1.0
    patience_left = int(cfg.patience)

    for epoch in range(1, int(cfg.max_epochs) + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, cfg.device)

        if len(y_va) == 0:
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

    # Threshold selection on validation
    chosen_thresh = 0.5
    if len(y_va) > 0:
        val_prob = predict_proba(model, x_va, cfg.batch_size, cfg.device)
        chosen_thresh = choose_threshold_on_val(y_va, val_prob)
        print(f"[INFO] chosen threshold on val (max balanced acc): {chosen_thresh:.3f}")
    else:
        print("[INFO] no validation split; using threshold=0.5")

    eval_split("train", model, x_tr, y_tr, cfg, chosen_thresh)
    if len(y_va):
        eval_split("val", model, x_va, y_va, cfg, chosen_thresh)
    eval_split("test", model, x_te, y_te, cfg, chosen_thresh)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "state_dict": model.state_dict(),
        "imputer": imp,
        "scaler": scaler,
        "base_feature_cols": base_cols,
        "split_cfg": split_cfg,
        "lstm_cfg": cfg,
        "threshold": float(chosen_thresh),
        "data_path_event_table": str(EVENT_TABLE),
        "data_path_final_data": str(FINAL_DATA),
        "sentiment_cols": SENTIMENT_COLS,
        "add_pos_minus_neg": bool(ADD_DERIVED_POS_MINUS_NEG),
    }
    joblib.dump(bundle, OUT)
    print(f"\nSaved model bundle to: {OUT}")


if __name__ == "__main__":
    main()

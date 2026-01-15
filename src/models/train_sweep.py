import os
import time
import random

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloaders import create_dataloader
from model import StockLSTM, StockTransformer, StockLogReg


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


class Config:
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 60
    BATCH_SIZE = 32
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.2
    PATIENCE = 8

    INPUT_SIZE = 21
    OUTPUT_SIZE = 3  # UP=0, DOWN=1, NEUTRAL=2

    DIM_FFN = 4 * HIDDEN_SIZE
    NUMBER_OF_ENCODERS = 2
    NUMBER_OF_HEADS = 4

    SEQ_LEN = 30

    # We will override these in the sweep loop
    MODEL = "attention"  # {'lstm', 'attention', 'logreg'}
    USE_SENTIMENT = False
    USE_WEIGHTED_LOSS = False
    CKPT_METRIC = "val_loss"  # {'val_loss', 'val_acc', 'val_macro_f1'}

    # Custom cost parameters
    COST_NEUTRAL = 1.0
    COST_UPDOWN = 3.0

    FIG_DIR = "./src/figures"
    NETWORK_DIR = "./networks"


def setup_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EarlyStopper:
    """Early stopping that MINIMIZES the monitored value (validation loss)."""

    def __init__(self, patience=7, min_delta=0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.counter = 0
        self.best = float("inf")

    def early_stop(self, monitored_value: float) -> bool:
        monitored_value = float(monitored_value)

        if monitored_value < self.best - self.min_delta:
            self.best = monitored_value
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def compute_custom_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_neutral: float = 1.0,
    cost_updown: float = 3.0,
) -> float:
    """
    Average misclassification cost with asymmetric penalties:
      - correct: 0
      - UP<->DOWN: cost_updown
      - (UP or DOWN)<->NEUTRAL: cost_neutral

    Labels assumed: UP=0, DOWN=1, NEUTRAL=2. Lower is better.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape

    cost = np.zeros_like(y_true, dtype=np.float32)
    correct = (y_true == y_pred)
    if correct.all():
        return 0.0

    # UP<->DOWN mistakes (0<->1), neither side is NEUTRAL (2)
    updown_mistake = (~correct) & (y_true != 2) & (y_pred != 2)
    cost[updown_mistake] = float(cost_updown)

    # neutral<->direction mistakes (exactly one side is 2)
    neutral_mistake = (~correct) & ((y_true == 2) ^ (y_pred == 2))
    cost[neutral_mistake] = float(cost_neutral)

    return float(cost.mean())


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for X_batch, Y_batch in dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float, list[int], list[int]]:
    """Returns: (avg_loss, accuracy, macro_f1, custom_cost, y_true, y_pred)"""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_predictions: list[int] = []
    all_targets: list[int] = []

    for X_batch, Y_batch in dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        total_loss += loss.item()

        predicted = outputs.argmax(dim=1)
        total_samples += Y_batch.size(0)
        correct_predictions += (predicted == Y_batch).sum().item()

        all_predictions.extend(predicted.cpu().tolist())
        all_targets.extend(Y_batch.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples if total_samples else 0.0
    macro_f1 = float(f1_score(all_targets, all_predictions, average="macro", zero_division=0))
    custom_cost = compute_custom_cost(
        y_true=np.asarray(all_targets),
        y_pred=np.asarray(all_predictions),
        cost_neutral=Config.COST_NEUTRAL,
        cost_updown=Config.COST_UPDOWN,
    )

    return avg_loss, accuracy, macro_f1, custom_cost, all_targets, all_predictions


def build_model(device: torch.device) -> nn.Module:
    if Config.MODEL == "attention":
        return StockTransformer(
            input_size=Config.INPUT_SIZE,
            d_model=Config.HIDDEN_SIZE,
            nhead=Config.NUMBER_OF_HEADS,
            num_encoder_layers=Config.NUMBER_OF_ENCODERS,
            dim_feedforward=Config.DIM_FFN,
            output_size=Config.OUTPUT_SIZE,
        ).to(device)

    if Config.MODEL == "lstm":
        return StockLSTM(
            input_size=Config.INPUT_SIZE,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            output_size=Config.OUTPUT_SIZE,
            dropout_rate=Config.DROPOUT_RATE,
        ).to(device)

    if Config.MODEL == "logreg":
        return StockLogReg(
            seq_len=Config.SEQ_LEN,
            input_size=Config.INPUT_SIZE,
            output_size=Config.OUTPUT_SIZE,
        ).to(device)

    raise ValueError("Invalid model type. Choose one of {'lstm', 'attention', 'logreg'}.")


def model_title() -> str:
    model_map = {"lstm": "LSTM", "attention": "Transformer", "logreg": "LogReg"}
    base = model_map.get(Config.MODEL, Config.MODEL)
    sent = "with sentiment" if Config.USE_SENTIMENT else "without sentiment"
    weighted = "weighted loss" if Config.USE_WEIGHTED_LOSS else "unweighted loss"
    return f"{base} ({sent}, {weighted})"


def model_checkpoint_path() -> str:
    os.makedirs(Config.NETWORK_DIR, exist_ok=True)
    sent_tag = "sent" if Config.USE_SENTIMENT else "no_sent"
    weighted_tag = "weighted" if Config.USE_WEIGHTED_LOSS else "unweighted"
    ckpt_tag = Config.CKPT_METRIC
    return os.path.join(
        Config.NETWORK_DIR,
        f"{Config.MODEL}_{sent_tag}_{weighted_tag}_{ckpt_tag}_best.pth",
    )


def append_results_csv(macro_f1: float, accuracy: float, custom_cost: float) -> None:
    os.makedirs(Config.FIG_DIR, exist_ok=True)
    results_path = os.path.join(Config.FIG_DIR, "results.csv")

    row = {
        "model": Config.MODEL,
        "sentiment": int(bool(Config.USE_SENTIMENT)),
        "weighted_loss": int(bool(Config.USE_WEIGHTED_LOSS)),
        "ckpt_metric": Config.CKPT_METRIC,
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
        "custom_cost": float(custom_cost),
    }

    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
        pd.DataFrame([row]).to_csv(results_path, index=False)
        return

    pd.DataFrame([row]).to_csv(results_path, mode="a", header=False, index=False)


def save_confusion_matrix_plot(all_targets, all_predictions, filename_tag: str):
    # Desired order: DOWN / NEUTRAL / UP (both rows and columns)
    order = [1, 2, 0]  # DOWN, NEUTRAL, UP
    names = ["DOWN (1)", "NEUTRAL (2)", "UP (0)"]

    cm = confusion_matrix(all_targets, all_predictions)
    cm_reordered = cm[np.ix_(order, order)]  # reorder rows AND columns

    plt.figure(figsize=(8.5, 6.5))
    sns.set_context("paper", font_scale=1.35)

    sns.heatmap(
        cm_reordered,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        cbar=True,
    )

    plt.ylabel("True label", fontweight="bold")
    plt.xlabel("Predicted label", fontweight="bold")
    plt.title(f"Confusion Matrix — {model_title()}", fontweight="bold")

    os.makedirs(Config.FIG_DIR, exist_ok=True)
    file_path = os.path.join(Config.FIG_DIR, f"cm_{filename_tag}.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot to: {file_path}")


def train_and_test_once() -> None:
    device = setup_device()

    ckpt_path = model_checkpoint_path()

    # If checkpoint already exists, skip training and directly evaluate that model.
    if os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 0:
        print(f"\nFound existing checkpoint, skipping training: {ckpt_path}")

        print("\nLoading data...")
        train_loader, feature_scaler = create_dataloader(
            batch_size=Config.BATCH_SIZE,
            use_sentiment=Config.USE_SENTIMENT,
            is_train=True,
        )

        xb, _ = next(iter(train_loader))
        Config.INPUT_SIZE = xb.shape[-1]
        Config.SEQ_LEN = xb.shape[-2]

        test_loader, _ = create_dataloader(
            batch_size=Config.BATCH_SIZE,
            is_train=False,
            scaler=feature_scaler,
            use_sentiment=Config.USE_SENTIMENT,
        )

        model_test = build_model(device)
        state_dict = torch.load(ckpt_path, map_location=device)
        model_test.load_state_dict(state_dict)

        criterion_test = nn.CrossEntropyLoss()

        test_loss, test_acc, test_macro_f1, test_custom_cost, all_targets, all_predictions = evaluate_model(
            model=model_test,
            dataloader=test_loader,
            criterion=criterion_test,
            device=device,
        )

        class_names = ["UP (0)", "DOWN (1)", "NEUTRAL (2)"]
        print("\n--- FINAL TEST RESULTS ---")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        print(f"Test Macro-F1: {test_macro_f1:.4f}")
        print(f"Test Custom Cost: {test_custom_cost:.4f}")

        print("\n--- CLASSIFICATION REPORT ---")
        report = classification_report(
            y_true=all_targets,
            y_pred=all_predictions,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
        print(report)

        append_results_csv(macro_f1=test_macro_f1, accuracy=test_acc, custom_cost=test_custom_cost)

        sent_tag = "with_sentiment" if Config.USE_SENTIMENT else "without_sentiment"
        weighted_tag = "weighted" if Config.USE_WEIGHTED_LOSS else "unweighted"
        file_tag = f"{Config.MODEL}_{sent_tag}_{weighted_tag}_{Config.CKPT_METRIC}"
        save_confusion_matrix_plot(all_targets, all_predictions, filename_tag=file_tag)
        return

    print("\nLoading data...")
    train_loader, feature_scaler = create_dataloader(
        batch_size=Config.BATCH_SIZE,
        use_sentiment=Config.USE_SENTIMENT,
        is_train=True,
    )

    xb, _ = next(iter(train_loader))
    Config.INPUT_SIZE = xb.shape[-1]
    Config.SEQ_LEN = xb.shape[-2]

    val_loader, _ = create_dataloader(
        batch_size=Config.BATCH_SIZE,
        is_train=False,
        use_sentiment=Config.USE_SENTIMENT,
        scaler=feature_scaler,
    )

    model = build_model(device)

    if Config.USE_WEIGHTED_LOSS:
        weights_tensor = torch.tensor([1.0, 1.3, 1.0], dtype=torch.float32).to(device)
    else:
        weights_tensor = None
    criterion_train = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_val_loss = float("inf")

    if Config.CKPT_METRIC == "val_loss":
        best_ckpt_score = float("inf")
        ckpt_mode = "min"
    elif Config.CKPT_METRIC in {"val_acc", "val_macro_f1"}:
        best_ckpt_score = -float("inf")
        ckpt_mode = "max"
    else:
        raise ValueError(f"Unknown CKPT_METRIC: {Config.CKPT_METRIC}")

    early_stopper = EarlyStopper(patience=Config.PATIENCE)

    print(f"Starting training... (early-stop on val loss, checkpoint on {Config.CKPT_METRIC}) | {model_title()}")
    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion_train, optimizer, device)

        val_loss, val_acc, val_macro_f1, val_custom_cost, _, _ = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion_train,
            device=device,
        )

        print(
            f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"Val Macro-F1: {val_macro_f1:.4f} | "
            f"Val Custom Cost: {val_custom_cost:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if Config.CKPT_METRIC == "val_loss":
            ckpt_score = val_loss
        elif Config.CKPT_METRIC == "val_acc":
            ckpt_score = val_acc
        elif Config.CKPT_METRIC == "val_macro_f1":
            ckpt_score = val_macro_f1
        else:
            raise ValueError(f"Unknown CKPT_METRIC: {Config.CKPT_METRIC}")

        improved = (ckpt_score < best_ckpt_score) if ckpt_mode == "min" else (ckpt_score > best_ckpt_score)
        if improved:
            best_ckpt_score = ckpt_score
            torch.save(model.state_dict(), ckpt_path)
            if ckpt_mode == "min":
                print(f"--> Saved best model (min {Config.CKPT_METRIC}): {best_ckpt_score:.4f} to {ckpt_path}")
            else:
                print(f"--> Saved best model (max {Config.CKPT_METRIC}): {best_ckpt_score:.4f} to {ckpt_path}")

        if early_stopper.early_stop(val_loss):
            print(f"!!! Early stopping after {early_stopper.counter} epochs without improvement in validation loss.")
            break

    print(f"Training complete in {(time.time() - start_time):.2f} seconds.")
    print(f"Best validation loss (early stopping metric): {best_val_loss:.4f}")
    print(f"Best checkpoint {Config.CKPT_METRIC}: {best_ckpt_score:.4f}")

    print("\nReloading best checkpoint and evaluating on test set...")

    test_loader, _ = create_dataloader(
        batch_size=Config.BATCH_SIZE,
        is_train=False,
        scaler=feature_scaler,
        use_sentiment=Config.USE_SENTIMENT,
    )

    model_test = build_model(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model_test.load_state_dict(state_dict)

    criterion_test = nn.CrossEntropyLoss()

    test_loss, test_acc, test_macro_f1, test_custom_cost, all_targets, all_predictions = evaluate_model(
        model=model_test,
        dataloader=test_loader,
        criterion=criterion_test,
        device=device,
    )

    class_names = ["UP (0)", "DOWN (1)", "NEUTRAL (2)"]
    print("\n--- FINAL TEST RESULTS ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Macro-F1: {test_macro_f1:.4f}")
    print(f"Test Custom Cost: {test_custom_cost:.4f}")

    print("\n--- CLASSIFICATION REPORT ---")
    report = classification_report(
        y_true=all_targets,
        y_pred=all_predictions,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(report)

    append_results_csv(macro_f1=test_macro_f1, accuracy=test_acc, custom_cost=test_custom_cost)

    sent_tag = "with_sentiment" if Config.USE_SENTIMENT else "without_sentiment"
    weighted_tag = "weighted" if Config.USE_WEIGHTED_LOSS else "unweighted"
    file_tag = f"{Config.MODEL}_{sent_tag}_{weighted_tag}_{Config.CKPT_METRIC}"
    save_confusion_matrix_plot(all_targets, all_predictions, filename_tag=file_tag)


def run_sweep():
    combos = []
    for ckpt_metric in ["val_loss", "val_acc", "val_macro_f1"]:
        for model_name in ["lstm", "attention", "logreg"]:
            for use_sentiment in [False, True]:
                for use_weighted_loss in [False, True]:
                    combos.append((ckpt_metric, model_name, use_sentiment, use_weighted_loss))

    print(f"Will run {len(combos)} trainings:")
    for (cm, m, s, w) in combos:
        print(f"  - ckpt_metric={cm}, model={m}, sentiment={int(s)}, weighted_loss={int(w)}")

    overall_start = time.time()

    for i, (ckpt_metric, model_name, use_sentiment, use_weighted_loss) in enumerate(combos, start=1):
        print("\n" + "=" * 90)
        print(
            f"RUN {i}/{len(combos)} | ckpt_metric={ckpt_metric} | model={model_name} | "
            f"sentiment={int(use_sentiment)} | weighted_loss={int(use_weighted_loss)}"
        )
        print("=" * 90)

        Config.CKPT_METRIC = ckpt_metric
        Config.MODEL = model_name
        Config.USE_SENTIMENT = use_sentiment
        Config.USE_WEIGHTED_LOSS = use_weighted_loss


        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        try:
            train_and_test_once()
        except Exception as e:
            os.makedirs(Config.FIG_DIR, exist_ok=True)
            results_path = os.path.join(Config.FIG_DIR, "results.csv")
            fail_row = {
                "model": Config.MODEL,
                "sentiment": int(bool(Config.USE_SENTIMENT)),
                "weighted_loss": int(bool(Config.USE_WEIGHTED_LOSS)),
                "ckpt_metric": Config.CKPT_METRIC,
                "macro_f1": np.nan,
                "accuracy": np.nan,
                "custom_cost": np.nan,
                "error": repr(e),
            }
            if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
                pd.DataFrame([fail_row]).to_csv(results_path, index=False)
            else:
                pd.DataFrame([fail_row]).to_csv(results_path, mode="a", header=False, index=False)

            print(f"!!! Run failed for model={Config.MODEL}, sentiment={int(Config.USE_SENTIMENT)}, weighted_loss={int(Config.USE_WEIGHTED_LOSS)}")
            print(f"    Error: {e}")

    print("\n" + "=" * 90)
    print(f"SWEEP DONE. Total wall time: {(time.time() - overall_start) / 60.0:.1f} minutes")
    print(f"Results appended to: {os.path.join(Config.FIG_DIR, 'results.csv')}")
    print("=" * 90)


if __name__ == "__main__":
    run_sweep()

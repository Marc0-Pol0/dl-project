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
    NUM_EPOCHS = 40
    BATCH_SIZE = 32
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.2
    PATIENCE = 5
    INPUT_SIZE = 21
    OUTPUT_SIZE = 3  # UP, DOWN, NEUTRAL
    MODEL_SAVE_PATH = "./networks/lstm_nosent_buffer_ea_date.pth"
    DIM_FFN = 4 * HIDDEN_SIZE
    NUMBER_OF_ENCODERS = 2
    NUMBER_OF_HEADS = 4
    MODEL = "lstm"  # choose model in {'lstm', 'attention', 'logreg'}
    USE_SENTIMENT = False
    SEQ_LEN = 30

    # Metric used for checkpoint selection AND printed score in confusion matrix figure.
    # Choose in {"macro_f1", "custom_cost"}.
    # - macro_f1: maximize
    # - custom_cost: minimize (0 for correct, 1 for neutral<->direction, 2 for up<->down)
    METRIC = "macro_f1"
    COST_NEUTRAL = 1.0
    COST_UPDOWN = 3.0


def setup_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EarlyStopper:
    """
    Early stopping that MINIMIZES the monitored value.

    To early-stop on a metric you want to MAXIMIZE (e.g., macro-F1), pass -metric.
    To early-stop on a metric you want to MINIMIZE (e.g., custom_cost), pass metric.
    """
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


def metric_to_earlystop_value(metric_value: float, higher_is_better: bool) -> float:
    """
    Convert your chosen metric into a scalar that should be MINIMIZED by EarlyStopper.
      - If the metric is "higher is better" (macro-F1), minimize -metric.
      - If the metric is "lower is better" (custom cost), minimize metric directly.
    """
    metric_value = float(metric_value)
    return -metric_value if higher_is_better else metric_value


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
    Labels assumed: UP=0, DOWN=1, NEUTRAL=2.
    Lower is better.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape

    cost = np.zeros_like(y_true, dtype=np.float32)

    correct = (y_true == y_pred)
    if correct.all():
        return 0.0

    # UP<->DOWN (0<->1) mistakes
    updown_mistake = (~correct) & (y_true != 2) & (y_pred != 2)
    cost[updown_mistake] = float(cost_updown)

    # neutral<->direction mistakes (one side is 2)
    neutral_mistake = (~correct) & ((y_true == 2) ^ (y_pred == 2))
    cost[neutral_mistake] = float(cost_neutral)

    # Any other unexpected labels would remain 0; keep behavior explicit:
    # (If you introduce other classes, update this function.)
    return float(cost.mean())


def compute_metric(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
) -> tuple[float, bool, str]:
    """
    Returns (metric_value, higher_is_better, metric_label).
    """
    metric = Config.METRIC.lower()
    if metric == "macro_f1":
        val = f1_score(y_true, y_pred, average="macro", zero_division=0)
        return float(val), True, "Macro-F1"
    if metric == "custom_cost":
        val = compute_custom_cost(
            y_true=y_true,
            y_pred=y_pred,
            cost_neutral=Config.COST_NEUTRAL,
            cost_updown=Config.COST_UPDOWN,
        )
        return float(val), False, "Custom Cost"
    raise ValueError("Config.METRIC must be one of {'macro_f1', 'custom_cost'}.")


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
) -> tuple[float, float, float, list[int], list[int], str]:
    """
    Returns: (avg_loss, accuracy, metric_value, y_true, y_pred, metric_label)
    """
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

    metric_value, _, metric_label = compute_metric(all_targets, all_predictions)
    return avg_loss, accuracy, metric_value, all_targets, all_predictions, metric_label


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


def is_improvement(new_val: float, best_val: float, higher_is_better: bool) -> bool:
    if higher_is_better:
        return new_val > best_val
    return new_val < best_val


def run_training():
    device = setup_device()

    print("Loading training and testing data...")
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
        use_sentiment=Config.USE_SENTIMENT,
        scaler=feature_scaler,
    )

    model = build_model(device)

    weights_tensor = torch.tensor([1, 1.3, 1], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Metric choice + direction
    _, higher_is_better, metric_label = compute_metric([0], [0])
    best_metric = -float("inf") if higher_is_better else float("inf")

    # Early stopping now uses the SAME metric as checkpoint selection
    early_stopper = EarlyStopper(patience=Config.PATIENCE)
    start_time = time.time()

    print(f"Starting training... (checkpoint/early-stop metric: {metric_label})")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss, val_acc, val_metric, _, _, _ = evaluate_model(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"{metric_label}: {val_metric:.4f}"
        )

        # Save best checkpoint by metric
        if is_improvement(val_metric, best_metric, higher_is_better):
            best_metric = val_metric
            os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            direction = "max" if higher_is_better else "min"
            print(f"--> Saved best model with {metric_label} ({direction}): {best_metric:.4f}")

        # Early stop by metric (converted to "lower is better")
        early_stop_value = metric_to_earlystop_value(val_metric, higher_is_better)
        if early_stopper.early_stop(early_stop_value):
            print(
                f"!!! Early stopping triggered after {early_stopper.counter} epochs without improvement "
                f"in {metric_label}."
            )
            break

    end_time = time.time()
    print(f"\nTraining complete in {(end_time - start_time):.2f} seconds.")


def run_testing(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
    class_names = ["UP (0)", "DOWN (1)", "NEUTRAL (2)"]

    print("\n--- Running Final Test Evaluation ---")
    test_loss, test_acc, test_metric, all_targets, all_predictions, metric_label = evaluate_model(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
    )

    print("\n--- FINAL TEST RESULTS ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test {metric_label}: {test_metric:.4f}")
    print("--------------------------")

    print("\n--- CLASSIFICATION REPORT ---")
    report = classification_report(
        y_true=all_targets,
        y_pred=all_predictions,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(report)

    cm = confusion_matrix(all_targets, all_predictions)
    print("\n--- CONFUSION MATRIX (Row = True, Column = Predicted) ---")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)

    return test_acc, test_metric, metric_label, all_targets, all_predictions


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


if __name__ == "__main__":
    run_training()

    device = setup_device()

    train_loader, feature_scaler = create_dataloader(
        batch_size=Config.BATCH_SIZE,
        is_train=True,
        use_sentiment=Config.USE_SENTIMENT,
    )
    test_loader, _ = create_dataloader(
        batch_size=Config.BATCH_SIZE,
        is_train=False,
        scaler=feature_scaler,
        use_sentiment=Config.USE_SENTIMENT,
    )

    model = build_model(device)
    criterion = nn.CrossEntropyLoss()

    state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)

    accuracy, metric_value, metric_label, all_targets, all_predictions = run_testing(
        model, test_loader, criterion, device
    )

    model_tag = f"{Config.MODEL}{'' if Config.USE_SENTIMENT else '_nosent'}_{Config.METRIC}"
    save_confusion_matrix_plot(all_targets, all_predictions, model_tag)

    print("Accuracy on the test set: ", accuracy)

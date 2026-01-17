import os
import time
import csv

from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from dataloaders import create_dataloader
from model import StockLSTM, StockTransformer


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


class Config:
    REDO_TRAINING_IF_EXISTS = False

    # If False -> do not include the 3 sentiment columns (input features go 21 -> 18)
    USE_SENTIMENT = False

    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 15
    BATCH_SIZE = 8
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.5
    PATIENCE = NUM_EPOCHS

    OUTPUT_SIZE = 3  # UP=0, DOWN=1, NEUTRAL=2

    DIM_FFN = 4 * HIDDEN_SIZE
    NUMBER_OF_ENCODERS = 2
    NUMBER_OF_HEADS = 4

    # Choose model in {'lstm', 'attention', 'logreg'}
    MODEL = "logreg"

    CLASS_WEIGHTS_TORCH = [1.0, 1.3, 1.0]

    NETWORKS_DIR = "./networks"
    FIGURES_DIR = "./src/figures"


def setup_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EarlyStopper:
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_validation_loss = float("inf")

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.best_validation_loss - self.min_delta:
            self.best_validation_loss = validation_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def compute_metrics(y_true: list[int], y_pred: list[int]) -> tuple[float, float, int]:
    """
    Returns (macro_f1, accuracy, custom_cost)

    Custom cost:
      - correct: 0
      - UP<->DOWN mistakes (0<->1): 3
      - any other mistake: 1
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    accuracy = float(np.mean(y_true_arr == y_pred_arr))

    wrong = y_true_arr != y_pred_arr
    up_down = ((y_true_arr == 0) & (y_pred_arr == 1)) | ((y_true_arr == 1) & (y_pred_arr == 0))

    raw_cost = np.sum(wrong.astype(int) * 1 + (up_down & wrong).astype(int) * 2)
    custom_cost = float(raw_cost / len(y_true_arr))
    return macro_f1, accuracy, custom_cost


def append_results_csv(
    figures_dir: str, model: str, use_sentiment: bool, macro_f1: float, accuracy: float, custom_cost: int
) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    results_path = os.path.join(figures_dir, "results.csv")
    file_exists = os.path.exists(results_path)

    with open(results_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["model", "sentiment", "macro_f1", "accuracy", "custom_cost"])
        w.writerow([model, int(use_sentiment), f"{macro_f1:.6f}", f"{accuracy:.6f}", f"{custom_cost:.6f}"])

    print(f"Appended results to: {results_path}")


def train_one_epoch(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer, device: torch.device
) -> float:
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


def validate_model(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += Y_batch.size(0)
            correct += (predicted == Y_batch).sum().item()

    return total_loss / len(dataloader), (correct / total if total > 0 else 0.0)


def run_testing_torch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0

    all_predictions: list[int] = []
    all_targets: list[int] = []

    class_names = ["UP (0)", "DOWN (1)", "NEUTRAL (2)"]

    print("\n--- Running Final Test Evaluation (Torch) ---")

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_targets.extend(Y_batch.cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader)
    macro_f1, accuracy, custom_cost = compute_metrics(all_targets, all_predictions)

    print("\n--- FINAL TEST RESULTS ---")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy*100:.2f}% | Macro-F1: {macro_f1:.4f} | Custom Cost: {custom_cost}")
    print("--------------------------")

    print("\n--- CLASSIFICATION REPORT ---")
    print(
        classification_report(
            y_true=all_targets,
            y_pred=all_predictions,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(all_targets, all_predictions)
    print("\n--- CONFUSION MATRIX (Row = True, Column = Predicted) ---")
    print(pd.DataFrame(cm, index=class_names, columns=class_names))

    return macro_f1, accuracy, custom_cost, all_targets, all_predictions


def dataloader_to_numpy(dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for X_batch, y_batch in dataloader:
        Xs.append(X_batch.detach().cpu().numpy())
        ys.append(y_batch.detach().cpu().numpy())
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


def flatten_time_series(X: np.ndarray) -> np.ndarray:
    if X.ndim == 3:
        n, t, f = X.shape
        return X.reshape(n, t * f)
    if X.ndim == 2:
        return X
    raise ValueError(f"Unexpected X shape for logreg: {X.shape}")


def run_training_logreg(train_loader: DataLoader, test_loader: DataLoader, save_path: str):
    if (not Config.REDO_TRAINING_IF_EXISTS) and os.path.exists(save_path):
        print(f"Found existing LogReg model at {save_path} -> skipping training.")
        return load(save_path)

    print("Training LogReg (sklearn)...")

    X_train, y_train = dataloader_to_numpy(train_loader)
    X_test, y_test = dataloader_to_numpy(test_loader)

    X_train = flatten_time_series(X_train)
    X_test = flatten_time_series(X_test)

    clf = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None,
    )

    t0 = time.time()
    clf.fit(X_train, y_train)
    t1 = time.time()

    val_acc = float(clf.score(X_test, y_test))
    print(f"LogReg training complete in {(t1 - t0):.2f}s | Val Acc: {val_acc*100:.2f}%")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dump(clf, save_path)
    print(f"--> Saved LogReg model to {save_path}")

    return clf


def run_testing_logreg(clf, test_loader: DataLoader):
    X_test, y_test = dataloader_to_numpy(test_loader)
    X_test = flatten_time_series(X_test)

    y_pred = clf.predict(X_test).astype(int).tolist()
    y_true = y_test.astype(int).tolist()

    class_names = ["UP (0)", "DOWN (1)", "NEUTRAL (2)"]

    macro_f1, accuracy, custom_cost = compute_metrics(y_true, y_pred)

    print("\n--- Running Final Test Evaluation (LogReg) ---")
    print(f"Accuracy: {accuracy*100:.2f}% | Macro-F1: {macro_f1:.4f} | Custom Cost: {custom_cost}")

    print("\n--- CLASSIFICATION REPORT ---")
    print(
        classification_report(
            y_true=y_true,
            y_pred=y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred)
    print("\n--- CONFUSION MATRIX (Row = True, Column = Predicted) ---")
    print(pd.DataFrame(cm, index=class_names, columns=class_names))

    return macro_f1, accuracy, custom_cost, y_true, y_pred


def run_training_torch(
    train_loader: DataLoader, test_loader: DataLoader, device: torch.device, save_path: str, input_size: int
) -> None:
    if (not Config.REDO_TRAINING_IF_EXISTS) and os.path.exists(save_path):
        print(f"Found existing torch model at {save_path} -> skipping training.")
        return

    if Config.MODEL == "attention":
        model = StockTransformer(
            input_size=input_size,
            d_model=Config.HIDDEN_SIZE,
            nhead=Config.NUMBER_OF_HEADS,
            num_encoder_layers=Config.NUMBER_OF_ENCODERS,
            dim_feedforward=Config.DIM_FFN,
            output_size=Config.OUTPUT_SIZE,
        ).to(device)
    elif Config.MODEL == "lstm":
        model = StockLSTM(
            input_size=input_size,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            output_size=Config.OUTPUT_SIZE,
            dropout_rate=Config.DROPOUT_RATE,
        ).to(device)
    else:
        raise ValueError(f"Config.MODEL must be one of {{'lstm','attention','logreg'}}, got: {Config.MODEL}")

    weights_tensor = torch.tensor(Config.CLASS_WEIGHTS_TORCH, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_val_loss = float("inf")
    early_stopper = EarlyStopper(patience=Config.PATIENCE)

    print("Starting training (Torch)...")
    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate_model(model, test_loader, criterion, device)

        print(
            f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy*100:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model with Val Loss: {best_val_loss:.4f}")

        if early_stopper.early_stop(val_loss):
            print(f"!!! Early stopping triggered after {early_stopper.counter} epochs without improvement.")
            break

    end_time = time.time()
    print(f"\nTraining complete in {(end_time - start_time):.2f} seconds.")


def save_confusion_matrix_plot(all_targets, all_predictions, model_name: str, figures_dir: str):
    labels_order = [1, 2, 0]
    tick_labels = ["DOWN (1)", "NEUTRAL (2)", "UP (0)"]

    cm = confusion_matrix(all_targets, all_predictions, labels=labels_order)

    plt.figure(figsize=(8, 6))
    sns.set_context("paper", font_scale=1.4)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cbar=True,
    )

    plt.ylabel("True Label", fontweight="bold")
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.title(f"Confusion Matrix: {model_name}", fontweight="bold")

    os.makedirs(figures_dir, exist_ok=True)
    file_path = os.path.join(figures_dir, f"cm_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot to: {file_path}")


def main():
    device = setup_device()

    sent_suffix = "" if Config.USE_SENTIMENT else "_nosent"
    model_tag = f"{Config.MODEL}{sent_suffix}"

    torch_model_path = os.path.join(Config.NETWORKS_DIR, f"{model_tag}.pth")
    logreg_model_path = os.path.join(Config.NETWORKS_DIR, f"{model_tag}.joblib")

    print("Loading training and testing data...")
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

    first_X, _first_y = next(iter(train_loader))
    if first_X.ndim in (2, 3):
        input_size = int(first_X.shape[-1])
    else:
        raise ValueError(f"Unexpected batch X shape: {tuple(first_X.shape)}")

    if Config.MODEL == "logreg":
        clf = run_training_logreg(train_loader, test_loader, logreg_model_path)
        macro_f1, accuracy, custom_cost, y_true, y_pred = run_testing_logreg(clf, test_loader)
        save_confusion_matrix_plot(y_true, y_pred, model_tag, Config.FIGURES_DIR)

        append_results_csv(
            figures_dir=Config.FIGURES_DIR,
            model=Config.MODEL,
            use_sentiment=Config.USE_SENTIMENT,
            macro_f1=macro_f1,
            accuracy=accuracy,
            custom_cost=custom_cost,
        )
        return

    run_training_torch(train_loader, test_loader, device, torch_model_path, input_size)

    if Config.MODEL == "attention":
        model = StockTransformer(
            input_size=input_size,
            d_model=Config.HIDDEN_SIZE,
            nhead=Config.NUMBER_OF_HEADS,
            num_encoder_layers=Config.NUMBER_OF_ENCODERS,
            dim_feedforward=Config.DIM_FFN,
            output_size=Config.OUTPUT_SIZE,
        ).to(device)
    elif Config.MODEL == "lstm":
        model = StockLSTM(
            input_size=input_size,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            output_size=Config.OUTPUT_SIZE,
            dropout_rate=Config.DROPOUT_RATE,
        ).to(device)
    else:
        raise ValueError(f"Config.MODEL must be one of {{'lstm','attention','logreg'}}, got: {Config.MODEL}")

    state_dict = torch.load(torch_model_path, map_location=device)
    model.load_state_dict(state_dict)

    weights_tensor = torch.tensor(Config.CLASS_WEIGHTS_TORCH, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    macro_f1, accuracy, custom_cost, y_true, y_pred = run_testing_torch(model, test_loader, criterion, device)
    save_confusion_matrix_plot(y_true, y_pred, model_tag, Config.FIGURES_DIR)

    append_results_csv(
        figures_dir=Config.FIGURES_DIR,
        model=Config.MODEL,
        use_sentiment=Config.USE_SENTIMENT,
        macro_f1=macro_f1,
        accuracy=accuracy,
        custom_cost=custom_cost,
    )


if __name__ == "__main__":
    main()

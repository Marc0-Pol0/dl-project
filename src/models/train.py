import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import save_confusion_matrix_plot
from dataloaders import create_dataloader
from model import StockLSTM, StockTransformer


class Config:
    # Global switch: if False and model file exists -> skip training and reuse it
    REDO_TRAINING_IF_EXISTS = False

    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 15
    BATCH_SIZE = 8
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.5
    PATIENCE = NUM_EPOCHS

    INPUT_SIZE = 21
    OUTPUT_SIZE = 3  # UP=0, DOWN=1, NEUTRAL=2

    DIM_FFN = 4 * HIDDEN_SIZE
    NUMBER_OF_ENCODERS = 2
    NUMBER_OF_HEADS = 4

    # Choose model in {'lstm', 'attention', 'logreg'}
    MODEL = "attention"

    # Torch model path (used for lstm/attention). For logreg we save a .joblib next to it.
    MODEL_SAVE_PATH = "./networks/attention_buffer_ea_date.pth"

    # Weighted CE only used for torch models
    CLASS_WEIGHTS_TORCH = [1.0, 1.3, 1.0]


def setup_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# =========================
# Torch helpers
# =========================
def build_torch_model(device: torch.device) -> nn.Module:
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

    raise ValueError(f"Config.MODEL must be one of {{'lstm','attention','logreg'}}, got: {Config.MODEL}")


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer, device: torch.device) -> float:
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


def validate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
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
    correct = 0
    total = 0

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

            total += Y_batch.size(0)
            correct += (predicted == Y_batch).sum().item()

            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_targets.extend(Y_batch.cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0

    print("\n--- FINAL TEST RESULTS ---")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
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

    return accuracy, all_targets, all_predictions


# =========================
# LogReg helpers
# =========================
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
    # expected X shape: (N, T, F) for torch models
    if X.ndim == 3:
        n, t, f = X.shape
        return X.reshape(n, t * f)
    if X.ndim == 2:
        return X
    raise ValueError(f"Unexpected X shape for logreg: {X.shape}")


def get_logreg_save_path() -> str:
    root, _ext = os.path.splitext(Config.MODEL_SAVE_PATH)
    return root + "_logreg.joblib"


def run_training_logreg(train_loader: DataLoader, test_loader: DataLoader):
    from joblib import dump, load  # local import to avoid dependency if unused

    save_path = get_logreg_save_path()
    if (not Config.REDO_TRAINING_IF_EXISTS) and os.path.exists(save_path):
        print(f"Found existing LogReg model at {save_path} -> skipping training.")
        clf = load(save_path)
        return clf

    print("Training LogReg (sklearn)...")

    X_train, y_train = dataloader_to_numpy(train_loader)
    X_test, y_test = dataloader_to_numpy(test_loader)  # for quick validation prints, optional

    X_train = flatten_time_series(X_train)
    X_test = flatten_time_series(X_test)

    clf = LogisticRegression(
        max_iter=2000,
        multi_class="auto",
        class_weight="balanced",  # simple default to handle imbalance
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

    accuracy = float(np.mean(np.array(y_pred) == np.array(y_true)))

    print("\n--- Running Final Test Evaluation (LogReg) ---")
    print(f"Test Accuracy: {accuracy*100:.2f}%")

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

    return accuracy, y_true, y_pred


# =========================
# Training entrypoints
# =========================
def run_training_torch(train_loader: DataLoader, test_loader: DataLoader, device: torch.device) -> None:
    # If we can reuse an existing torch model, skip training.
    if (not Config.REDO_TRAINING_IF_EXISTS) and os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Found existing torch model at {Config.MODEL_SAVE_PATH} -> skipping training.")
        return

    model = build_torch_model(device)

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
            os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"--> Saved best model with Val Loss: {best_val_loss:.4f}")

        if early_stopper.early_stop(val_loss):
            print(f"!!! Early stopping triggered after {early_stopper.counter} epochs without improvement.")
            break

    end_time = time.time()
    print(f"\nTraining complete in {(end_time - start_time):.2f} seconds.")


def main():
    device = setup_device()

    print("Loading training and testing data...")
    train_loader, feature_scaler = create_dataloader(batch_size=Config.BATCH_SIZE, is_train=True)
    test_loader, _ = create_dataloader(batch_size=Config.BATCH_SIZE, is_train=False, scaler=feature_scaler)

    if Config.MODEL == "logreg":
        clf = run_training_logreg(train_loader, test_loader)
        accuracy, y_true, y_pred = run_testing_logreg(clf, test_loader)
        save_confusion_matrix_plot(y_true, y_pred, "logreg")
        print("Accuracy on the test set: ", accuracy)
        return

    # Torch models
    run_training_torch(train_loader, test_loader, device)

    model = build_torch_model(device)
    state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)

    # Match criterion to training for consistent loss reporting
    weights_tensor = torch.tensor(Config.CLASS_WEIGHTS_TORCH, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    accuracy, y_true, y_pred = run_testing_torch(model, test_loader, criterion, device)
    save_confusion_matrix_plot(y_true, y_pred, Config.MODEL)
    print("Accuracy on the test set: ", accuracy)


if __name__ == "__main__":
    main()

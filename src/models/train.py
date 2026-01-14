import os
import time
import random

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import save_confusion_matrix_plot
from dataloaders import create_dataloader
from model import StockLSTM, StockTransformer


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available(): 
    torch.cuda.manual_seed_all(0)


class Config:
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 15
    BATCH_SIZE = 32
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.5
    PATIENCE = 5
    INPUT_SIZE = 21
    OUTPUT_SIZE = 3  # UP, DOWN, NEUTRAL
    MODEL_SAVE_PATH = "./networks/logreg_buffer_ea_date.pth"
    DIM_FFN = 4 * HIDDEN_SIZE  # feedforward network dimension for Transformer
    NUMBER_OF_ENCODERS = 2
    NUMBER_OF_HEADS = 4
    MODEL = "logreg"  # choose model in {'lstm', 'attention', 'logreg'}
    USE_SENTIMENT = True
    SEQ_LEN = 30


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


class EarlyStopper:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.best_validation_loss - self.min_delta:
            self.best_validation_loss = validation_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


class StockLogReg(nn.Module):
    """
    Multiclass logistic regression over flattened sequence input.
    Expects x as (B, T, F) and flattens to (B, T*F).
    """

    def __init__(self, seq_len: int, input_size: int, output_size: int):
        super().__init__()
        self.seq_len = int(seq_len)
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.linear = nn.Linear(self.seq_len * self.input_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"logreg expects (B, T, F), got {tuple(x.shape)}")

        b, t, f = x.shape
        if t != self.seq_len or f != self.input_size:
            raise ValueError(
                f"logreg got (T,F)=({t},{f}) but expected ({self.seq_len},{self.input_size}). "
                "Set Config.SEQ_LEN to your dataset window length."
            )

        x = x.reshape(b, t * f)
        return self.linear(x)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for _, (X_batch, Y_batch) in enumerate(dataloader):
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_samples += Y_batch.size(0)
            correct_predictions += (predicted == Y_batch).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy


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


def run_training():
    device = setup_device()

    print("Loading training and testing data...")
    train_loader, feature_scaler = create_dataloader(
        batch_size=Config.BATCH_SIZE, 
        use_sentiment=Config.USE_SENTIMENT,
        is_train=True
    )

    xb, _ = next(iter(train_loader))
    Config.INPUT_SIZE = xb.shape[-1]   # feature dim F
    Config.SEQ_LEN = xb.shape[-2]      # seq len T

    test_loader, _ = create_dataloader(
        batch_size=Config.BATCH_SIZE,
        is_train=False,
        use_sentiment=Config.USE_SENTIMENT,
        scaler=feature_scaler,  # use same scaler to prevent data leakage
    )

    model = build_model(device)

    weights_tensor = torch.tensor([1, 1.3, 1], dtype=torch.float32).to(device)  # class weights for imbalance
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_val_loss = float("inf")
    early_stopper = EarlyStopper(patience=Config.PATIENCE)
    start_time = time.time()

    print("Starting training...")

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


def run_testing(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_predictions = []
    all_targets = []

    class_names = ["UP (0)", "DOWN (1)", "NEUTRAL (2)"]

    print("\n--- Running Final Test Evaluation ---")

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_samples += Y_batch.size(0)
            correct_predictions += (predicted == Y_batch).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(Y_batch.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    weighted_f1 = f1_score(all_targets, all_predictions, average="weighted", zero_division=0)

    print("\n--- FINAL TEST RESULTS ---")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Test Weighted F1: {weighted_f1:.4f}")
    print("--------------------------")

    print("\n--- CLASSIFICATION REPOR ---")
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

    return accuracy, weighted_f1, all_targets, all_predictions


if __name__ == "__main__":
    run_training()

    device = setup_device()
    criterion = nn.CrossEntropyLoss()

    model = build_model(device)

    train_loader, feature_scaler = create_dataloader(
        batch_size=Config.BATCH_SIZE, is_train=True, use_sentiment=Config.USE_SENTIMENT
    )
    test_loader, _ = create_dataloader(
        batch_size=Config.BATCH_SIZE, is_train=False, scaler=feature_scaler, use_sentiment=Config.USE_SENTIMENT
    )

    state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)

    accuracy, weighted_f1, all_targets, all_predictions = run_testing(model, test_loader, criterion, device)
    model_tag = f"{Config.MODEL}{'' if Config.USE_SENTIMENT else '_nosent'}"

    save_confusion_matrix_plot(all_targets, all_predictions, model_tag, weighted_f1=weighted_f1)

    print("Accuracy on the test set: ", accuracy)

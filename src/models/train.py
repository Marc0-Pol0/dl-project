import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from utils import save_confusion_matrix_plot

# Import your custom modules
from dataloaders import create_dataloader
from model import StockLSTM, StockTransformer

# Configuration class
class Config:
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 15    
    BATCH_SIZE = 8
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.5  
    PATIENCE = NUM_EPOCHS    # No Early Stopping, dropout regularization used.
    INPUT_SIZE = 21     # Number of input features
    OUTPUT_SIZE = 3     # Number of output classes (UP, DOWN, NEUTRAL)
    MODEL_SAVE_PATH = './networks/attention_buffer_ea_date.pth'
    DIM_FFN = 4 * HIDDEN_SIZE       # Feedforward network dimension for Transformer
    NUMBER_OF_ENCODERS = 2      # Transformer encoder layers
    NUMBER_OF_HEADS = 4     # Multi-head attention heads
    MODEL='attention'    # Chose model in {'lstm', 'attention'}

# Setup device
def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


# Early Stopping Class (not used in this configuration, but still defined for completeness and future use)
class EarlyStopper:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.best_validation_loss - self.min_delta:
            self.best_validation_loss = validation_loss
            self.counter = 0  
            return False
        elif validation_loss > self.best_validation_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True  
            return False
        return False

# Training and evaluation functions
def train_one_epoch(model, dataloader, criterion, optimizer, device):

    model.train()  # Set model to training mode
    total_loss = 0.0
    
    # Iterate through batches
    for batch_idx, (X_batch, Y_batch) in enumerate(dataloader):
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Evaluation step
def validate_model(model, dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():  
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1) # Get the class with max probability
            total_samples += Y_batch.size(0)
            correct_predictions += (predicted == Y_batch).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy

# Training function
def run_training():
    
    device = setup_device()

    # Load data
    print("Loading training and testing data...")
    train_loader, feature_scaler = create_dataloader(
        batch_size=Config.BATCH_SIZE, 
        is_train=True
    )
    test_loader, _ = create_dataloader(
        batch_size=Config.BATCH_SIZE, 
        is_train=False,
        scaler=feature_scaler # Pass the fitted scaler to prevent data leakage
    )

    # Initialize model, loss and optimizer
    if Config.MODEL == 'attention':
        model = StockTransformer(
            input_size=Config.INPUT_SIZE,
            d_model=Config.HIDDEN_SIZE,
            nhead=Config.NUMBER_OF_HEADS,
            num_encoder_layers=Config.NUMBER_OF_ENCODERS,
            dim_feedforward=Config.DIM_FFN,
            output_size=Config.OUTPUT_SIZE
        ).to(device)

    elif Config.MODEL == 'lstm':
        model = StockLSTM(
        input_size=Config.INPUT_SIZE,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        output_size=Config.OUTPUT_SIZE,
        dropout_rate=Config.DROPOUT_RATE
        ).to(device)
    

    # WeightedCrossEntropyLoss loss for multi-class classification
    weights_tensor = torch.tensor([1, 1.3, 1], dtype=torch.float32) # Class weights for imbalance
    weights_tensor = weights_tensor.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Training loop
    best_val_loss = float('inf')
    early_stopper = EarlyStopper(patience=Config.PATIENCE) # Initialize Early Stopper
    start_time = time.time()
    
    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        # Training phase
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_accuracy = validate_model(model, test_loader, criterion, device)

        # Logging and saving
        print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_accuracy*100:.2f}%")

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"--> Saved best model with Val Loss: {best_val_loss:.4f}")
            
        # Early stopping check
        if early_stopper.early_stop(val_loss):
            print(f"!!! Early stopping triggered after {early_stopper.counter} epochs without improvement.")
            break # Exit the loop

    end_time = time.time()
    print(f"\nTraining complete in {(end_time - start_time):.2f} seconds.")

# Testing function
def run_testing(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    all_predictions = []
    all_targets = []
    
    class_names = ['UP (0)', 'DOWN (1)', 'NEUTRAL (2)']
    
    print("\n--- Running Final Test Evaluation ---")
    
    with torch.no_grad():  
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            mean_probs = probs.mean(dim=0).cpu().numpy()
            print("mean predicted probs [UP, DOWN, NEU]:", mean_probs)
            
            # Calculate metrics
            _, predicted = torch.max(outputs.data, 1) 
            total_samples += Y_batch.size(0)
            correct_predictions += (predicted == Y_batch).sum().item()
            
            # Store results for Confusion Matrix and Classification Report
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(Y_batch.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples

    print("\n--- FINAL TEST RESULTS ---")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print("--------------------------")

    # Classification Report
    print("\n--- CLASSIFICATION REPORT (Actionable Metrics) ---")
    
    # The classification report is the essential tool for imbalanced data.
    # It shows F1 and Precision for UP (0) and DOWN (1).
    report = classification_report(
        y_true=all_targets, 
        y_pred=all_predictions, 
        target_names=class_names, 
        digits=4, 
        zero_division=0 # Handle cases where a class has no samples or predictions
    )
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    print("\n--- CONFUSION MATRIX (Row = True, Column = Predicted) ---")
    
    # Format the matrix nicely for display
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    
    return accuracy, all_targets, all_predictions


if __name__ == '__main__':
    run_training()

    device = setup_device()

    criterion = nn.CrossEntropyLoss()

    if Config.MODEL == 'attention':
        model = StockTransformer(
            input_size=Config.INPUT_SIZE,
            d_model=Config.HIDDEN_SIZE,
            nhead=Config.NUMBER_OF_HEADS,
            num_encoder_layers=Config.NUMBER_OF_ENCODERS,
            dim_feedforward=Config.DIM_FFN,
            output_size=Config.OUTPUT_SIZE
        ).to(device)
    elif Config.MODEL == 'lstm':
        model = StockLSTM(
        input_size=Config.INPUT_SIZE,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        output_size=Config.OUTPUT_SIZE,
        dropout_rate=Config.DROPOUT_RATE
        ).to(device)

    train_loader, feature_scaler = create_dataloader(
        batch_size=Config.BATCH_SIZE, 
        is_train=True
    )
    test_loader, _ = create_dataloader(
        batch_size=Config.BATCH_SIZE, 
        is_train=False,
        scaler=feature_scaler # Pass the fitted scaler to prevent data leakage
    )

    state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(state_dict)

    # Predictions on test set
    accuracy, all_targets, all_predictions = run_testing(model, test_loader, criterion, device)

    # Generate and save confusion matrix plot
    save_confusion_matrix_plot(all_targets, all_predictions, Config.MODEL)

    print('Accuracy on the test set: ', accuracy)
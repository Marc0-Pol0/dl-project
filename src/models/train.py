import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Import your custom modules
from dataloaders import create_dataloader
from model import StockLSTM, StockTransformer

# --- Configuration (Hyperparameters) ---
class Config:
    LEARNING_RATE = 1e-4 
    NUM_EPOCHS = 50      # Increase max epochs, as Early Stopping will halt it.
    BATCH_SIZE = 8
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.5  
    PATIENCE = 7         # Early Stopping Patience
    INPUT_SIZE = 21
    OUTPUT_SIZE = 3
    MODEL_SAVE_PATH = './networks/transformer.pth'
    DIM_FFN = 4 * HIDDEN_SIZE
    NUMBER_OF_ENCODERS = 2
    NUMBER_OF_HEADS = 4
    MODEL='lstm'    # Model in {'lstm', 'attention'}

# --- Setup Device (GPU/CPU) ---
def setup_device():
    # Prioritize CUDA for your cluster, fallback to MPS (M2 Mac) or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device

def calculate_gradient_norm(model):
    """Calculates the total L2 norm of all gradients."""
    # Get all model parameters that require gradients
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            # Calculate the norm for the current parameter's gradient
            param_norm = p.grad.data.norm(2) # L2 norm
            total_norm += param_norm.item() ** 2
    return total_norm ** (1./2) # Square root of the sum of squares


class EarlyStopper:
    """Stops training early if validation loss does not improve after a given patience."""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.best_validation_loss - self.min_delta:
            self.best_validation_loss = validation_loss
            self.counter = 0  # Reset counter if improvement is seen
            return False
        elif validation_loss > self.best_validation_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop if patience is exceeded
            return False
        return False

# --- Training and Evaluation Functions ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the model for one full epoch."""
    model.train()  # Set model to training mode
    total_loss = 0.0
    
    for batch_idx, (X_batch, Y_batch) in enumerate(dataloader):
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        # Forward pass
        outputs = model(X_batch)
        # print(outputs)
        loss = criterion(outputs, Y_batch)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Optional: Print progress every 100 batches
        # if batch_idx % 100 == 0:
        #     print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate_model(model, dataloader, criterion, device):
    """Evaluates the model on the validation/test set."""
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():  # Disable gradient calculations during validation
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

# --- Main Training Function ---
def run_training():
    """Initializes and runs the full training process."""
    
    device = setup_device()

    # --- 1. Load Data ---
    # The scaler is fitted on the train data and passed to the test data.
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

    # --- 2. Initialize Model, Loss, and Optimizer ---
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
    

    # CrossEntropyLoss is standard for multi-class classification (includes Softmax internally)
    weights_tensor = torch.tensor([2.5, 3, 1], dtype=torch.float32)
    weights_tensor = weights_tensor.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # --- 3. Training Loop ---
    best_val_loss = float('inf')
    early_stopper = EarlyStopper(patience=Config.PATIENCE) # Initialize Early Stopper
    start_time = time.time()
    
    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        # Training phase
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_accuracy = validate_model(model, test_loader, criterion, device)

        # Logging and Saving
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
            
        # --- EARLY STOPPING CHECK ---
        if early_stopper.early_stop(val_loss):
            print(f"!!! Early stopping triggered after {early_stopper.counter} epochs without improvement.")
            break # Exit the loop

    end_time = time.time()
    print(f"\nTraining complete in {(end_time - start_time):.2f} seconds.")


def run_testing(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
    """
    Evaluates the final model on the test set, calculating overall metrics 
    and the Confusion Matrix for class-specific performance.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    all_predictions = []
    all_targets = []
    
    class_names = ['UP (0)', 'DOWN (1)', 'NEUTRAL (2)']
    
    print("\n--- Running Final Test Evaluation ---")
    
    with torch.no_grad():  # Disable gradient calculations
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            total_loss += loss.item()
            
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

    # --- Generate Classification Report (Precision, Recall, F1-Score) ---
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
    
    # --- Generate Confusion Matrix ---
    cm = confusion_matrix(all_targets, all_predictions)
    
    print("\n--- CONFUSION MATRIX (Row = True, Column = Predicted) ---")
    
    # Format the matrix nicely for display
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    
    return accuracy


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

    accuracy = run_testing(model, test_loader, criterion, device)


    print('Accuracy on the test set: ', accuracy)
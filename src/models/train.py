import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time

# Import your custom modules
from src.data.dataloaders import create_dataloader
from src.models.model import StockLSTM     

# --- Configuration (Hyperparameters) ---
class Config:
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    BATCH_SIZE = 4
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.4
    INPUT_SIZE = 21
    OUTPUT_SIZE = 3
    MODEL_SAVE_PATH = './networks/lstm.pth'

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
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    model = StockLSTM(
        input_size=Config.INPUT_SIZE,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        output_size=Config.OUTPUT_SIZE,
        dropout_rate=Config.DROPOUT_RATE
    ).to(device)

    # CrossEntropyLoss is standard for multi-class classification (includes Softmax internally)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # --- 3. Training Loop ---
    best_val_loss = float('inf')
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

    end_time = time.time()
    print(f"\nTraining complete in {(end_time - start_time):.2f} seconds.")

if __name__ == '__main__':
    run_training()
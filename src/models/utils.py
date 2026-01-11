import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Function to generate and save confusion matrix plot
def save_confusion_matrix_plot(all_targets, all_predictions, model_name):
    class_names = ['UP (0)', 'DOWN (1)', 'NEUTRAL (2)']
    cm = confusion_matrix(all_targets, all_predictions)
    
    plt.figure(figsize=(8, 6))
    sns.set_context("paper", font_scale=1.4) 
    
    # Create the heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar=True
    )
    
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.title(f'Confusion Matrix: {model_name}', fontweight='bold')
    
    # Save the plot for your 4-page PDF report
    file_path = f'./src/figures/cm_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix plot to: {file_path}")


# Oversampling function for minority classes
def oversample_minority_classes(samples_df: pd.DataFrame, target_ratio: float = 0.5) -> pd.DataFrame:
    # Identify Majority Class and its Count
    label_counts = samples_df['label'].value_counts()
    majority_class = label_counts.index[0]
    majority_count = label_counts.iloc[0]
    
    # Determine Target Count
    target_count = int(majority_count * target_ratio)
    
    print(f"\n--- Applying Sequence Oversampling ---")
    print(f"Majority Class ({majority_class}) Count: {majority_count}")
    print(f"Target Count for Minority Classes: {target_count}")

    df_list = [samples_df[samples_df['label'] == majority_class]]
    
    # Process Minority Classes (UP=0, DOWN=1)
    for minority_class in [0, 1]:
        if minority_class not in label_counts.index:
            continue
            
        df_minority = samples_df[samples_df['label'] == minority_class]
        current_count = len(df_minority)
        
        if current_count < target_count:
            # Calculate how many duplicates are needed
            n_duplicates = target_count - current_count
            
            # Use Pandas sample with replacement=True to duplicate samples
            # This generates the exact number of rows needed
            df_oversampled = df_minority.sample(
                n=n_duplicates, 
                replace=True, 
                random_state=42 # Ensure reproducibility
            )
            
            # Combine original minority samples and the new duplicated samples
            df_new = pd.concat([df_minority, df_oversampled], ignore_index=True)
            print(f"Class {minority_class} (Original: {current_count}) oversampled to: {len(df_new)}")
            
            df_list.append(df_new)
        else:
            df_list.append(df_minority) # Add if already at target count or higher

    # Concatenate all classes and shuffle the final training set
    df_balanced = pd.concat(df_list, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Final Total Samples (Training): {len(df_balanced)}")
    return df_balanced

# Function to check and print class distribution in a DataLoader
def check_class_distribution(dataloader: DataLoader, dataset_name: str, device: torch.device):
    
    class_counts = {0: 0, 1: 0, 2: 0} # 0=Up, 1=Down, 2=Neutral
    total_samples = 0
    
    # We use the dataset directly to ensure we get all samples without shuffling issues
    dataset = dataloader.dataset 
    
    print(f"\n--- Class Distribution Check ({dataset_name}) ---")
    
    # Assuming StockDataset has a 'samples_df' attribute (as per your dataloaders.py):
    if hasattr(dataset, 'samples_df'):
        df = dataset.samples_df
        if 'label' in df.columns:
            class_counts = df['label'].value_counts().to_dict()
            total_samples = len(df)
        else:
            # Fallback: iterate through the DataLoader if samples_df is complex
            print("Falling back to DataLoader iteration for class count...")
            for _, Y_batch in dataloader:
                Y_batch = Y_batch.cpu().numpy()
                for label in Y_batch:
                    class_counts[label] += 1
                total_samples += len(Y_batch)
    else:
        # Fallback for datasets without samples_df
        print("Falling back to DataLoader iteration for class count...")
        for _, Y_batch in dataloader:
            Y_batch = Y_batch.cpu().numpy()
            for label in Y_batch:
                class_counts[label] += 1
            total_samples += len(Y_batch)


    if total_samples > 0:
        print(f"Total Samples: {total_samples}")
        
        # Calculate and print the percentage for each class
        for label, count in sorted(class_counts.items()):
            percentage = (count / total_samples) * 100
            label_name = {0: "UP", 1: "DOWN", 2: "NEUTRAL"}.get(label, "UNKNOWN")
            
            # Print with clear formatting
            print(f"  Class {label} ({label_name}): {count} samples ({percentage:.2f}%)")
        
        # Determine the Majority Class Baseline
        max_count = max(class_counts.values())
        majority_baseline = (max_count / total_samples) * 100
        print(f"\nMajority Class Baseline (Guessing Only): {majority_baseline:.2f}%")
        
    else:
        print("No samples found in the dataset.")
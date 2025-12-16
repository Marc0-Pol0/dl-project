import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

def oversample_minority_classes(samples_df: pd.DataFrame, target_ratio: float = 0.5) -> pd.DataFrame:
    """
    Performs random oversampling on the minority classes (0 and 1) 
    in the training set to address class imbalance.
    
    Args:
        samples_df: DataFrame containing all training sequences/samples, 
                    with a 'label' column.
        target_ratio: Target size for minority classes relative to the majority.
        
    Returns:
        A new DataFrame with duplicated samples and a fully shuffled index.
    """
    
    # 1. Identify Majority Class and its Count
    label_counts = samples_df['label'].value_counts()
    majority_class = label_counts.index[0]
    majority_count = label_counts.iloc[0]
    
    # 2. Determine Target Count
    target_count = int(majority_count * target_ratio)
    
    print(f"\n--- Applying Sequence Oversampling ---")
    print(f"Majority Class ({majority_class}) Count: {majority_count}")
    print(f"Target Count for Minority Classes: {target_count}")

    df_list = [samples_df[samples_df['label'] == majority_class]]
    
    # 3. Process Minority Classes (UP=0, DOWN=1)
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

    # 4. Concatenate all classes and shuffle the final training set
    df_balanced = pd.concat(df_list, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Final Total Samples (Training): {len(df_balanced)}")
    print("---------------------------------")
    return df_balanced



def check_class_distribution(dataloader: DataLoader, dataset_name: str, device: torch.device):
    """Calculates and prints the total count and distribution of each class."""
    
    class_counts = {0: 0, 1: 0, 2: 0} # 0=Up, 1=Down, 2=Neutral
    total_samples = 0
    
    # We use the dataset directly to ensure we get all samples without shuffling issues
    dataset = dataloader.dataset 
    
    print(f"\n--- Class Distribution Check ({dataset_name}) ---")

    # Iterate through the samples list directly for efficiency (if possible), 
    # but the safest way is via the DataLoader or by accessing the samples_df attribute.
    
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
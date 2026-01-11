import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import datetime as dt
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

# Configurations
DATA_DIR = './data/trainable/' 
EA_FILENAME = 'earning_dates_500.pkl'
FEATURE_FILENAME = 'final_data_500.pkl'
SEQUENCE_LENGTH = 30  # Days preceding the EA
PRICE_CHANGE_THRESHOLD = 0.03  # 3% threshold for UP/DOWN label

# Calculate label from stock price change
def calculate_label(ticker: str, ea_date: dt.datetime, consolidated_data: Dict[str, pd.DataFrame]) -> int:
    df = consolidated_data[ticker]
    
    # Get stock prices the day after (EA influences the market on the day after) and two days before the EA date
    ea_date = pd.to_datetime(ea_date) + dt.timedelta(days=1)
    pre_ea_date = ea_date - dt.timedelta(days=2)  # Two days before EA to reduce noise
      
    # Get the 'adj_price' for the pre-EA date
    price_pre_ea = df.loc[pre_ea_date]['adj_price']
    
    # Get the stock price the day of the EA
    price_ea = df.loc[ea_date]['adj_price']
        
    # Calculate Return
    daily_return = (price_ea - price_pre_ea) / price_pre_ea
    
    # Assign Label given the threshold
    if daily_return >= PRICE_CHANGE_THRESHOLD:
        return 0  # Up (Positive)
    elif daily_return <= -PRICE_CHANGE_THRESHOLD:
        return 1  # Down (Negative)
    else:
        return 2  # Neutral

# PyTorch custom dataset class (input, label)
class StockDataset(Dataset):

    def __init__(self, consolidated_data: Dict[str, pd.DataFrame], ea_dates: Dict[str, List[dt.datetime]], scaler=None, is_train=True):
        
        self.consolidated_data = consolidated_data
        self.ea_dates = ea_dates
        self.scaler = scaler
        self.is_train = is_train
        self.samples = []  # List of (ticker, start_date, ea_date, label) tuples

        # Indexing: Create all 30-day sequences leading up to each EA date
        for ticker, ea_df in ea_dates.items():
            if ticker not in consolidated_data:
                continue
            
            # Extract the list of EA dates for this ticker
            dates_list = ea_df['Earnings Date'].tolist()
        
            # For each EA date, determine if a valid 30-day sequence exists
            for ea_date in dates_list:
                ea_date = pd.to_datetime(ea_date, errors='coerce').tz_localize(None).normalize().date()
                
                # Go back SEQUENCE_LENGTH days to get the start date
                start_date = ea_date - dt.timedelta(days=SEQUENCE_LENGTH)

                # Check for completeness: Ensure 30 consecutive days of data exist
                # The consolidated DF is already aligned, so we just check for the start date
                if start_date in [consolidated_data[ticker].index[i].date() for i in range(len(consolidated_data[ticker].index))]:
                    # Check if the label is calculable (i.e., prices exist on EA day)
                    label = calculate_label(ticker, ea_date, self.consolidated_data)

                    if label != -1:
                        # Store the valid sample pointer
                        self.samples.append((ticker, start_date, ea_date, label))
        
        # Convert list of pointers to a DataFrame for easier slicing/tracking
        self.samples_df = pd.DataFrame(self.samples, columns=['ticker', 'start_date', 'ea_date', 'label'])
        
        # Scaling: Fit scaler only on training data (critical to avoid leakage)
        if self.is_train and self.scaler is None:
            self.scaler = StandardScaler()
            
            # Create a matrix of all feature data from the training windows
            all_features = []
            for _, row in self.samples_df.iterrows():
                df_window = consolidated_data[row['ticker']].loc[row['start_date']:row['ea_date'] - dt.timedelta(days=1)]
                all_features.append(df_window.values)
                
            # Training data matrix
            X_train_matrix = np.concatenate(all_features, axis=0)
            
            # Fit scaler (improves stability)
            self.scaler.fit(X_train_matrix)
            print(f"Scaler fitted on {X_train_matrix.shape[0]} daily training observations.")
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Retrieve the pointer for this index
        ticker, start_date, ea_date, label = self.samples[idx]
        
        # Extract 30-Day Sequence
        end_date = ea_date - dt.timedelta(days=1)
        df_window = self.consolidated_data[ticker].loc[start_date:end_date]
        
        # Convert to NumPy array
        X = df_window.values
        
        # Scale the features
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Convert to Tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        Y_tensor = torch.tensor(label, dtype=torch.long)

        # X_tensor shape will be (30, 21) -> Sequence Length, Feature Dimension (22 features - 1 adj_price)
        return X_tensor, Y_tensor

# Dataloader creation function
def create_dataloader(batch_size: int, is_train: bool, scaler: StandardScaler = None, **kwargs) -> Tuple[DataLoader, StandardScaler]:
    
    # Load the processed data files
    print("Loading feature and EA date files...")
    consolidated_data = pd.read_pickle(os.path.join(DATA_DIR, FEATURE_FILENAME))
    ea_dates = pd.read_pickle(os.path.join(DATA_DIR, EA_FILENAME))

    all_ea_dates = []
    for ticker, ea_df in ea_dates.items():
        # Ensure dates are clean Timestamps
        ea_df['Earnings Date'] = pd.to_datetime(ea_df['Earnings Date'], errors='coerce').dt.tz_localize(None).dt.normalize()
        ea_df.sort_values('Earnings Date', inplace=True)
        all_ea_dates.extend(ea_df['Earnings Date'].dropna().tolist())
        ea_dates[ticker] = ea_df
    
    # Get global max and min EA dates in the dataset
    max_date = max(all_ea_dates)
    min_date = max_date - dt.timedelta(days=365)
    
    # Calculate the date corresponding to 80% of the total duration (0.87 from experiments)
    GLOBAL_CUTOFF_DATE = min_date + (max_date - min_date) * 0.87
    
    # Filter dates based on train/test split
    filtered_ea_dates = {}
    for ticker, ea_df in ea_dates.items():
        if is_train:
            # Training: Use events on or before the cutoff date
            filtered_df = ea_df[ea_df['Earnings Date'] <= GLOBAL_CUTOFF_DATE].copy()
        else:
            # Testing: Use events strictly the cutoff date, prevent leakage
            filtered_df = ea_df[ea_df['Earnings Date'] > GLOBAL_CUTOFF_DATE].copy()

        if not filtered_df.empty:
            filtered_ea_dates[ticker] = filtered_df
    
    # Create the Dataset
    dataset = StockDataset(
        consolidated_data=consolidated_data, 
        ea_dates=filtered_ea_dates,
        scaler=scaler,  # Pass existing scaler for test/validation sets
        is_train=is_train
    )
    
    # Create the DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=is_train,  # Only shuffle the training set
        num_workers=4,     # Use multiple workers for faster data loading
        **kwargs
    )
    
    return dataloader, dataset.scaler

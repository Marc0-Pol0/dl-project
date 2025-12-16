import pandas as pd
import pprint
import os

# Dataset date from 2024-10-01 to 2025-08-01

# Load pickle
# obj = pd.read_pickle('./data/raw/DL_dataset.pkl')
obj_news = pd.read_pickle('./data/trainable/final_data.pkl')
obj = pd.read_pickle('./data/trainable/final_data_500.pkl')
obj_news_500 = pd.read_pickle('./data/processed/news_sentiment_500.pkl')
dl_dataset = pd.read_pickle('./data/raw/DL_dataset(2).pkl')

print(dl_dataset['B97MP0-R']['earnings']['Earnings Date'].head())

# print(obj_news_500['AAPL'].tail())

# --- Configuration ---
FEATURE_FILE = 'final_data_500.pkl'
DATA_DIR = './data/trainable/' 
FILE_PATH = os.path.join(DATA_DIR, FEATURE_FILE)

# --- Check Function ---
BANK_TICKERS = []
def check_for_nans_in_pickled_data(file_path):
    """
    Loads the dictionary of DataFrames and checks each DataFrame for NaN entries.
    """
    try:
        data_dict = pd.read_pickle(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    total_nans_found = False
    
    print(f"--- Checking {len(data_dict)} Tickers for NaN Entries ---")
    
    for ticker, df in data_dict.items():
        # Check if ANY value in the entire DataFrame is NaN
        if df.isnull().values.any():
            total_nans_found = True
            
            # Count NaNs per column
            nan_counts = df.isnull().sum()
            
            # Filter for columns that actually have NaNs
            nan_columns = nan_counts[nan_counts > 0]

            BANK_TICKERS.append(ticker)
            
            print(f"\n❌ NaN FOUND in Ticker: {ticker}")
            print(f"Total rows in DataFrame: {len(df)}")
            print("NaN counts per feature:")
            print(nan_columns.to_string())
            print("-" * 30)

    if not total_nans_found:
        print("\n✅ Success! No NaN entries found across any of the DataFrames.")

# check_for_nans_in_pickled_data(FILE_PATH)

# Define the specific imputation rules for the problematic bank tickers/features
IMPUTATION_RULES = {
    'Gross_Margin': 0.0,      # Set to zero (banks don't have it)
    'inven': 5000.0,          # Inventory: Placeholder value
    'cash_st': 50000.0        # Cash Short-Term: Placeholder value
}


# --- Cleaning Function ---

def clean_and_overwrite_data(file_path):
    """
    Loads the data, applies imputation rules, and overwrites the original pickle file.
    """
    print(f"Loading data from: {file_path}")
    
    try:
        consolidated_data = pd.read_pickle(file_path)
    except FileNotFoundError:
        print(f"FATAL ERROR: File not found at {file_path}. Cannot proceed.")
        return

    tickers_imputed_count = 0
    total_dataframes = len(consolidated_data)
    
    for ticker, df in consolidated_data.items():
        
        # --- 1. Apply Specific Imputation for Banks (JPM, BAC) ---
        if ticker in BANK_TICKERS:
            print(f"Applying specific imputation for {ticker}...")
            tickers_imputed_count += 1
            for feature, value in IMPUTATION_RULES.items():
                if feature in df.columns and df[feature].isnull().all():
                    # Set the missing feature column to the defined constant value
                    df[feature] = value
                    
        # --- 2. General Time-Series Imputation (Across ALL Tickers) ---
        
        # Fills NaNs using the previous valid observation (Forward Fill)
        df = df.fillna(method='ffill') 
        
        # Fills any remaining NaNs (e.g., at the very start) using the next valid observation (Backward Fill)
        df = df.fillna(method='bfill')
        
        # --- 3. Final Safety Net ---
        # Fills any remaining NaNs (e.g., entire columns still missing after bfill) with 0.
        df = df.fillna(0.0) 
        
        # Check to ensure no NaNs remain after cleaning
        if df.isnull().values.any():
            print(f"WARNING: Ticker {ticker} still contains NaNs after full cleaning. Investigate.")

        consolidated_data[ticker] = df
    
    # --- 4. Save the Cleaned Dictionary, Overwriting the Old File ---
    pd.to_pickle(consolidated_data, file_path)
    
    print("\n" + "="*50)
    print(f"✅ CLEANUP COMPLETE. Processed {total_dataframes} tickers.")
    print(f"Applied specific rules to {tickers_imputed_count} ticker(s).")
    print(f"File OVERWRITTEN successfully at: {file_path}")
    print("="*50)

# --- Execution ---
# clean_and_overwrite_data(FILE_PATH)
import pandas as pd
import os
from typing import Dict, Tuple

# --- Configuration ---
ROOT_DIR = './data/'
PROCESSED_DIR = os.path.join(ROOT_DIR, 'processed')
TRAINABLE_DIR = os.path.join(ROOT_DIR, 'trainable')

os.makedirs(TRAINABLE_DIR, exist_ok=True)

# File names (Files are assumed to be CLEAN, ALIGNED, but with 0, 1, 2 index)
STOCK_FILENAME = 'stock_values_500.pkl'
FUNDAMENTALS_FILENAME = 'fundamentals_500.pkl'
SENTIMENT_FILENAME = 'news_sentiment_500.pkl'
OUTPUT_FILENAME = 'final_data_500.pkl'

# The complete list of 21 features for the final DataFrame
FINAL_FEATURES = [
    'positive', 'negative', 'neutral',
    'Dividend_Yield', 'Net_Margin', 'Gross_Margin', 'ROE', 'ROA', 
    'Debt_to_Equity', 'eps_basic', 'assets', 'shldrs_eq', 'inven', 
    'cash_st', 'debt', 'net_debt', 'net_inc', 'oper_cf',
    'adj_price', 'MA3', 'MA6'
]

# --- 1. Data Loading and Indexing Helper ---

def set_date_index(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Sets the 'date' column as the DatetimeIndex for all DataFrames in the dictionary."""
    indexed_data = {}
    for ticker, df in data_dict.items():
        if 'date' in df.columns:
            # Convert date column to datetime type
            df['date'] = pd.to_datetime(df['date'])
            # Set the date column as the index
            df = df.set_index('date')
            # Ensure the index is sorted ascending immediately
            df.sort_index(inplace=True)
            
        indexed_data[ticker] = df
    return indexed_data

def load_data() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Loads all three files and ensures the index is the date."""
    print("Loading aligned data files...")
    stock_values = pd.read_pickle(os.path.join(PROCESSED_DIR, STOCK_FILENAME))
    fundamentals = pd.read_pickle(os.path.join(PROCESSED_DIR, FUNDAMENTALS_FILENAME))
    news_sentiment = pd.read_pickle(os.path.join(PROCESSED_DIR, SENTIMENT_FILENAME))

    print("Setting date as index for all DataFrames...")
    stock_values = set_date_index(stock_values)
    fundamentals = set_date_index(fundamentals)
    news_sentiment = set_date_index(news_sentiment)

    return stock_values, fundamentals, news_sentiment


# --- 2. Merging and Final Processing ---

def generate_final_data(stock_values, fundamentals, news_sentiment):
    """
    Merges all three daily-aligned datasets by the date index, and selects final features.
    """
    final_data = {}
    all_tickers = stock_values.keys()

    for ticker in all_tickers:
        
        if ticker not in fundamentals or ticker not in news_sentiment:
            print(f"Skipping {ticker}: Missing one or more required input datasets.")
            continue
            
        print(f"Merging ticker: {ticker}")
        
        # --- A. Start with Stock Values ---
        df_final = stock_values[ticker].copy()
        
        # --- B. Merge Sentiment and Fundamentals on Date Index ---
        
        # 1. Merge Sentiment (Inner Join on the common date index)
        df_final = df_final.merge(news_sentiment[ticker], 
                                  left_index=True, 
                                  right_index=True, 
                                  how='inner') 
                                  
        # 2. Merge Fundamentals
        df_final = df_final.merge(fundamentals[ticker], 
                                   left_index=True, 
                                   right_index=True, 
                                   how='inner')
        
        # --- C. Final Selection and Sort ---
        
        # Check for missing columns and select the final feature set.
        missing_cols = [col for col in FINAL_FEATURES if col not in df_final.columns]
        if missing_cols:
             print(f"FATAL WARNING: Ticker {ticker} is missing required columns: {missing_cols}. Skipping.")
             continue

        # Select the final features in the specified order (index is already sorted from set_date_index)
        df_final = df_final[FINAL_FEATURES].copy()
        
        if df_final.empty:
            print(f"Skipping {ticker}: No valid merged data remaining.")
            continue

        final_data[ticker] = df_final
        print(f"Ticker {ticker} successfully merged. Final shape: {df_final.shape}")

    return final_data

# --- 3. Execution ---

if __name__ == '__main__':
    # 1. Load and prepare data (Index set to date)
    stock_values, fundamentals, news_sentiment = load_data()
    
    # 2. Generate the final merged, cleaned, and featured data
    consolidated_data = generate_final_data(stock_values, fundamentals, news_sentiment)
    
    # 3. Save the final dictionary of DataFrames
    output_path = os.path.join(TRAINABLE_DIR, OUTPUT_FILENAME)
    print(f"\nSaving final data to: {output_path}")
    pd.to_pickle(consolidated_data, output_path)
    
    total_samples = sum(len(df) for df in consolidated_data.values())
    print(f"File successfully created with {len(consolidated_data)} tickers and a total of {total_samples} daily observations.")
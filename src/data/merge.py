import pandas as pd
import os
from typing import Dict, Tuple

# Path configurations
ROOT_DIR = './data/'
PROCESSED_DIR = os.path.join(ROOT_DIR, 'processed')
TRAINABLE_DIR = os.path.join(ROOT_DIR, 'trainable')

os.makedirs(TRAINABLE_DIR, exist_ok=True)

# File names (Clean and aligned data)
STOCK_FILENAME = 'stock_values_500.pkl'
FUNDAMENTALS_FILENAME = 'fundamentals_500.pkl'
SENTIMENT_FILENAME = 'news_sentiment_500.pkl'
OUTPUT_FILENAME = 'final_data_500.pkl'

# Complete list of 21 features for the final DataFrame
FINAL_FEATURES = [
    'positive', 'negative', 'neutral',
    'Dividend_Yield', 'Net_Margin', 'Gross_Margin', 'ROE', 'ROA', 
    'Debt_to_Equity', 'eps_basic', 'assets', 'shldrs_eq', 'inven', 
    'cash_st', 'debt', 'net_debt', 'net_inc', 'oper_cf',
    'adj_price', 'MA3', 'MA6'
]

# Data loading and preparation functions
def set_date_index(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
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

# Load the data files from the three sources
def load_data() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:

    # Loading
    print("Loading data files...")
    stock_values = pd.read_pickle(os.path.join(PROCESSED_DIR, STOCK_FILENAME))
    fundamentals = pd.read_pickle(os.path.join(PROCESSED_DIR, FUNDAMENTALS_FILENAME))
    news_sentiment = pd.read_pickle(os.path.join(PROCESSED_DIR, SENTIMENT_FILENAME))

    # Indexing by date
    print("Setting date as index for all DataFrames...")
    stock_values = set_date_index(stock_values)
    fundamentals = set_date_index(fundamentals)
    news_sentiment = set_date_index(news_sentiment)

    return stock_values, fundamentals, news_sentiment

# Merging into one finel dataset
def generate_final_data(stock_values, fundamentals, news_sentiment):

    final_data = {}
    all_tickers = stock_values.keys()

    for ticker in all_tickers:
        # Skip if ticker is not present in all datasets
        if ticker not in fundamentals or ticker not in news_sentiment:
            print(f"Skipping {ticker}: Missing one or more required input datasets.")
            continue
            
        print(f"Merging ticker: {ticker}")
        
        # Stock data
        df_final = stock_values[ticker].copy()
        
        # Merge news sentiment to stock data
        df_final = df_final.merge(news_sentiment[ticker], 
                                  left_index=True, 
                                  right_index=True, 
                                  how='inner') 
                                  
        # Add fundamentals to the merged dataframe
        df_final = df_final.merge(fundamentals[ticker], 
                                   left_index=True, 
                                   right_index=True, 
                                   how='inner')
        
        # Check for missing columns and select the final feature set
        missing_cols = [col for col in FINAL_FEATURES if col not in df_final.columns]
        if missing_cols:
             print(f"WARNING: Ticker {ticker} is missing required columns: {missing_cols}")
             continue

        # Select the final features in the specified order
        df_final = df_final[FINAL_FEATURES].copy()
        
        # Store the final merged DataFrame
        final_data[ticker] = df_final
        print(f"Ticker {ticker} successfully merged. Final shape: {df_final.shape}")

    return final_data

# Execute
if __name__ == '__main__':
    # Load
    stock_values, fundamentals, news_sentiment = load_data()
    
    # Generate final merged data
    consolidated_data = generate_final_data(stock_values, fundamentals, news_sentiment)
    
    # Save final dataset
    output_path = os.path.join(TRAINABLE_DIR, OUTPUT_FILENAME)
    print(f"\nSaving final data to: {output_path}")
    pd.to_pickle(consolidated_data, output_path)
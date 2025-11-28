# Preprocess raw data to create news and fundamentals dataset
import os
import pandas as pd
import datetime as dt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Paths
DATA_DIR = './data/raw'
OUTPUT_DIR = './data/processed/'

# Variables
start_date = dt.datetime(2024, 10, 1)
end_date = dt.datetime(2025, 8, 1)
FUND, STOCK, NEWS = False, False, False
SAVE = False

# Load raw data
obj = pd.read_pickle(DATA_DIR + '/DL_dataset.pkl')

# --- FUNDAMENTALS ---
if FUND:
    fundamentals_dict = {}

    # Iterate over each company in the dataset
    for id in obj.keys():

        # Extract ticker
        ticker = obj[id]['companyMapping']['Code'].item()
        ticker = ticker.replace('.US', '')

        # Extract ratios
        ratios = obj[id]['ratios']

        # Create dataframe with ratios
        fundamentals_df = pd.DataFrame(ratios)

        # Create a complete daily date range
        all_days = pd.date_range(start=start_date, end=end_date, freq='D')

        # Reindex dataframe to include all days
        fundamentals_df = fundamentals_df.set_index('date').reindex(all_days)

        # Add fundamentals to dataframe
        fundamentals = obj[id]['fundamentals']
        for key, values in fundamentals.items():
            # Ensure 'date' is datetime
            values['date'] = pd.to_datetime(values['date'])

            # Set date as index
            values = values.set_index('date')

            # Select last column of each fundamental dataframe
            last_col = values.iloc[:, -1]

            # Add it to fundamentals_df, aligning by date
            fundamentals_df[key] = last_col

        # Forward-fill missing values
        fundamentals_df = fundamentals_df.ffill().bfill()
        fundamentals_df = fundamentals_df.reset_index()               # index becomes a column named 'index'
        fundamentals_df = fundamentals_df.rename(columns={'index': 'date'})

        fundamentals_dict[ticker] = fundamentals_df

    # Save fundamentals dataset
    if SAVE:
        pd.to_pickle(fundamentals_dict, OUTPUT_DIR + 'fundamentals.pkl')


# --- STOCK VALUES ---
if STOCK:
    stock_values_dict = {}

    # Iterate over each company in the dataset
    for id in obj.keys():

        # Extract ticker
        ticker = obj[id]['companyMapping']['Code'].item()
        ticker = ticker.replace('.US', '')

        # Extract stock values
        stock_values = obj[id]['prices']
        stock_df = pd.DataFrame(stock_values)
        stock_df.pop('unadj_price')

        # Create a complete daily date range
        all_days = pd.date_range(start=start_date, end=end_date, freq='D')

        # Reindex dataframe to include all days
        stock_df = stock_df.set_index('date').reindex(all_days)

        # Forward-fill missing prices
        stock_df['adj_price'] = stock_df['adj_price'].ffill()

        # Compute moving averages, TODO: add more features if needed
        stock_df['MA3'] = stock_df['adj_price'].rolling(window=3, min_periods=1).mean()
        stock_df['MA6'] = stock_df['adj_price'].rolling(window=6, min_periods=1).mean()

        # Reset index and rename
        stock_df = stock_df.reset_index().rename(columns={'index': 'date'})
    
        stock_values_dict[ticker] = stock_df

    # Save stock values dataset
    if SAVE:
        pd.to_pickle(stock_values_dict, OUTPUT_DIR + 'stock_values.pkl')


# --- NEWS ---
news_dict = {}

# Iterate over each company in the dataset
if NEWS:
    for id in obj.keys():

        # Extract ticker
        ticker = obj[id]['companyMapping']['Code'].item()
        ticker = ticker.replace('.US', '')

        # Extract raw news
        news_raw = obj[id]['news']  

        # Convert event_time to datetime, letting pandas infer format
        news_raw['event_time'] = pd.to_datetime(news_raw['event_time'], utc=True, errors='coerce')

        # Drop rows that couldn't be parsed
        news_raw = news_raw.dropna(subset=['event_time'])

        # Now you can safely extract the date
        news_raw = news_raw.copy()
        news_raw['date'] = news_raw['event_time'].dt.date

        # Extract only the 'body' field from event_data
        
        news_raw['body'] = news_raw['event_data'].apply(lambda x: x.get('body', '') 
                                                        if isinstance(x, dict) else '')

        # Keep only needed columns
        news_raw = news_raw[['event_time', 'body']]

        # Create a daily date range
        all_days = pd.date_range(start=start_date, end=end_date, freq='D')
        news_df = pd.DataFrame(index=all_days)
        news_df.index.name = 'date'

        # Group news by date
        news_raw['date'] = news_raw['event_time'].dt.date
        news_grouped = news_raw.groupby('date')

        # Count number of news per day
        news_df['news_count'] = news_grouped.size().reindex(all_days.date, fill_value=0).values

        # Concatenate bodies per day
        news_df['body'] = news_grouped['body'] \
            .apply(lambda texts: " ".join(texts)) \
            .reindex(all_days.date, fill_value='') \
            .values

        # Reset index so 'date' is a column
        news_df = news_df.reset_index()
        
        if ticker == 'AAPL':
            print(news_df.head(20))

        # Save to dictionary
        news_dict[ticker] = news_df

    # Save news dataset
    if SAVE:
        pd.to_pickle(news_dict, OUTPUT_DIR + 'news.pkl')



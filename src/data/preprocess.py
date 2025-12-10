# Preprocess raw data to create news and fundamentals dataset
import os
import pandas as pd
import datetime as dt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sentiment import get_finbert_probabilities

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load the FinBERT model and tokenizer from ProsusAI
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

# Paths
DATA_DIR = './data/raw'
OUTPUT_DIR = './data/processed/'

# Variables
start_date = dt.datetime(2024, 10, 1)
end_date = dt.datetime(2025, 8, 1)
FUND, STOCK, NEWS, DATES = False, False, True, False
SAVE = False

# Load raw data
obj = pd.read_pickle(DATA_DIR + '/DL_dataset(1).pkl')

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
i = 0
# Iterate over each company in the dataset
if NEWS:
    for id in obj.keys():
        if i == 1: break
        # Extract ticker (Same as before)
        ticker = obj[id]['companyMapping']['Code'].item().replace('.US', '')

        # Extract raw news
        news_raw = obj[id]['news'].copy() # Use .copy() to avoid SettingWithCopyWarning

        # 1. CLEANING AND BODY EXTRACTION (Same as before)
        news_raw['event_time'] = pd.to_datetime(news_raw['event_time'], utc=True, errors='coerce')
        news_raw = news_raw.dropna(subset=['event_time'])
        
        news_raw['date'] = news_raw['event_time'].dt.date
        
        # Extract the 'body' for FinBERT
        news_raw['body'] = news_raw['event_data'].apply(
            lambda x: x.get('body', '') if isinstance(x, dict) else ''
        )
        
        # Drop news with an empty body
        news_raw = news_raw[news_raw['body'] != ''].reset_index(drop=True)

        # 2. CALCULATE SENTIMENT PER ARTICLE (CRUCIAL CHANGE)
        
        # Apply your function to each 'body' and expand the resulting dictionary 
        # into new columns named 'positive', 'negative', 'neutral'
        def sentiment_caller(text):
        # This wrapper passes the globally loaded objects to your function
            return get_finbert_probabilities(text, model, tokenizer, device)
        
        # The result_type='expand' parameter is critical here for efficiency.
        sentiment_series = news_raw['body'].apply(sentiment_caller)

        # 2. Convert the Series of Dictionaries to a DataFrame
        sentiment_scores = sentiment_series.apply(pd.Series)
        
        # 3. COMBINE DATA AND CALCULATE DAILY MEAN
        
        # Concatenate the sentiment scores with the date
        news_sentiment = pd.concat([news_raw[['date']], sentiment_scores], axis=1)

        # Calculate the mean of the three probability columns for each day
        daily_sentiment = news_sentiment.groupby('date')[
            ['positive', 'negative', 'neutral']
        ].mean().reset_index()

        # 4. REINDEX TO INCLUDE ALL DAYS (Same as before)
        all_days = pd.date_range(start=start_date, end=end_date, freq='D')
        
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])

        # Reindex and use FFILL/BFILL to handle days with NO news
        daily_sentiment = daily_sentiment.set_index('date').reindex(all_days)
        daily_sentiment = daily_sentiment.ffill().bfill() # Forward-fill then back-fill
        daily_sentiment = daily_sentiment.reset_index().rename(columns={'index': 'date'})

        # Save to dictionary
        news_dict[ticker] = daily_sentiment
        i += 1
        print(news_dict[ticker].head())
    # Save news dataset
    if SAVE:
        pd.to_pickle(news_dict, OUTPUT_DIR + 'news.pkl')


# --- DATES ---
if DATES:
    dates_dict = {}

    # Iterate over each company in the dataset
    for id in obj.keys():

        # Extract ticker
        ticker = obj[id]['companyMapping']['Code'].item()
        ticker = ticker.replace('.US', '')

        # Extract stock values
        ea_date = obj[id]['earnings']['Earnings Date']
        date_df = pd.DataFrame(ea_date)

        # Reset index and rename
        date_df = date_df.reset_index().rename(columns={'index': 'date'})
    
        dates_dict[ticker] = date_df

    # Save stock values dataset
    if SAVE:
        pd.to_pickle(dates_dict, OUTPUT_DIR + 'earning_dates.pkl')
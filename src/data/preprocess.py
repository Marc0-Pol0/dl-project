import os
import pandas as pd
import datetime as dt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sentiment import get_finbert_probabilities


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load the FinBERT model and tokenizer from ProsusAI
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

DATA_DIR = './data/raw'
OUTPUT_DIR = './data/processed/'

start_date = dt.datetime(2024, 10, 1)
end_date = dt.datetime(2025, 8, 1)
FUND, STOCK, NEWS, DATES = False, False, False, True
SAVE = True

obj = pd.read_pickle(DATA_DIR + '/DL_dataset(2).pkl')


# --- FUNDAMENTALS ---
if FUND:
    fundamentals_dict = {}

    for id in obj.keys():
        ticker = obj[id]['companyMapping']['Code'].item()
        ticker = ticker.replace('.US', '')

        ratios = obj[id]['ratios']
        fundamentals_df = pd.DataFrame(ratios)
        all_days = pd.date_range(start=start_date, end=end_date, freq='D')
        fundamentals_df = fundamentals_df.set_index('date').reindex(all_days)

        fundamentals = obj[id]['fundamentals']
        for key, values in fundamentals.items():
            values['date'] = pd.to_datetime(values['date'])
            values = values.set_index('date')
            last_col = values.iloc[:, -1]
            fundamentals_df[key] = last_col

        fundamentals_df = fundamentals_df.ffill().bfill()
        fundamentals_df = fundamentals_df.reset_index() 
        fundamentals_df = fundamentals_df.rename(columns={'index': 'date'})

        fundamentals_dict[ticker] = fundamentals_df

    if SAVE:
        pd.to_pickle(fundamentals_dict, OUTPUT_DIR + 'fundamentals_500.pkl')


# --- STOCK VALUES ---
if STOCK:
    stock_values_dict = {}

    for id in obj.keys():
        ticker = obj[id]['companyMapping']['Code'].item()
        ticker = ticker.replace('.US', '')

        stock_values = obj[id]['prices']
        stock_df = pd.DataFrame(stock_values)
        stock_df.pop('unadj_price')

        all_days = pd.date_range(start=start_date, end=end_date, freq='D')

        stock_df = stock_df.set_index('date').reindex(all_days)
        stock_df['adj_price'] = stock_df['adj_price'].ffill()

        # Compute moving averages, TODO: add more features if needed
        stock_df['MA3'] = stock_df['adj_price'].rolling(window=3, min_periods=1).mean()
        stock_df['MA6'] = stock_df['adj_price'].rolling(window=6, min_periods=1).mean()

        stock_df = stock_df.reset_index().rename(columns={'index': 'date'})
    
        stock_values_dict[ticker] = stock_df

    if SAVE:
        pd.to_pickle(stock_values_dict, OUTPUT_DIR + 'stock_values_500.pkl')


# --- NEWS ---
news_dict = {}
if NEWS:
    for id in obj.keys():
        ticker = obj[id]['companyMapping']['Code'].item().replace('.US', '')

        news_raw = obj[id]['news'].copy()  # Use .copy() to avoid SettingWithCopyWarning

        # 1. CLEANING AND BODY EXTRACTION
        news_raw['event_time'] = pd.to_datetime(news_raw['event_time'], utc=True, errors='coerce')
        news_raw = news_raw.dropna(subset=['event_time'])
        news_raw['date'] = news_raw['event_time'].dt.date
        news_raw['body'] = news_raw['event_data'].apply(
            lambda x: x.get('body', '') if isinstance(x, dict) else ''
        )
        news_raw = news_raw[news_raw['body'] != ''].reset_index(drop=True)

        # 2. CALCULATE SENTIMENT PER ARTICLE
        
        def sentiment_caller(text):
            return get_finbert_probabilities(text, model, tokenizer, device)
        
        sentiment_series = news_raw['body'].apply(sentiment_caller)

        sentiment_scores = sentiment_series.apply(pd.Series)
        
        # 3. COMBINE DATA AND CALCULATE DAILY MEAN
        
        news_sentiment = pd.concat([news_raw[['date']], sentiment_scores], axis=1)

        daily_sentiment = news_sentiment.groupby('date')[
            ['positive', 'negative', 'neutral']
        ].mean().reset_index()

        # 4. REINDEX TO INCLUDE ALL DAYS
        all_days = pd.date_range(start=start_date, end=end_date, freq='D')
        
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])

        daily_sentiment = daily_sentiment.set_index('date').reindex(all_days)
        daily_sentiment = daily_sentiment.ffill().bfill()
        daily_sentiment = daily_sentiment.reset_index().rename(columns={'index': 'date'})

        news_dict[ticker] = daily_sentiment

        print(news_dict[ticker].head())

    if SAVE:
        pd.to_pickle(news_dict, OUTPUT_DIR + 'news_sentiment_500.pkl')


# --- DATES ---
if DATES:
    dates_dict = {}
    for id in obj.keys():

        ticker = obj[id]['companyMapping']['Code'].item()
        ticker = ticker.replace('.US', '')

        ea_date = obj[id]['earnings']['Earnings Date']
        date_df = pd.DataFrame(ea_date)

        date_df = date_df.reset_index().rename(columns={'index': 'date'})
    
        dates_dict[ticker] = date_df

    if SAVE:
        pd.to_pickle(dates_dict, OUTPUT_DIR + 'earning_dates_500.pkl')

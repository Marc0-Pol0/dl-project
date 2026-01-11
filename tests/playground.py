import pandas as pd
import pprint
import os

# IMPORTANT: This script is designed to run experiments in a playground environment.
# It has no relevance to the actual model training or data loading pipeline

# Dataset date from 2024-10-01 to 2025-08-01

# Load pickle
# obj = pd.read_pickle('./data/raw/DL_dataset.pkl')
# obj_news = pd.read_pickle('./data/trainable/final_data.pkl')
# obj = pd.read_pickle('./data/trainable/final_data_500.pkl')
# obj_news_500 = pd.read_pickle('./data/processed/news_sentiment_500.pkl')
dl_dataset = pd.read_pickle('./data/trainable/final_data_500.pkl')

print(dl_dataset['AAPL'].head())

# print(obj_news_500['AAPL'].tail())

# --- Configuration ---
FEATURE_FILE = 'final_data_500.pkl'
DATA_DIR = './data/trainable/' 
FILE_PATH = os.path.join(DATA_DIR, FEATURE_FILE)
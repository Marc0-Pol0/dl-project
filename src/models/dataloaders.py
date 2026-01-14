import os
import datetime as dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


DATA_DIR = "./data/trainable/"
EA_FILENAME = "earning_dates_500.pkl"
FEATURE_FILENAME = "final_data_500.pkl"
SEQUENCE_LENGTH = 30  # days preceding the EA
PRICE_CHANGE_THRESHOLD = 0.03  # 3% threshold for UP/DOWN label

SENTIMENT_COLS = ["positive", "neutral", "negative"]


def calculate_label(ticker: str, ea_date: dt.datetime, consolidated_data: Dict[str, pd.DataFrame]) -> int:
    df = consolidated_data[ticker]

    # Get stock prices the day after (EA influences the market on the day after) and two days before the EA date
    ea_date = pd.to_datetime(ea_date) + dt.timedelta(days=1)
    pre_ea_date = ea_date - dt.timedelta(days=2)  # Two days before EA to reduce noise

    price_pre_ea = df.loc[pre_ea_date]["adj_price"]
    price_ea = df.loc[ea_date]["adj_price"]

    daily_return = (price_ea - price_pre_ea) / price_pre_ea

    if daily_return >= PRICE_CHANGE_THRESHOLD:
        return 0  # Up (Positive)
    elif daily_return <= -PRICE_CHANGE_THRESHOLD:
        return 1  # Down (Negative)
    else:
        return 2  # Neutral


class StockDataset(Dataset):
    def __init__(
        self,
        consolidated_data: Dict[str, pd.DataFrame],
        ea_dates: Dict[str, List[dt.datetime]],
        scaler=None,
        is_train=True,
        use_sentiment: bool = True,
    ):
        self.consolidated_data = consolidated_data
        self.ea_dates = ea_dates
        self.scaler = scaler
        self.is_train = is_train
        self.use_sentiment = bool(use_sentiment)

        self.samples = []  # List of (ticker, start_date, ea_date, label) tuples

        # Indexing: Create all 30-day sequences leading up to each EA date
        for ticker, ea_df in ea_dates.items():
            if ticker not in consolidated_data:
                continue

            dates_list = ea_df["Earnings Date"].tolist()

            # For each EA date, determine if a valid 30-day sequence exists
            for ea_date in dates_list:
                ea_date = pd.to_datetime(ea_date, errors="coerce").tz_localize(None).normalize().date()
                start_date = ea_date - dt.timedelta(days=SEQUENCE_LENGTH)

                if start_date in [consolidated_data[ticker].index[i].date() for i in range(len(consolidated_data[ticker].index))]:
                    label = calculate_label(ticker, ea_date, self.consolidated_data)
                    if label != -1:
                        self.samples.append((ticker, start_date, ea_date, label))

        self.samples_df = pd.DataFrame(self.samples, columns=["ticker", "start_date", "ea_date", "label"])

        # Fit scaler on TRAIN split only (on the feature columns actually used)
        if self.is_train and self.scaler is None:
            self.scaler = StandardScaler()

            all_features = []
            for _, row in self.samples_df.iterrows():
                df_window = consolidated_data[row["ticker"]].loc[row["start_date"] : row["ea_date"] - dt.timedelta(days=1)]
                df_window = self._select_features(df_window)
                all_features.append(df_window.values)

            X_train_matrix = np.concatenate(all_features, axis=0)
            self.scaler.fit(X_train_matrix)
            print(f"Scaler fitted on {X_train_matrix.shape[0]} daily training observations.")

    def _select_features(self, df_window: pd.DataFrame) -> pd.DataFrame:
        """Select feature columns depending on sentiment usage."""
        if self.use_sentiment:
            return df_window
        return df_window.drop(columns=SENTIMENT_COLS, errors="ignore")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ticker, start_date, ea_date, label = self.samples[idx]

        end_date = ea_date - dt.timedelta(days=1)
        df_window = self.consolidated_data[ticker].loc[start_date:end_date]
        df_window = self._select_features(df_window)  # <-- NEW

        X = df_window.values

        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        Y_tensor = torch.tensor(label, dtype=torch.long)

        # X_tensor shape will be (SEQUENCE_LENGTH, feature_dim)
        return X_tensor, Y_tensor


def create_dataloader(
    batch_size: int,
    is_train: bool,
    scaler: StandardScaler = None,
    use_sentiment: bool = True,
    **kwargs,
) -> Tuple[DataLoader, StandardScaler]:
    print("Loading feature and EA date files...")
    consolidated_data = pd.read_pickle(os.path.join(DATA_DIR, FEATURE_FILENAME))
    ea_dates = pd.read_pickle(os.path.join(DATA_DIR, EA_FILENAME))

    all_ea_dates = []
    for ticker, ea_df in ea_dates.items():
        ea_df["Earnings Date"] = pd.to_datetime(ea_df["Earnings Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
        ea_df.sort_values("Earnings Date", inplace=True)
        all_ea_dates.extend(ea_df["Earnings Date"].dropna().tolist())
        ea_dates[ticker] = ea_df

    max_date = max(all_ea_dates)
    min_date = max_date - dt.timedelta(days=365)

    GLOBAL_CUTOFF_DATE = min_date + (max_date - min_date) * 0.87

    filtered_ea_dates = {}
    for ticker, ea_df in ea_dates.items():
        if is_train:
            filtered_df = ea_df[ea_df["Earnings Date"] <= GLOBAL_CUTOFF_DATE].copy()
        else:
            filtered_df = ea_df[ea_df["Earnings Date"] > GLOBAL_CUTOFF_DATE].copy()

        if not filtered_df.empty:
            filtered_ea_dates[ticker] = filtered_df

    dataset = StockDataset(
        consolidated_data=consolidated_data,
        ea_dates=filtered_ea_dates,
        scaler=scaler,
        is_train=is_train,
        use_sentiment=use_sentiment,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=4,
        **kwargs,
    )

    return dataloader, dataset.scaler

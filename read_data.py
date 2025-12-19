import pickle
import pandas as pd


# file = "final_data_500.pkl"
file = "earning_dates_500.pkl"

with open(f"data/trainable/{file}", "rb") as f:
    data = pickle.load(f)

print("Type:", type(data))
print("Tickers:", list(data.keys()))
print("n_tickers:", len(data))

# Pick one example ticker
t = next(iter(data))
df = data[t]

print("\n--- Example ticker ---")
print("ticker:", t)
print("type(df):", type(df))
print("shape:", df.shape)
print("index:", type(df.index), "| name:", df.index.name)
print("date range:", df.index.min(), "→", df.index.max())

print("\ncolumns:", list(df.columns))
print("\ndtypes:\n", df.dtypes)

print("\nhead:\n", df.head(3))
print("\ntail:\n", df.tail(3))

print("\nmissing values per column (top 10):\n", df.isna().sum().sort_values(ascending=False).head(10))

# Quick descriptive stats for key columns if present
for col in ["adj_price", "positive", "negative", "neutral"]:
    if col in df.columns:
        s = df[col]
        print(f"\n{col} summary: min={s.min():.4g} mean={s.mean():.4g} max={s.max():.4g}")

# Check that sentiment columns look like probabilities
sent_cols = [c for c in ["positive","negative","neutral"] if c in df.columns]
if len(sent_cols) == 3:
    row_sum = df[sent_cols].sum(axis=1)
    print("\nSentiment row-sum: min/mean/max =",
          row_sum.min(), row_sum.mean(), row_sum.max())

# Compare across tickers (shapes, date ranges)
summary = []
for tick, dfi in data.items():
    summary.append({
        "ticker": tick,
        "rows": len(dfi),
        "cols": dfi.shape[1],
        "start": dfi.index.min(),
        "end": dfi.index.max(),
        "na_total": int(dfi.isna().sum().sum()),
    })
summary_df = pd.DataFrame(summary).sort_values("ticker")
print("\n--- Per-ticker summary (first 10) ---")
print(summary_df.head(10).to_string(index=False))

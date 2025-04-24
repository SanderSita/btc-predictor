import pandas as pd
from datetime import datetime
from index_list import fear_greed_indexes

# Load the dataset
df = pd.read_csv("bitcoin_prices_with_fear_greed.csv")
df["Open time"] = pd.to_datetime(df["Open time"], format="%Y-%m-%d")
df["Date"] = df["Open time"].dt.normalize()

# Convert the index list into a DataFrame
fg_df = pd.DataFrame(fear_greed_indexes, columns=["timestamp_ms", "fg_score"])
fg_df["timestamp"] = pd.to_datetime(fg_df["timestamp_ms"], unit="ms").dt.normalize()

# Prepare for merge
fg_df = fg_df.rename(columns={"fg_score": "fear_greed"})

# Merge on date — this will overwrite missing or existing values
df = df.drop(columns=["fear_greed"], errors="ignore")  # drop if already there
df = df.merge(fg_df[["timestamp", "fear_greed"]], left_on="Date", right_on="timestamp", how="left")

# Clean up
df.drop(columns=["timestamp", "Date"], inplace=True)

# Save final result
df.to_csv("bitcoin_prices_fear_greed_full.csv", index=False)
print("✅ Updated CSV with historical fear & greed index saved.")

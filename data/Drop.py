import pandas as pd

# Load CSV
df = pd.read_csv("data/future_predictions.csv")

# Remove specific columns
df = df.drop(columns=["home_goals","away_goals","score","result","venue","year","month","day","day_of_week","is_weekend","goal_difference","season_year"])

# Save back to CSV
df.to_csv("future_predictions.csv", index=False)

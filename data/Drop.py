import pandas as pd

# Load CSV
df = pd.read_csv("data\la_liga_results_10_years.csv")

# Remove specific columns
df = df.drop(columns=["ti", "att"])

# Save back to CSV
df.to_csv("la_liga.csv", index=False)

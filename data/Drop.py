import pandas as pd

# Load CSV
df = pd.read_csv("data\LaLiga.csv")

# Remove specific columns
df = df.drop(columns=["time" , "attendance"])

# Save back to CSV
df.to_csv("la_liga.csv", index=False)

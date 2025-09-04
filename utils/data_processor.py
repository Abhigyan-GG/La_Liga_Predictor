import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.teams = None
        
    def load_data(self):
        """Load and preprocess the La Liga data"""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            
            # Check for missing values
            missing_values = self.data.isnull().sum()
            if missing_values.any():
                print("Missing values detected:")
                print(missing_values[missing_values > 0])
                
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return False

        # Create a copy to avoid modifying the original
        df = self.data.copy()

        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Drop rows with invalid dates
        df = df.dropna(subset=['date'])

        # Ensure numeric columns (keep NaNs for future matches)
        df['home_goals'] = pd.to_numeric(df['home_goals'], errors='coerce')
        df['away_goals'] = pd.to_numeric(df['away_goals'], errors='coerce')

        # ⚠️ DO NOT drop NaN rows here – keep them for prediction
        goals_nan_count = df[['home_goals', 'away_goals']].isnull().sum().sum()
        if goals_nan_count > 0:
            print(f"Found {goals_nan_count} NaN values in goals columns (future matches will be kept).")

        # Extract date components
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Calculate goal difference (only for past matches)
        df['goal_difference'] = df['home_goals'] - df['away_goals']

        # Create result column (only for past matches)
        df['result'] = np.where(
            df['home_goals'].notna() & df['away_goals'].notna(),
            np.where(df['home_goals'] > df['away_goals'], 1,
                     np.where(df['home_goals'] < df['away_goals'], -1, 0)),
            np.nan
        )

        # Extract season from the season column if available
        if 'season' in df.columns:
            df['season_year'] = df['season'].str.split('-').str[0].astype(int)
        else:
            df['season_year'] = np.where(df['month'] > 7, df['year'], df['year'] - 1)

        # Get list of all teams
        self.teams = sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))

        # ⚠️ Don't drop NaN rows at the end either — keep them for predictions
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            print(f"Note: {nan_count} NaN values remain (likely future matches).")

        self.data = df
        print("Data preprocessing completed.")
        return True

    def get_processed_data(self):
        """Return the processed data"""
        return self.data
    
    def get_teams(self):
        """Return the list of teams"""
        return self.teams
    
    def save_processed_data(self, output_path):
        """Save processed data to CSV"""
        if self.data is not None:
            self.data.to_csv(output_path, index=False)
            print(f"Processed data saved to {output_path}")
            return True
        return False
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self, data):
        self.data = data
        self.features = None
        self.scalers = {}
        self.encoders = {}
        
    def create_team_features(self):
        """Create features based on team performance"""
        df = self.data.copy()
        
        # Ensure we have the required columns
        required_columns = ['date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        teams = sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))
        
        # Create team performance dictionaries
        team_stats = {}
        for team in teams:
            team_stats[team] = {
                'goals_scored': 0, 'goals_conceded': 0, 'wins': 0, 
                'draws': 0, 'losses': 0, 'matches_played': 0,
                'home_goals_scored': 0, 'home_goals_conceded': 0, 'home_wins': 0,
                'away_goals_scored': 0, 'away_goals_conceded': 0, 'away_wins': 0
            }
        
        # Lists to store features
        home_team_strength = []
        away_team_strength = []
        home_form = []
        away_form = []
        h2h_home_wins = []
        h2h_away_wins = []
        h2h_draws = []
        
        # Iterate through matches to calculate features
        for idx, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            date = row['date']
            season = row['season_year'] if 'season_year' in row else row['year']
            
            # Filter previous matches for form calculation
            prev_matches = df[(df['date'] < date) & (df['season_year'] == season)]
            
            # Home team form (last 5 matches)
            home_prev = prev_matches[
                ((prev_matches['home_team'] == home_team) | (prev_matches['away_team'] == home_team))
            ].tail(5)
            
            home_points = 0
            for _, match in home_prev.iterrows():
                if match['home_team'] == home_team:
                    if match['result'] == 1:
                        home_points += 3
                    elif match['result'] == 0:
                        home_points += 1
                else:
                    if match['result'] == -1:
                        home_points += 3
                    elif match['result'] == 0:
                        home_points += 1
            home_form.append(home_points / (len(home_prev) * 3) if len(home_prev) > 0 else 0.5)
            
            # Away team form (last 5 matches)
            away_prev = prev_matches[
                ((prev_matches['home_team'] == away_team) | (prev_matches['away_team'] == away_team))
            ].tail(5)
            
            away_points = 0
            for _, match in away_prev.iterrows():
                if match['home_team'] == away_team:
                    if match['result'] == 1:
                        away_points += 3
                    elif match['result'] == 0:
                        away_points += 1
                else:
                    if match['result'] == -1:
                        away_points += 3
                    elif match['result'] == 0:
                        away_points += 1
            away_form.append(away_points / (len(away_prev) * 3) if len(away_prev) > 0 else 0.5)
            
            # Head-to-head stats
            h2h_matches = prev_matches[
                ((prev_matches['home_team'] == home_team) & (prev_matches['away_team'] == away_team)) |
                ((prev_matches['home_team'] == away_team) & (prev_matches['away_team'] == home_team))
            ]
            
            home_wins = len(h2h_matches[h2h_matches['result'] == 1])
            away_wins = len(h2h_matches[h2h_matches['result'] == -1])
            draws = len(h2h_matches[h2h_matches['result'] == 0])
            total = len(h2h_matches)
            
            h2h_home_wins.append(home_wins / total if total > 0 else 0)
            h2h_away_wins.append(away_wins / total if total > 0 else 0)
            h2h_draws.append(draws / total if total > 0 else 0)
            
            # Team strength (season performance so far)
            home_goals_scored = team_stats[home_team]['goals_scored']
            home_goals_conceded = team_stats[home_team]['goals_conceded']
            home_matches = team_stats[home_team]['matches_played']
            
            away_goals_scored = team_stats[away_team]['goals_scored']
            away_goals_conceded = team_stats[away_team]['goals_conceded']
            away_matches = team_stats[away_team]['matches_played']
            
            home_strength = (home_goals_scored - home_goals_conceded) / home_matches if home_matches > 0 else 0
            away_strength = (away_goals_scored - away_goals_conceded) / away_matches if away_matches > 0 else 0
            
            home_team_strength.append(home_strength)
            away_team_strength.append(away_strength)
            
            # Update team stats with current match
            team_stats[home_team]['goals_scored'] += row['home_goals']
            team_stats[home_team]['goals_conceded'] += row['away_goals']
            team_stats[home_team]['matches_played'] += 1
            
            team_stats[away_team]['goals_scored'] += row['away_goals']
            team_stats[away_team]['goals_conceded'] += row['home_goals']
            team_stats[away_team]['matches_played'] += 1
            
            if row['result'] == 1:
                team_stats[home_team]['wins'] += 1
                team_stats[away_team]['losses'] += 1
            elif row['result'] == -1:
                team_stats[away_team]['wins'] += 1
                team_stats[home_team]['losses'] += 1
            else:
                team_stats[home_team]['draws'] += 1
                team_stats[away_team]['draws'] += 1
        
        # Add features to dataframe
        df['home_form'] = home_form
        df['away_form'] = away_form
        df['home_strength'] = home_team_strength
        df['away_strength'] = away_team_strength
        df['h2h_home_wins'] = h2h_home_wins
        df['h2h_away_wins'] = h2h_away_wins
        df['h2h_draws'] = h2h_draws
        
        # Check for NaN in features and replace with 0
        feature_columns = ['home_form', 'away_form', 'home_strength', 'away_strength', 
                          'h2h_home_wins', 'h2h_away_wins', 'h2h_draws']
        
        for col in feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Encode categorical variables
        if 'venue' in df.columns:
            self.encoders['venue'] = LabelEncoder()
            df['venue_encoded'] = self.encoders['venue'].fit_transform(df['venue'].fillna('Unknown'))
        
        self.encoders['home_team'] = LabelEncoder()
        self.encoders['away_team'] = LabelEncoder()
        
        df['home_team_encoded'] = self.encoders['home_team'].fit_transform(df['home_team'])
        df['away_team_encoded'] = self.encoders['away_team'].fit_transform(df['away_team'])
        
        # Select features for model
        feature_columns = [
            'home_team_encoded', 'away_team_encoded', 'home_form', 'away_form',
            'home_strength', 'away_strength', 'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
            'month', 'day_of_week', 'is_weekend'
        ]
        
        if 'venue_encoded' in df.columns:
            feature_columns.append('venue_encoded')
            
        self.features = df[feature_columns]
        
        # Check for NaN in features
        if self.features.isnull().any().any():
            print("Warning: NaN values found in features. Filling with 0.")
            self.features = self.features.fillna(0)
        
        # Scale features
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.features)
        
        # Return target variables
        y_home = df['home_goals'].values
        y_away = df['away_goals'].values
        y_result = df['result'].values
        
        return self.scaled_features, y_home, y_away, y_result
    
    def prepare_future_match(self, home_team, away_team, date, venue=None):
        """Prepare features for a future match prediction"""
        # This is a simplified version - in practice, you'd need to calculate
        # current form and stats for the teams
        
        features = {}
        features['home_team_encoded'] = self.encoders['home_team'].transform([home_team])[0]
        features['away_team_encoded'] = self.encoders['away_team'].transform([away_team])[0]
        
        # For a real implementation, you'd need to calculate these based on current season data
        features['home_form'] = 0.5  # Placeholder
        features['away_form'] = 0.5  # Placeholder
        features['home_strength'] = 0  # Placeholder
        features['away_strength'] = 0  # Placeholder
        features['h2h_home_wins'] = 0.3  # Placeholder
        features['h2h_away_wins'] = 0.3  # Placeholder
        features['h2h_draws'] = 0.4  # Placeholder
        
        date_obj = pd.to_datetime(date)
        features['month'] = date_obj.month
        features['day_of_week'] = date_obj.dayofweek
        features['is_weekend'] = 1 if features['day_of_week'] in [5, 6] else 0
        
        if venue and 'venue_encoded' in self.encoders:
            features['venue_encoded'] = self.encoders['venue'].transform([venue])[0]
        
        # Convert to array in the correct feature order
        feature_array = np.array([features[col] for col in self.features.columns])
        
        # Scale the features
        scaled_features = self.scaler.transform([feature_array])
        
        return scaled_features
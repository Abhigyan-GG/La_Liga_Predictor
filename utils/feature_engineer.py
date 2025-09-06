import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self, data):
        self.data = data
        self.features = None
        self.scalers = {}
        self.encoders = {}
        
    def calculate_elo_ratings(self, k_factor=32, initial_rating=1500):
        """Calculate ELO ratings for teams over time"""
        df = self.data.copy().sort_values('date')
        
        # Initialize team ratings
        teams = sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))
        elo_ratings = {team: initial_rating for team in teams}
        
        home_elo = []
        away_elo = []
        
        for _, row in df.iterrows():
            home_team, away_team = row['home_team'], row['away_team']
            
            # Record current ELO ratings
            home_elo.append(elo_ratings[home_team])
            away_elo.append(elo_ratings[away_team])
            
            # Update ELO ratings if result is available
            if not pd.isna(row['result']):
                home_rating = elo_ratings[home_team]
                away_rating = elo_ratings[away_team]
                
                # Expected scores
                expected_home = 1 / (1 + 10**((away_rating - home_rating) / 400))
                expected_away = 1 - expected_home
                
                # Actual scores based on result
                if row['result'] == 1:  # Home win
                    actual_home, actual_away = 1, 0
                elif row['result'] == -1:  # Away win
                    actual_home, actual_away = 0, 1
                else:  # Draw
                    actual_home, actual_away = 0.5, 0.5
                
                # Update ratings
                elo_ratings[home_team] += k_factor * (actual_home - expected_home)
                elo_ratings[away_team] += k_factor * (actual_away - expected_away)
        
        return home_elo, away_elo
    
    def calculate_form_metrics(self, window_sizes=[3, 5, 10]):
        """Calculate various form metrics with different time windows"""
        df = self.data.copy().sort_values('date')
        
        form_features = {}
        for window in window_sizes:
            form_features[f'home_form_{window}'] = []
            form_features[f'away_form_{window}'] = []
            form_features[f'home_goals_form_{window}'] = []
            form_features[f'away_goals_form_{window}'] = []
            form_features[f'home_conceded_form_{window}'] = []
            form_features[f'away_conceded_form_{window}'] = []
        
        for idx, row in df.iterrows():
            home_team, away_team = row['home_team'], row['away_team']
            current_date = row['date']
            
            # Get previous matches for both teams
            prev_matches = df[(df['date'] < current_date)]
            
            for window in window_sizes:
                # Home team form
                home_prev = prev_matches[
                    (prev_matches['home_team'] == home_team) | 
                    (prev_matches['away_team'] == home_team)
                ].tail(window)
                
                # Away team form
                away_prev = prev_matches[
                    (prev_matches['home_team'] == away_team) | 
                    (prev_matches['away_team'] == away_team)
                ].tail(window)
                
                # Calculate form metrics
                home_points = self._calculate_team_points(home_prev, home_team)
                away_points = self._calculate_team_points(away_prev, away_team)
                
                home_goals_scored = self._calculate_goals_scored(home_prev, home_team)
                away_goals_scored = self._calculate_goals_scored(away_prev, away_team)
                
                home_goals_conceded = self._calculate_goals_conceded(home_prev, home_team)
                away_goals_conceded = self._calculate_goals_conceded(away_prev, away_team)
                
                # Normalize by number of matches
                n_home_matches = len(home_prev) if len(home_prev) > 0 else 1
                n_away_matches = len(away_prev) if len(away_prev) > 0 else 1
                
                form_features[f'home_form_{window}'].append(home_points / (n_home_matches * 3))
                form_features[f'away_form_{window}'].append(away_points / (n_away_matches * 3))
                form_features[f'home_goals_form_{window}'].append(home_goals_scored / n_home_matches)
                form_features[f'away_goals_form_{window}'].append(away_goals_scored / n_away_matches)
                form_features[f'home_conceded_form_{window}'].append(home_goals_conceded / n_home_matches)
                form_features[f'away_conceded_form_{window}'].append(away_goals_conceded / n_away_matches)
        
        return form_features
    
    def _calculate_team_points(self, matches, team):
        """Calculate points for a team from match history"""
        points = 0
        for _, match in matches.iterrows():
            if pd.isna(match['result']):
                continue
                
            if match['home_team'] == team:
                if match['result'] == 1:
                    points += 3
                elif match['result'] == 0:
                    points += 1
            else:
                if match['result'] == -1:
                    points += 3
                elif match['result'] == 0:
                    points += 1
        return points
    
    def _calculate_goals_scored(self, matches, team):
        """Calculate goals scored by a team"""
        goals = 0
        for _, match in matches.iterrows():
            if pd.isna(match['home_goals']) or pd.isna(match['away_goals']):
                continue
                
            if match['home_team'] == team:
                goals += match['home_goals']
            else:
                goals += match['away_goals']
        return goals
    
    def _calculate_goals_conceded(self, matches, team):
        """Calculate goals conceded by a team"""
        goals = 0
        for _, match in matches.iterrows():
            if pd.isna(match['home_goals']) or pd.isna(match['away_goals']):
                continue
                
            if match['home_team'] == team:
                goals += match['away_goals']
            else:
                goals += match['home_goals']
        return goals
    
    def calculate_head_to_head_features(self, max_h2h_matches=10):
        """Enhanced head-to-head features"""
        df = self.data.copy().sort_values('date')
        
        h2h_features = {
            'h2h_home_wins': [],
            'h2h_away_wins': [],
            'h2h_draws': [],
            'h2h_home_goals_avg': [],
            'h2h_away_goals_avg': [],
            'h2h_total_goals_avg': [],
            'h2h_matches_count': []
        }
        
        for idx, row in df.iterrows():
            home_team, away_team = row['home_team'], row['away_team']
            current_date = row['date']
            
            # Find head-to-head matches
            h2h_matches = df[
                (df['date'] < current_date) &
                (((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
                 ((df['home_team'] == away_team) & (df['away_team'] == home_team)))
            ].tail(max_h2h_matches)
            
            if len(h2h_matches) == 0:
                # Default values when no history
                h2h_features['h2h_home_wins'].append(0.33)
                h2h_features['h2h_away_wins'].append(0.33)
                h2h_features['h2h_draws'].append(0.33)
                h2h_features['h2h_home_goals_avg'].append(1.5)
                h2h_features['h2h_away_goals_avg'].append(1.5)
                h2h_features['h2h_total_goals_avg'].append(3.0)
                h2h_features['h2h_matches_count'].append(0)
            else:
                home_wins = away_wins = draws = 0
                home_goals_total = away_goals_total = 0
                
                for _, match in h2h_matches.iterrows():
                    if pd.isna(match['result']):
                        continue
                        
                    # Adjust perspective based on home/away
                    if match['home_team'] == home_team:
                        if match['result'] == 1:
                            home_wins += 1
                        elif match['result'] == -1:
                            away_wins += 1
                        else:
                            draws += 1
                        home_goals_total += match['home_goals']
                        away_goals_total += match['away_goals']
                    else:
                        if match['result'] == 1:
                            away_wins += 1
                        elif match['result'] == -1:
                            home_wins += 1
                        else:
                            draws += 1
                        home_goals_total += match['away_goals']
                        away_goals_total += match['home_goals']
                
                total = len(h2h_matches)
                h2h_features['h2h_home_wins'].append(home_wins / total)
                h2h_features['h2h_away_wins'].append(away_wins / total)
                h2h_features['h2h_draws'].append(draws / total)
                h2h_features['h2h_home_goals_avg'].append(home_goals_total / total)
                h2h_features['h2h_away_goals_avg'].append(away_goals_total / total)
                h2h_features['h2h_total_goals_avg'].append((home_goals_total + away_goals_total) / total)
                h2h_features['h2h_matches_count'].append(total)
        
        return h2h_features
    
    def calculate_venue_features(self):
        """Calculate venue-specific features"""
        df = self.data.copy()
        
        venue_features = {
            'home_venue_advantage': [],
            'away_venue_performance': []
        }
        
        for idx, row in df.iterrows():
            venue = row.get('venue', 'Unknown')
            home_team = row['home_team']
            away_team = row['away_team']
            current_date = row['date']
            
            # Home team performance at this venue
            home_venue_matches = df[
                (df['date'] < current_date) &
                (df['venue'] == venue) &
                (df['home_team'] == home_team)
            ]
            
            if len(home_venue_matches) > 0:
                home_venue_points = self._calculate_team_points(home_venue_matches, home_team)
                home_venue_avg = home_venue_points / (len(home_venue_matches) * 3)
            else:
                home_venue_avg = 0.5
            
            # Away team performance at this venue
            away_venue_matches = df[
                (df['date'] < current_date) &
                (df['venue'] == venue) &
                (df['away_team'] == away_team)
            ]
            
            if len(away_venue_matches) > 0:
                away_venue_points = self._calculate_team_points(away_venue_matches, away_team)
                away_venue_avg = away_venue_points / (len(away_venue_matches) * 3)
            else:
                away_venue_avg = 0.3  # Away teams typically perform worse
            
            venue_features['home_venue_advantage'].append(home_venue_avg)
            venue_features['away_venue_performance'].append(away_venue_avg)
        
        return venue_features
    
    def calculate_momentum_features(self):
        """Calculate momentum and streak features"""
        df = self.data.copy().sort_values('date')
        
        momentum_features = {
            'home_win_streak': [],
            'away_win_streak': [],
            'home_unbeaten_streak': [],
            'away_unbeaten_streak': [],
            'home_momentum': [],
            'away_momentum': []
        }
        
        for idx, row in df.iterrows():
            home_team, away_team = row['home_team'], row['away_team']
            current_date = row['date']
            
            # Get recent matches
            recent_matches = df[(df['date'] < current_date)].tail(50)  # Look at last 50 matches
            
            # Calculate streaks for home team
            home_win_streak = self._calculate_win_streak(recent_matches, home_team)
            home_unbeaten_streak = self._calculate_unbeaten_streak(recent_matches, home_team)
            home_momentum = self._calculate_momentum(recent_matches, home_team)
            
            # Calculate streaks for away team
            away_win_streak = self._calculate_win_streak(recent_matches, away_team)
            away_unbeaten_streak = self._calculate_unbeaten_streak(recent_matches, away_team)
            away_momentum = self._calculate_momentum(recent_matches, away_team)
            
            momentum_features['home_win_streak'].append(home_win_streak)
            momentum_features['away_win_streak'].append(away_win_streak)
            momentum_features['home_unbeaten_streak'].append(home_unbeaten_streak)
            momentum_features['away_unbeaten_streak'].append(away_unbeaten_streak)
            momentum_features['home_momentum'].append(home_momentum)
            momentum_features['away_momentum'].append(away_momentum)
        
        return momentum_features
    
    def _calculate_win_streak(self, matches, team):
        """Calculate current winning streak for a team"""
        team_matches = matches[
            (matches['home_team'] == team) | (matches['away_team'] == team)
        ].tail(10)  # Look at last 10 matches
        
        streak = 0
        for _, match in team_matches.iloc[::-1].iterrows():  # Reverse to get most recent first
            if pd.isna(match['result']):
                break
                
            won = False
            if match['home_team'] == team and match['result'] == 1:
                won = True
            elif match['away_team'] == team and match['result'] == -1:
                won = True
                
            if won:
                streak += 1
            else:
                break
                
        return streak
    
    def _calculate_unbeaten_streak(self, matches, team):
        """Calculate current unbeaten streak for a team"""
        team_matches = matches[
            (matches['home_team'] == team) | (matches['away_team'] == team)
        ].tail(15)
        
        streak = 0
        for _, match in team_matches.iloc[::-1].iterrows():
            if pd.isna(match['result']):
                break
                
            lost = False
            if match['home_team'] == team and match['result'] == -1:
                lost = True
            elif match['away_team'] == team and match['result'] == 1:
                lost = True
                
            if not lost:
                streak += 1
            else:
                break
                
        return streak
    
    def _calculate_momentum(self, matches, team, decay_factor=0.9):
        """Calculate momentum using weighted recent performance"""
        team_matches = matches[
            (matches['home_team'] == team) | (matches['away_team'] == team)
        ].tail(8)
        
        if len(team_matches) == 0:
            return 0.5
        
        momentum = 0
        weight = 1
        total_weight = 0
        
        for _, match in team_matches.iloc[::-1].iterrows():
            if pd.isna(match['result']):
                continue
                
            points = 0
            if match['home_team'] == team:
                if match['result'] == 1:
                    points = 3
                elif match['result'] == 0:
                    points = 1
            else:
                if match['result'] == -1:
                    points = 3
                elif match['result'] == 0:
                    points = 1
            
            momentum += (points / 3) * weight
            total_weight += weight
            weight *= decay_factor
        
        return momentum / total_weight if total_weight > 0 else 0.5
    
    def calculate_league_position_features(self):
        """Calculate league position and relative strength features"""
        df = self.data.copy().sort_values('date')
        
        position_features = {
            'home_league_position': [],
            'away_league_position': [],
            'position_difference': [],
            'home_points_per_game': [],
            'away_points_per_game': []
        }
        
        for idx, row in df.iterrows():
            current_date = row['date']
            season_year = row.get('season_year', row.get('year', 2024))
            
            # Get all matches from current season up to current date
            season_matches = df[
                (df['date'] < current_date) & 
                (df.get('season_year', df.get('year', 2024)) == season_year)
            ]
            
            if len(season_matches) == 0:
                # Start of season defaults
                position_features['home_league_position'].append(10)
                position_features['away_league_position'].append(10)
                position_features['position_difference'].append(0)
                position_features['home_points_per_game'].append(1.5)
                position_features['away_points_per_game'].append(1.5)
                continue
            
            # Calculate league table
            league_table = self._calculate_league_table(season_matches)
            
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Get positions
            home_pos = league_table.get(home_team, {}).get('position', 10)
            away_pos = league_table.get(away_team, {}).get('position', 10)
            home_ppg = league_table.get(home_team, {}).get('points_per_game', 1.5)
            away_ppg = league_table.get(away_team, {}).get('points_per_game', 1.5)
            
            position_features['home_league_position'].append(home_pos)
            position_features['away_league_position'].append(away_pos)
            position_features['position_difference'].append(home_pos - away_pos)
            position_features['home_points_per_game'].append(home_ppg)
            position_features['away_points_per_game'].append(away_ppg)
        
        return position_features
    
    def _calculate_league_table(self, matches):
        """Calculate league table from match results"""
        teams = sorted(list(set(matches['home_team'].unique()) | set(matches['away_team'].unique())))
        table = {team: {'points': 0, 'played': 0, 'gd': 0} for team in teams}
        
        for _, match in matches.iterrows():
            if pd.isna(match['result']):
                continue
                
            home_team = match['home_team']
            away_team = match['away_team']
            
            table[home_team]['played'] += 1
            table[away_team]['played'] += 1
            
            if not pd.isna(match['home_goals']) and not pd.isna(match['away_goals']):
                table[home_team]['gd'] += match['home_goals'] - match['away_goals']
                table[away_team]['gd'] += match['away_goals'] - match['home_goals']
            
            if match['result'] == 1:  # Home win
                table[home_team]['points'] += 3
            elif match['result'] == -1:  # Away win
                table[away_team]['points'] += 3
            else:  # Draw
                table[home_team]['points'] += 1
                table[away_team]['points'] += 1
        
        # Calculate points per game and sort
        for team in table:
            table[team]['points_per_game'] = table[team]['points'] / max(table[team]['played'], 1)
        
        # Sort by points, then by goal difference
        sorted_teams = sorted(teams, key=lambda x: (table[x]['points'], table[x]['gd']), reverse=True)
        
        # Assign positions
        for i, team in enumerate(sorted_teams):
            table[team]['position'] = i + 1
        
        return table
    
    def create_advanced_features(self):
        """Create comprehensive feature set with all advanced metrics"""
        df = self.data.copy()
        
        print("Calculating ELO ratings...")
        home_elo, away_elo = self.calculate_elo_ratings()
        df['home_elo'] = home_elo
        df['away_elo'] = away_elo
        df['elo_difference'] = df['home_elo'] - df['away_elo']
        
        print("Calculating form metrics...")
        form_features = self.calculate_form_metrics()
        for feature_name, values in form_features.items():
            df[feature_name] = values
        
        print("Calculating head-to-head features...")
        h2h_features = self.calculate_head_to_head_features()
        for feature_name, values in h2h_features.items():
            df[feature_name] = values
        
        print("Calculating venue features...")
        venue_features = self.calculate_venue_features()
        for feature_name, values in venue_features.items():
            df[feature_name] = values
        
        print("Calculating momentum features...")
        momentum_features = self.calculate_momentum_features()
        for feature_name, values in momentum_features.items():
            df[feature_name] = values
        
        print("Calculating league position features...")
        position_features = self.calculate_league_position_features()
        for feature_name, values in position_features.items():
            df[feature_name] = values
        
        # Add time-based features
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['season_progress'] = df.groupby('season_year').cumcount() / df.groupby('season_year')['date'].transform('count')
        
        # Encode categorical variables
        self.encoders['home_team'] = LabelEncoder()
        self.encoders['away_team'] = LabelEncoder()
        
        df['home_team_encoded'] = self.encoders['home_team'].fit_transform(df['home_team'])
        df['away_team_encoded'] = self.encoders['away_team'].fit_transform(df['away_team'])
        
        if 'venue' in df.columns:
            self.encoders['venue'] = LabelEncoder()
            df['venue_encoded'] = self.encoders['venue'].fit_transform(df['venue'].fillna('Unknown'))
        
        # Select final feature set
        feature_columns = [
            # Team encoding
            'home_team_encoded', 'away_team_encoded',
            
            # ELO ratings
            'home_elo', 'away_elo', 'elo_difference',
            
            # Form features (multiple windows)
            'home_form_3', 'away_form_3', 'home_form_5', 'away_form_5', 'home_form_10', 'away_form_10',
            'home_goals_form_3', 'away_goals_form_3', 'home_goals_form_5', 'away_goals_form_5',
            'home_conceded_form_3', 'away_conceded_form_3', 'home_conceded_form_5', 'away_conceded_form_5',
            
            # Head-to-head
            'h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 
            'h2h_home_goals_avg', 'h2h_away_goals_avg', 'h2h_total_goals_avg', 'h2h_matches_count',
            
            # Venue
            'home_venue_advantage', 'away_venue_performance',
            
            # Momentum
            'home_win_streak', 'away_win_streak', 'home_unbeaten_streak', 'away_unbeaten_streak',
            'home_momentum', 'away_momentum',
            
            # League position
            'home_league_position', 'away_league_position', 'position_difference',
            'home_points_per_game', 'away_points_per_game',
            
            # Time features
            'month', 'day_of_week', 'is_weekend', 'season_progress'
        ]
        
        if 'venue_encoded' in df.columns:
            feature_columns.append('venue_encoded')
        
        # Filter columns that exist
        available_columns = [col for col in feature_columns if col in df.columns]
        self.features = df[available_columns]
        
        # Handle missing values
        self.features = self.features.fillna(self.features.median())
        
        # Store the updated data
        self.data = df
        
        print(f"Created {len(available_columns)} features")
        print("Feature engineering complete!")
        
        # Return target variables
        y_home = df['home_goals'].values
        y_away = df['away_goals'].values  
        y_result = df['result'].values
        
        return self.features.values, y_home, y_away, y_result
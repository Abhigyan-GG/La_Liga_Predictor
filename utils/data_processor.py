import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self):
        """Load the La Liga data from CSV"""
        try:
            print(f"Loading data from {self.data_path}")
            self.raw_data = pd.read_csv(self.data_path)
            print(f"Loaded {len(self.raw_data)} matches")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def clean_team_names(self, df):
        """Clean and standardize team names"""
        # Common team name variations to standardize
        team_mappings = {
            'Real Madrid CF': 'Real Madrid',
            'Real Madrid C.F.': 'Real Madrid',
            'FC Barcelona': 'Barcelona',
            'F.C. Barcelona': 'Barcelona',
            'Atlético Madrid': 'Atletico Madrid',
            'Atlético de Madrid': 'Atletico Madrid',
            'Athletic Bilbao': 'Athletic Club',
            'Athletic Club Bilbao': 'Athletic Club',
            'Real Sociedad de Fútbol': 'Real Sociedad',
            'Valencia CF': 'Valencia',
            'Sevilla FC': 'Sevilla',
            'Real Betis Balompié': 'Real Betis',
            'Villarreal CF': 'Villarreal',
            'RC Celta de Vigo': 'Celta Vigo',
            'RCD Espanyol': 'Espanyol',
            'RCD Espanyol de Barcelona': 'Espanyol',
            'Deportivo Alavés': 'Alaves',
            'Deportivo de La Coruña': 'Deportivo La Coruna',
            'RC Deportivo': 'Deportivo La Coruna',
            'UD Las Palmas': 'Las Palmas',
            'CA Osasuna': 'Osasuna',
            'Real Valladolid CF': 'Real Valladolid',
            'SD Eibar': 'Eibar',
            'Getafe CF': 'Getafe',
            'CD Leganés': 'Leganes',
            'Levante UD': 'Levante',
            'Granada CF': 'Granada',
            'Real Mallorca': 'Mallorca',
            'RCD Mallorca': 'Mallorca',
            'Cádiz CF': 'Cadiz',
            'SD Huesca': 'Huesca',
            'Elche CF': 'Elche',
            'Real Zaragoza': 'Zaragoza',
            'Racing de Santander': 'Racing Santander',
            'Xerez CD': 'Xerez',
            'CD Numancia': 'Numancia'
        }
        
        # Apply mappings
        df['home_team'] = df['home_team'].replace(team_mappings)
        df['away_team'] = df['away_team'].replace(team_mappings)
        
        # Remove extra whitespace and standardize
        df['home_team'] = df['home_team'].str.strip()
        df['away_team'] = df['away_team'].str.strip()
        
        return df
    
    def parse_dates(self, df):
        """Parse and standardize date formats"""
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Extract useful date components
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Calculate season year (season starts in August/September)
            df['season_year'] = df['year']
            df.loc[df['month'] >= 8, 'season_year'] = df['year']
            df.loc[df['month'] < 8, 'season_year'] = df['year'] - 1
            
            print(f"Date parsing complete. Date range: {df['date'].min()} to {df['date'].max()}")
            return df
            
        except Exception as e:
            print(f"Error parsing dates: {e}")
            return df
    
    def clean_scores_and_results(self, df):
        """Clean score data and ensure consistent result encoding"""
        # Handle missing or invalid scores
        df['home_goals'] = pd.to_numeric(df['home_goals'], errors='coerce')
        df['away_goals'] = pd.to_numeric(df['away_goals'], errors='coerce')
        
        # Remove matches with negative goals (data errors)
        df.loc[df['home_goals'] < 0, 'home_goals'] = np.nan
        df.loc[df['away_goals'] < 0, 'away_goals'] = np.nan
        
        # Remove matches with unrealistic scores (likely data errors)
        df.loc[df['home_goals'] > 10, 'home_goals'] = np.nan
        df.loc[df['away_goals'] > 10, 'away_goals'] = np.nan
        
        # Create consistent result encoding
        df['result'] = np.nan
        valid_scores = ~(pd.isna(df['home_goals']) | pd.isna(df['away_goals']))
        
        df.loc[valid_scores & (df['home_goals'] > df['away_goals']), 'result'] = 1   # Home win
        df.loc[valid_scores & (df['home_goals'] < df['away_goals']), 'result'] = -1  # Away win
        df.loc[valid_scores & (df['home_goals'] == df['away_goals']), 'result'] = 0  # Draw
        
        print(f"Score cleaning complete. {valid_scores.sum()} matches with valid scores")
        return df
    
    def handle_venue_data(self, df):
        """Clean and standardize venue information"""
        if 'venue' in df.columns:
            # Clean venue names
            df['venue'] = df['venue'].str.strip()
            df['venue'] = df['venue'].fillna('Unknown')
            
            # Standardize common venue name variations
            venue_mappings = {
                'Santiago Bernabéu': 'Santiago Bernabeu',
                'Camp Nou': 'Camp Nou',
                'Wanda Metropolitano': 'Metropolitano',
                'Estadio de la Cerámica': 'Estadio de la Ceramica',
                'San Mamés': 'San Mames'
            }
            
            df['venue'] = df['venue'].replace(venue_mappings)
        else:
            # Create venue column based on home team if not available
            df['venue'] = df['home_team'] + ' Stadium'
        
        return df
    
    def handle_attendance_data(self, df):
        """Clean attendance data"""
        if 'attendance' in df.columns:
            # Remove non-numeric characters and convert to numeric
            df['attendance'] = df['attendance'].astype(str).str.replace(',', '').str.replace('.', '')
            df['attendance'] = pd.to_numeric(df['attendance'], errors='coerce')
            
            # Remove unrealistic attendance values
            df.loc[df['attendance'] > 100000, 'attendance'] = np.nan
            df.loc[df['attendance'] < 0, 'attendance'] = np.nan
        
        return df
    
    def remove_invalid_matches(self, df):
        """Remove matches with invalid or incomplete data"""
        initial_count = len(df)
        
        # Remove matches with missing team names
        df = df.dropna(subset=['home_team', 'away_team'])
        
        # Remove matches where team plays against itself (data error)
        df = df[df['home_team'] != df['away_team']]
        
        # Remove matches with invalid dates
        df = df.dropna(subset=['date'])
        
        # Sort by date to ensure chronological order
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"Removed {initial_count - len(df)} invalid matches. {len(df)} matches remaining")
        return df
    
    def add_derived_features(self, df):
        """Add useful derived features"""
        # Goal-related features
        df['total_goals'] = df['home_goals'] + df['away_goals']
        df['goal_difference'] = df['home_goals'] - df['away_goals']
        
        # Match characteristics
        df['high_scoring'] = (df['total_goals'] > 3).astype(int)
        df['low_scoring'] = (df['total_goals'] < 2).astype(int)
        df['competitive'] = (np.abs(df['goal_difference']) <= 1).astype(int)
        
        # Time-based features
        df['is_midweek'] = (~df['day_of_week'].isin([5, 6])).astype(int)
        df['month_category'] = df['month'].map({
            8: 'early_season', 9: 'early_season', 10: 'early_season',
            11: 'mid_season', 12: 'mid_season', 1: 'mid_season',
            2: 'late_season', 3: 'late_season', 4: 'late_season', 5: 'late_season'
        })
        
        return df
    
    def validate_data_quality(self, df):
        """Validate data quality and report issues"""
        print("\n=== Data Quality Report ===")
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['date', 'home_team', 'away_team']).sum()
        print(f"Duplicate matches: {duplicates}")
        
        # Check match distribution by season
        if 'season_year' in df.columns:
            season_counts = df.groupby('season_year').size()
            print(f"\nMatches by season:")
            for season, count in season_counts.items():
                print(f"  {season}-{season+1}: {count} matches")
                if count < 300:  # La Liga typically has ~380 matches per season
                    print(f"    WARNING: Low match count for season {season}-{season+1}")
        
        # Check team distribution
        home_teams = set(df['home_team'].unique())
        away_teams = set(df['away_team'].unique())
        all_teams = home_teams | away_teams
        print(f"\nUnique teams: {len(all_teams)}")
        
        # Check for teams that only appear as home or away
        home_only = home_teams - away_teams
        away_only = away_teams - home_teams
        if home_only:
            print(f"Teams only appearing at home: {home_only}")
        if away_only:
            print(f"Teams only appearing away: {away_only}")
        
        # Check result distribution
        if 'result' in df.columns:
            result_counts = df['result'].value_counts()
            total_with_results = result_counts.sum()
            print(f"\nResult distribution:")
            print(f"  Home wins: {result_counts.get(1, 0)} ({result_counts.get(1, 0)/total_with_results*100:.1f}%)")
            print(f"  Draws: {result_counts.get(0, 0)} ({result_counts.get(0, 0)/total_with_results*100:.1f}%)")
            print(f"  Away wins: {result_counts.get(-1, 0)} ({result_counts.get(-1, 0)/total_with_results*100:.1f}%)")
            print(f"  Missing results: {df['result'].isna().sum()}")
        
        # Check for missing data
        print(f"\nMissing data:")
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")
        
        print("=== End Data Quality Report ===\n")
        
        return df
    
    def preprocess_data(self):
        """Main preprocessing pipeline"""
        if self.raw_data is None:
            print("No data loaded. Please load data first.")
            return False
        
        try:
            df = self.raw_data.copy()
            
            print("Starting data preprocessing...")
            
            # Apply all cleaning steps
            df = self.clean_team_names(df)
            df = self.parse_dates(df)
            df = self.clean_scores_and_results(df)
            df = self.handle_venue_data(df)
            df = self.handle_attendance_data(df)
            df = self.remove_invalid_matches(df)
            df = self.add_derived_features(df)
            df = self.validate_data_quality(df)
            
            self.processed_data = df
            print("Data preprocessing completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return False
    
    def get_processed_data(self):
        """Return the processed data"""
        return self.processed_data
    
    def save_processed_data(self, output_path):
        """Save processed data to CSV"""
        if self.processed_data is not None:
            self.processed_data.to_csv(output_path, index=False)
            print(f"Processed data saved to {output_path}")
        else:
            print("No processed data to save")
    
    def get_team_statistics(self):
        """Get basic statistics for each team"""
        if self.processed_data is None:
            print("No processed data available")
            return None
        
        df = self.processed_data
        teams = sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))
        
        team_stats = {}
        for team in teams:
            home_matches = df[df['home_team'] == team]
            away_matches = df[df['away_team'] == team]
            
            # Basic counts
            total_matches = len(home_matches) + len(away_matches)
            home_matches_count = len(home_matches)
            away_matches_count = len(away_matches)
            
            # Goals
            goals_scored = home_matches['home_goals'].sum() + away_matches['away_goals'].sum()
            goals_conceded = home_matches['away_goals'].sum() + away_matches['home_goals'].sum()
            
            # Results
            home_wins = len(home_matches[home_matches['result'] == 1])
            away_wins = len(away_matches[away_matches['result'] == -1])
            home_draws = len(home_matches[home_matches['result'] == 0])
            away_draws = len(away_matches[away_matches['result'] == 0])
            
            total_wins = home_wins + away_wins
            total_draws = home_draws + away_draws
            total_losses = total_matches - total_wins - total_draws
            
            team_stats[team] = {
                'total_matches': total_matches,
                'home_matches': home_matches_count,
                'away_matches': away_matches_count,
                'wins': total_wins,
                'draws': total_draws,
                'losses': total_losses,
                'goals_scored': goals_scored,
                'goals_conceded': goals_conceded,
                'goal_difference': goals_scored - goals_conceded,
                'win_percentage': total_wins / total_matches * 100 if total_matches > 0 else 0,
                'goals_per_match': goals_scored / total_matches if total_matches > 0 else 0
            }
        
        return team_stats

# Example usage
if __name__ == "__main__":
    processor = ImprovedDataProcessor('data/LaLiga.csv')
    
    if processor.load_data():
        if processor.preprocess_data():
            processor.save_processed_data('data/LaLiga_processed.csv')
            
            # Get team statistics
            team_stats = processor.get_team_statistics()
            if team_stats:
                print("\nTop 5 teams by win percentage:")
                sorted_teams = sorted(team_stats.items(), key=lambda x: x[1]['win_percentage'], reverse=True)
                for i, (team, stats) in enumerate(sorted_teams[:5]):
                    print(f"{i+1}. {team}: {stats['win_percentage']:.1f}% wins, {stats['goals_per_match']:.1f} goals/match")
        else:
            print("Data preprocessing failed")
    else:
        print("Data loading failed")
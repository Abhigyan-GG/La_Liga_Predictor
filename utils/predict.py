import pandas as pd
import numpy as np
import pickle
import sys
import os
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

try:
    from data_processor import DataProcessor
    from feature_engineer import FeatureEngineer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure data_processor.py and feature_engineer.py are in the utils directory")
    sys.exit(1)


class MatchPredictor:
    def __init__(self, goal_model_path, result_model_path, feature_engineer_path):
        """Initialize the predictor with trained models"""
        with open(goal_model_path, 'rb') as f:
            self.goal_model = pickle.load(f)
        
        with open(result_model_path, 'rb') as f:
            self.result_model = pickle.load(f)
        
        with open(feature_engineer_path, 'rb') as f:
            self.feature_engineer = pickle.load(f)
    
    def predict_match(self, home_team, away_team, date, venue=None):
        """Predict the outcome of a single match"""
        # Prepare features for the match
        features = self.feature_engineer.prepare_future_match(
            home_team, away_team, date, venue
        )
        
        # Predict goals
        goals_pred = self.goal_model.predict(features)
        home_goals = round(goals_pred[0][0])
        away_goals = round(goals_pred[0][1])
        
        # Ensure goals are non-negative
        home_goals = max(0, home_goals)
        away_goals = max(0, away_goals)
        
        # Predict result
        result_pred = self.result_model.predict(features)[0]
        result_map = {1: "Home Win", 0: "Draw", -1: "Away Win"}
        result = result_map[result_pred]
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'date': date,
            'predicted_home_goals': home_goals,
            'predicted_away_goals': away_goals,
            'predicted_score': f"{home_goals}-{away_goals}",
            'predicted_result': result
        }
    
    def predict_matches(self, matches_df):
        """Predict outcomes for multiple matches"""
        predictions = []
        
        for _, row in matches_df.iterrows():
            prediction = self.predict_match(
                row['home_team'], 
                row['away_team'], 
                row['date'],
                row.get('venue', None)
            )
            predictions.append(prediction)
        
        return pd.DataFrame(predictions)

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = MatchPredictor(
        'models/goal_model.pkl',
        'models/result_model.pkl',
        'models/feature_engineer.pkl'
    )
    
    # Example: Predict a single match
    prediction = predictor.predict_match(
        "Barcelona", 
        "Real Madrid", 
        "2024-10-28"
    )
    
    print("Match Prediction:")
    for key, value in prediction.items():
        print(f"{key}: {value}")
    
    # Example: Predict multiple matches from a CSV
    future_matches = pd.read_csv('data/future_matches.csv')
    predictions = predictor.predict_matches(future_matches)
    
    print("\nAll Predictions:")
    print(predictions.to_string(index=False))
    
    # Save predictions
    predictions.to_csv('data/predictions.csv', index=False)
    print("\nPredictions saved to data/predictions.csv")
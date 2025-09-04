import pandas as pd
import numpy as np
import pickle
import sys
import os
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add the utils directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

try:
    from data_processor import DataProcessor
    from feature_engineer import FeatureEngineer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure data_processor.py and feature_engineer.py are in the utils directory")
    sys.exit(1)



class ModelEvaluator:
    def __init__(self, data_path, goal_model_path, result_model_path, feature_engineer_path):
        """Initialize the evaluator with trained models and data"""
        self.data_path = data_path
        
        with open(goal_model_path, 'rb') as f:
            self.goal_model = pickle.load(f)
        
        with open(result_model_path, 'rb') as f:
            self.result_model = pickle.load(f)
        
        with open(feature_engineer_path, 'rb') as f:
            self.feature_engineer = pickle.load(f)
    
    def load_and_preprocess_data(self):
        """Load and preprocess the data for evaluation"""
        from data_processor import DataProcessor
        
        processor = DataProcessor(self.data_path)
        if not processor.load_data():
            return False, None, None, None, None
        
        if not processor.preprocess_data():
            return False, None, None, None, None
        
        data = processor.get_processed_data()
        
        # Engineer features
        X, y_home, y_away, y_result = self.feature_engineer.create_team_features()
        y_goals = np.column_stack((y_home, y_away))
        
        return True, X, y_goals, y_result, data
    
    def evaluate_models(self):
        """Evaluate the performance of the trained models"""
        success, X, y_goals, y_result, data = self.load_and_preprocess_data()
        if not success:
            print("Failed to load data for evaluation")
            return
        
        # Predict goals
        goals_pred = self.goal_model.predict(X)
        
        # Calculate MAE for home and away goals
        mae_home = mean_absolute_error(y_goals[:, 0], goals_pred[:, 0])
        mae_away = mean_absolute_error(y_goals[:, 1], goals_pred[:, 1])
        
        print("Goal Prediction Performance:")
        print(f"Home Goals MAE: {mae_home:.2f}")
        print(f"Away Goals MAE: {mae_away:.2f}")
        print(f"Overall MAE: {(mae_home + mae_away) / 2:.2f}")
        
        # Predict results
        result_pred = self.result_model.predict(X)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_result, result_pred)
        
        print("\nResult Prediction Performance:")
        print(f"Accuracy: {accuracy:.2f}")
        
        # Confusion matrix
        result_map = {1: "Home Win", 0: "Draw", -1: "Away Win"}
        y_result_labels = [result_map[x] for x in y_result]
        result_pred_labels = [result_map[x] for x in result_pred]
        
        cm = confusion_matrix(y_result_labels, result_pred_labels, labels=["Home Win", "Draw", "Away Win"])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=["Home Win", "Draw", "Away Win"],
                   yticklabels=["Home Win", "Draw", "Away Win"])
        plt.title('Confusion Matrix - Result Prediction')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('evaluation/confusion_matrix.png')
        plt.show()
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_result_labels, result_pred_labels))
        
        # Analyze performance by season
        if 'season_year' in data.columns:
            seasons = sorted(data['season_year'].unique())
            season_accuracy = []
            
            for season in seasons:
                season_mask = data['season_year'] == season
                if season_mask.sum() > 0:
                    season_accuracy.append(accuracy_score(
                        y_result[season_mask], result_pred[season_mask]
                    ))
                else:
                    season_accuracy.append(0)
            
            plt.figure(figsize=(10, 6))
            plt.plot(seasons, season_accuracy, marker='o')
            plt.title('Model Accuracy by Season')
            plt.xlabel('Season')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('evaluation/accuracy_by_season.png')
            plt.show()
        
        # Analyze performance by team
        teams = sorted(list(set(data['home_team'].unique()) | set(data['away_team'].unique())))
        team_accuracy = {}
        
        for team in teams:
            team_mask = (data['home_team'] == team) | (data['away_team'] == team)
            if team_mask.sum() > 0:
                team_accuracy[team] = accuracy_score(
                    y_result[team_mask], result_pred[team_mask]
                )
        
        # Get top and bottom 10 teams by accuracy
        sorted_teams = sorted(team_accuracy.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 Teams by Prediction Accuracy:")
        for team, acc in sorted_teams[:10]:
            print(f"{team}: {acc:.2f}")
        
        print("\nBottom 10 Teams by Prediction Accuracy:")
        for team, acc in sorted_teams[-10:]:
            print(f"{team}: {acc:.2f}")
        
        return {
            'goal_mae_home': mae_home,
            'goal_mae_away': mae_away,
            'result_accuracy': accuracy
        }

# Example usage
if __name__ == "__main__":
    evaluator = ModelEvaluator(
        'data/LaLiga.csv',
        'models/goal_model.pkl',
        'models/result_model.pkl',
        'models/feature_engineer.pkl'
    )
    
    evaluation_results = evaluator.evaluate_models()
    
    # Save evaluation results
    import json
    with open('evaluation/results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print("Evaluation completed. Results saved to evaluation/results.json")
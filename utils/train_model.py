import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_processor import DataProcessor
    from feature_engineer import FeatureEngineer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure data_processor.py and feature_engineer.py are in the same directory")
    sys.exit(1)


class ModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.processor = DataProcessor(data_path)
        self.feature_engineer = None
        self.goal_model = None
        self.result_model = None
        self.future_X = None
        self.future_matches = None

    def load_and_preprocess_data(self):
        """Load and preprocess the data"""
        if not self.processor.load_data():
            return False
        if not self.processor.preprocess_data():
            return False
        return True

    def train_models(self):
        """Train the prediction models"""
        # Get processed data
        data = self.processor.get_processed_data()

        # Engineer features
        self.feature_engineer = FeatureEngineer(data)
        X, y_home, y_away, y_result = self.feature_engineer.create_team_features()

        # Split past vs future
        is_past = ~(np.isnan(y_home) | np.isnan(y_away) | np.isnan(y_result))
        X_past, y_home_past, y_away_past, y_result_past = (
            X[is_past], y_home[is_past], y_away[is_past], y_result[is_past]
        )

        # Store future matches for prediction
        self.future_X = X[~is_past]
        self.future_matches = self.feature_engineer.data.loc[~is_past]

        # Handle NaN in features
        if np.isnan(X_past).any():
            print("NaN values found in features. Imputing...")
            imputer = SimpleImputer(strategy='mean')
            X_past = imputer.fit_transform(X_past)

        # Combine goals for multi-output regression
        y_goals = np.column_stack((y_home_past, y_away_past))

        # Train/test split
        X_train, X_test, y_goals_train, y_goals_test = train_test_split(
            X_past, y_goals, test_size=0.2, random_state=42
        )
        _, _, y_result_train, y_result_test = train_test_split(
            X_past, y_result_past, test_size=0.2, random_state=42
        )

        print("Training goal prediction model...")
        self.goal_model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, random_state=42)
        )
        self.goal_model.fit(X_train, y_goals_train)

        goals_pred = self.goal_model.predict(X_test)
        mae_home = mean_absolute_error(y_goals_test[:, 0], goals_pred[:, 0])
        mae_away = mean_absolute_error(y_goals_test[:, 1], goals_pred[:, 1])
        print(f"Goal prediction MAE - Home: {mae_home:.2f}, Away: {mae_away:.2f}")

        print("Training result prediction model...")
        self.result_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.result_model.fit(X_train, y_result_train)

        result_pred = self.result_model.predict(X_test)
        accuracy = accuracy_score(y_result_test, result_pred)
        print(f"Result prediction accuracy: {accuracy:.2f}")

        return True

    def predict_future_matches(self):
        """Predict outcomes for matches with missing goals/results"""
        if self.future_X is None or len(self.future_X) == 0:
            print("No future matches to predict.")
            return None

        goals_pred = self.goal_model.predict(self.future_X)
        home_goals_pred = goals_pred[:, 0].round().astype(int)
        away_goals_pred = goals_pred[:, 1].round().astype(int)

        result_pred = self.result_model.predict(self.future_X)

        predictions = self.future_matches.copy()
        predictions["pred_home_goals"] = home_goals_pred
        predictions["pred_away_goals"] = away_goals_pred
        predictions["pred_result"] = result_pred

        return predictions

    def save_models(self, goal_model_path, result_model_path, feature_engineer_path):
        """Save the trained models and feature engineer"""
        os.makedirs(os.path.dirname(goal_model_path), exist_ok=True)

        with open(goal_model_path, 'wb') as f:
            pickle.dump(self.goal_model, f)

        with open(result_model_path, 'wb') as f:
            pickle.dump(self.result_model, f)

        with open(feature_engineer_path, 'wb') as f:
            pickle.dump(self.feature_engineer, f)

        print(f"Models saved to {goal_model_path} and {result_model_path}")
        print(f"Feature engineer saved to {feature_engineer_path}")


# Example usage
if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)

    trainer = ModelTrainer('data/LaLiga.csv')

    if trainer.load_and_preprocess_data():
        if trainer.train_models():
            trainer.save_models(
                'models/goal_model.pkl',
                'models/result_model.pkl',
                'models/feature_engineer.pkl'
            )

            preds = trainer.predict_future_matches()
            if preds is not None:
                preds.to_csv("data/future_predictions.csv", index=False)
                print("Future match predictions saved to data/future_predictions.csv")
        else:
            print("Model training failed due to data issues")

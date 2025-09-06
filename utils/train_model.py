import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, make_scorer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import pickle
import warnings
from datetime import datetime
import json
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


class AdvancedModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.processor = DataProcessor(data_path)
        self.feature_engineer = None
        self.goal_model = None
        self.result_model = None
        self.future_X = None
        self.future_matches = None
        self.best_params = {}
        self.feature_importance = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data"""
        if not self.processor.load_data():
            return False
        if not self.processor.preprocess_data():
            return False
        return True

    def create_time_series_split(self, X, y, n_splits=5):
        """Create time series splits for cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return tscv.split(X, y)

    def optimize_hyperparameters(self, X_train, y_goals_train, y_result_train):
        """Perform hyperparameter optimization for both models"""
        print("Optimizing hyperparameters...")
        
        # Goal prediction hyperparameters (Random Forest)
        rf_goal_params = {
            'estimator__n_estimators': [100, 200, 300],
            'estimator__max_depth': [10, 15, 20, None],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__min_samples_leaf': [1, 2, 4],
            'estimator__max_features': ['sqrt', 'log2', None]
        }
        
        # Alternative: XGBoost for goals
        xgb_goal_params = {
            'estimator__n_estimators': [100, 200, 300],
            'estimator__max_depth': [3, 6, 9],
            'estimator__learning_rate': [0.01, 0.1, 0.2],
            'estimator__subsample': [0.8, 0.9, 1.0],
            'estimator__colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Result prediction hyperparameters
        rf_result_params = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', None]
        }
        
        # Optimize goal model
        print("Optimizing goal prediction model...")
        
        # Try Random Forest
        rf_goal_model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
        rf_goal_grid = GridSearchCV(
            rf_goal_model, rf_goal_params, cv=3, 
            scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1
        )
        rf_goal_grid.fit(X_train, y_goals_train)
        rf_best_score = rf_goal_grid.best_score_
        
        # Try XGBoost
        xgb_goal_model = MultiOutputRegressor(xgb.XGBRegressor(random_state=42, eval_metric='mae'))
        xgb_goal_grid = GridSearchCV(
            xgb_goal_model, xgb_goal_params, cv=3, 
            scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1
        )
        xgb_goal_grid.fit(X_train, y_goals_train)
        xgb_best_score = xgb_goal_grid.best_score_
        
        # Choose best goal model
        if rf_best_score > xgb_best_score:
            self.goal_model = rf_goal_grid.best_estimator_
            self.best_params['goal_model'] = {'type': 'RandomForest', 'params': rf_goal_grid.best_params_}
            print(f"Best goal model: Random Forest with score {rf_best_score:.4f}")
        else:
            self.goal_model = xgb_goal_grid.best_estimator_
            self.best_params['goal_model'] = {'type': 'XGBoost', 'params': xgb_goal_grid.best_params_}
            print(f"Best goal model: XGBoost with score {xgb_best_score:.4f}")
        
        # Optimize result model
        print("Optimizing result prediction model...")
        
        # Try Random Forest
        rf_result_grid = GridSearchCV(
            RandomForestClassifier(random_state=42), rf_result_params, 
            cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        rf_result_grid.fit(X_train, y_result_train)
        rf_result_score = rf_result_grid.best_score_
        
        # Try XGBoost
        xgb_result_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        xgb_result_grid = GridSearchCV(
            xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'), xgb_result_params,
            cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        xgb_result_grid.fit(X_train, y_result_train)
        xgb_result_score = xgb_result_grid.best_score_
        
        # Choose best result model
        if rf_result_score > xgb_result_score:
            self.result_model = rf_result_grid.best_estimator_
            self.best_params['result_model'] = {'type': 'RandomForest', 'params': rf_result_grid.best_params_}
            print(f"Best result model: Random Forest with accuracy {rf_result_score:.4f}")
        else:
            self.result_model = xgb_result_grid.best_estimator_
            self.best_params['result_model'] = {'type': 'XGBoost', 'params': xgb_result_grid.best_params_}
            print(f"Best result model: XGBoost with accuracy {xgb_result_score:.4f}")

    def analyze_feature_importance(self, X_train, feature_names):
        """Analyze feature importance for both models"""
        print("Analyzing feature importance...")
        
        # Goal model feature importance
        if hasattr(self.goal_model, 'feature_importances_'):
            # For multi-output, get average importance
            if hasattr(self.goal_model, 'estimators_'):
                importances = np.mean([est.feature_importances_ for est in self.goal_model.estimators_], axis=0)
            else:
                importances = self.goal_model.feature_importances_
            
            goal_importance = dict(zip(feature_names, importances))
            self.feature_importance['goal_model'] = sorted(goal_importance.items(), key=lambda x: x[1], reverse=True)
            
            print("Top 10 features for goal prediction:")
            for feature, importance in self.feature_importance['goal_model'][:10]:
                print(f"{feature}: {importance:.4f}")
        
        # Result model feature importance
        if hasattr(self.result_model, 'feature_importances_'):
            result_importance = dict(zip(feature_names, self.result_model.feature_importances_))
            self.feature_importance['result_model'] = sorted(result_importance.items(), key=lambda x: x[1], reverse=True)
            
            print("\nTop 10 features for result prediction:")
            for feature, importance in self.feature_importance['result_model'][:10]:
                print(f"{feature}: {importance:.4f}")

    def train_models_with_validation(self):
        """Train models with proper validation and hyperparameter tuning"""
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

        # Handle NaN in features with robust imputer
        if np.isnan(X_past).any():
            print("NaN values found in features. Using robust imputation...")
            imputer = SimpleImputer(strategy='median')  # More robust than mean
            X_past = imputer.fit_transform(X_past)
            
            # Save imputer for future use
            self.imputer = imputer

        # Use robust scaler instead of standard scaler
        scaler = RobustScaler()
        X_past_scaled = scaler.fit_transform(X_past)
        self.scaler = scaler

        # Combine goals for multi-output regression
        y_goals = np.column_stack((y_home_past, y_away_past))

        # Time-aware train/test split (use last 20% of matches as test set)
        split_idx = int(len(X_past_scaled) * 0.8)
        X_train, X_test = X_past_scaled[:split_idx], X_past_scaled[split_idx:]
        y_goals_train, y_goals_test = y_goals[:split_idx], y_goals[split_idx:]
        y_result_train, y_result_test = y_result_past[:split_idx], y_result_past[split_idx:]

        # Optimize hyperparameters
        self.optimize_hyperparameters(X_train, y_goals_train, y_result_train)

        # Get feature names for importance analysis
        feature_names = self.feature_engineer.features.columns.tolist()
        self.analyze_feature_importance(X_train, feature_names)

        # Evaluate on test set
        print("\nEvaluating on test set...")
        goals_pred = self.goal_model.predict(X_test)
        mae_home = mean_absolute_error(y_goals_test[:, 0], goals_pred[:, 0])
        mae_away = mean_absolute_error(y_goals_test[:, 1], goals_pred[:, 1])
        print(f"Goal prediction MAE - Home: {mae_home:.3f}, Away: {mae_away:.3f}")

        result_pred = self.result_model.predict(X_test)
        accuracy = accuracy_score(y_result_test, result_pred)
        print(f"Result prediction accuracy: {accuracy:.3f}")

        # Cross-validation scores
        print("\nCross-validation scores...")
        goal_cv_scores = cross_val_score(self.goal_model, X_train, y_goals_train, 
                                       cv=5, scoring='neg_mean_absolute_error')
        result_cv_scores = cross_val_score(self.result_model, X_train, y_result_train, 
                                         cv=5, scoring='accuracy')
        
        print(f"Goal CV MAE: {-goal_cv_scores.mean():.3f} (+/- {goal_cv_scores.std() * 2:.3f})")
        print(f"Result CV Accuracy: {result_cv_scores.mean():.3f} (+/- {result_cv_scores.std() * 2:.3f})")

        return True

    def predict_future_matches(self):
        """Predict outcomes for matches with missing goals/results"""
        if self.future_X is None or len(self.future_X) == 0:
            print("No future matches to predict.")
            return None

        # Apply same preprocessing to future matches
        if hasattr(self, 'imputer'):
            future_X_imputed = self.imputer.transform(self.future_X)
        else:
            future_X_imputed = self.future_X
            
        future_X_scaled = self.scaler.transform(future_X_imputed)

        goals_pred = self.goal_model.predict(future_X_scaled)
        home_goals_pred = np.maximum(0, goals_pred[:, 0].round()).astype(int)  # Ensure non-negative
        away_goals_pred = np.maximum(0, goals_pred[:, 1].round()).astype(int)

        result_pred = self.result_model.predict(future_X_scaled)
        result_proba = self.result_model.predict_proba(future_X_scaled)

        predictions = self.future_matches.copy()
        predictions["pred_home_goals"] = home_goals_pred
        predictions["pred_away_goals"] = away_goals_pred
        predictions["pred_result"] = result_pred
        
        # Add prediction probabilities
        result_classes = self.result_model.classes_
        for i, class_label in enumerate(result_classes):
            predictions[f"prob_{class_label}"] = result_proba[:, i]

        return predictions

    def save_models_and_metadata(self, base_path='models'):
        """Save models, scalers, and training metadata"""
        os.makedirs(base_path, exist_ok=True)

        # Save models
        with open(f'{base_path}/goal_model.pkl', 'wb') as f:
            pickle.dump(self.goal_model, f)

        with open(f'{base_path}/result_model.pkl', 'wb') as f:
            pickle.dump(self.result_model, f)

        with open(f'{base_path}/feature_engineer.pkl', 'wb') as f:
            pickle.dump(self.feature_engineer, f)

        # Save preprocessing objects
        if hasattr(self, 'imputer'):
            with open(f'{base_path}/imputer.pkl', 'wb') as f:
                pickle.dump(self.imputer, f)

        if hasattr(self, 'scaler'):
            with open(f'{base_path}/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)

        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'best_parameters': self.best_params,
            'feature_importance': self.feature_importance
        }

        with open(f'{base_path}/training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Models and metadata saved to {base_path}/")


# Example usage
if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    trainer = AdvancedModelTrainer('data/LaLiga.csv')

    if trainer.load_and_preprocess_data():
        if trainer.train_models_with_validation():
            trainer.save_models_and_metadata()

            preds = trainer.predict_future_matches()
            if preds is not None:
                preds.to_csv("data/future_predictions_optimized.csv", index=False)
                print("Optimized future match predictions saved to data/future_predictions_optimized.csv")
        else:
            print("Model training failed due to data issues")
    else:
        print("Data loading/preprocessing failed")
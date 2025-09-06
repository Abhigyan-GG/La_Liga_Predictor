import pandas as pd
import numpy as np
import pickle
import sys
import os
from sklearn.metrics import (mean_absolute_error, accuracy_score, confusion_matrix, 
                            classification_report, mean_squared_error, r2_score,
                            precision_recall_fscore_support, roc_auc_score, log_loss)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelEvaluator:
    def __init__(self, data_path, models_path='models/'):
        """Initialize the evaluator with trained models and data"""
        self.data_path = data_path
        self.models_path = models_path
        
        # Load models and preprocessors
        self._load_models()
        
        # Initialize results storage
        self.evaluation_results = {}
        self.predictions = None
        
    def _load_models(self):
        """Load all trained models and preprocessors"""
        try:
            with open(f'{self.models_path}/goal_model.pkl', 'rb') as f:
                self.goal_model = pickle.load(f)
            
            with open(f'{self.models_path}/result_model.pkl', 'rb') as f:
                self.result_model = pickle.load(f)
            
            with open(f'{self.models_path}/feature_engineer.pkl', 'rb') as f:
                self.feature_engineer = pickle.load(f)
                
            # Load additional preprocessors if they exist
            if os.path.exists(f'{self.models_path}/scaler.pkl'):
                with open(f'{self.models_path}/scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                self.scaler = None
                
            if os.path.exists(f'{self.models_path}/imputer.pkl'):
                with open(f'{self.models_path}/imputer.pkl', 'rb') as f:
                    self.imputer = pickle.load(f)
            else:
                self.imputer = None
                
            print("All models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def load_and_prepare_data(self):
        """Load and prepare data for evaluation"""
        try:
            # Load data using the improved processor
            from data_processor import DataProcessor
            
            processor = DataProcessor(self.data_path)
            if not processor.load_data() or not processor.preprocess_data():
                raise Exception("Failed to load/preprocess data")
            
            data = processor.get_processed_data()
            
            # Engineer features using the same feature engineer from training
            if hasattr(self.feature_engineer, 'create_advanced_features'):
                X, y_home, y_away, y_result = self.feature_engineer.create_advanced_features()
            else:
                X, y_home, y_away, y_result = self.feature_engineer.create_team_features()
            
            y_goals = np.column_stack((y_home, y_away))
            
            # Apply same preprocessing as training
            if self.imputer is not None:
                X = self.imputer.transform(X)
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            # Filter out matches without results for evaluation
            valid_matches = ~(np.isnan(y_home) | np.isnan(y_away) | np.isnan(y_result))
            X_valid = X[valid_matches]
            y_goals_valid = y_goals[valid_matches]
            y_result_valid = y_result[valid_matches]
            data_valid = data.loc[valid_matches].reset_index(drop=True)
            
            return X_valid, y_goals_valid, y_result_valid, data_valid
            
        except Exception as e:
            print(f"Error preparing data: {e}")
            raise
    
    def evaluate_goal_prediction(self, X, y_goals, data):
        """Comprehensive evaluation of goal prediction model"""
        print("\n" + "="*50)
        print("GOAL PREDICTION EVALUATION")
        print("="*50)
        
        # Make predictions
        goals_pred = self.goal_model.predict(X)
        home_goals_pred = goals_pred[:, 0]
        away_goals_pred = goals_pred[:, 1]
        
        # Basic metrics
        mae_home = mean_absolute_error(y_goals[:, 0], home_goals_pred)
        mae_away = mean_absolute_error(y_goals[:, 1], away_goals_pred)
        rmse_home = np.sqrt(mean_squared_error(y_goals[:, 0], home_goals_pred))
        rmse_away = np.sqrt(mean_squared_error(y_goals[:, 1], away_goals_pred))
        r2_home = r2_score(y_goals[:, 0], home_goals_pred)
        r2_away = r2_score(y_goals[:, 1], away_goals_pred)
        
        goal_results = {
            'mae_home': mae_home,
            'mae_away': mae_away,
            'mae_overall': (mae_home + mae_away) / 2,
            'rmse_home': rmse_home,
            'rmse_away': rmse_away,
            'r2_home': r2_home,
            'r2_away': r2_away
        }
        
        print(f"Home Goals - MAE: {mae_home:.3f}, RMSE: {rmse_home:.3f}, R²: {r2_home:.3f}")
        print(f"Away Goals - MAE: {mae_away:.3f}, RMSE: {rmse_away:.3f}, R²: {r2_away:.3f}")
        print(f"Overall MAE: {goal_results['mae_overall']:.3f}")
        
        # Accuracy within different tolerances
        tolerances = [0.5, 1.0, 1.5]
        for tol in tolerances:
            home_acc = np.mean(np.abs(y_goals[:, 0] - home_goals_pred) <= tol)
            away_acc = np.mean(np.abs(y_goals[:, 1] - away_goals_pred) <= tol)
            overall_acc = (home_acc + away_acc) / 2
            goal_results[f'accuracy_within_{tol}'] = overall_acc
            print(f"Accuracy within ±{tol} goals: {overall_acc:.3f}")
        
        # Analyze prediction errors
        self._analyze_goal_errors(y_goals, goals_pred, data)
        
        # Cross-validation
        cv_scores = cross_val_score(self.goal_model, X, y_goals, cv=5, 
                                  scoring='neg_mean_absolute_error')
        goal_results['cv_mae'] = -cv_scores.mean()
        goal_results['cv_mae_std'] = cv_scores.std()
        print(f"5-Fold CV MAE: {-cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        return goal_results, goals_pred
    
    def evaluate_result_prediction(self, X, y_result, data):
        """Comprehensive evaluation of result prediction model"""
        print("\n" + "="*50)
        print("RESULT PREDICTION EVALUATION")
        print("="*50)
        
        # Make predictions
        result_pred = self.result_model.predict(X)
        result_proba = self.result_model.predict_proba(X)
        
        # Basic accuracy
        accuracy = accuracy_score(y_result, result_pred)
        
        # Detailed metrics for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_result, result_pred, average=None
        )
        
        # Map results to labels
        result_map = {1: "Home Win", 0: "Draw", -1: "Away Win"}
        classes = self.result_model.classes_
        
        result_results = {
            'accuracy': accuracy,
            'macro_precision': np.mean(precision),
            'macro_recall': np.mean(recall),
            'macro_f1': np.mean(f1)
        }
        
        print(f"Overall Accuracy: {accuracy:.3f}")
        print(f"Macro F1-Score: {result_results['macro_f1']:.3f}")
        
        # Per-class metrics
        print("\nPer-class Performance:")
        for i, class_label in enumerate(classes):
            class_name = result_map[class_label]
            result_results[f'{class_name.lower()}_precision'] = precision[i]
            result_results[f'{class_name.lower()}_recall'] = recall[i]
            result_results[f'{class_name.lower()}_f1'] = f1[i]
            print(f"{class_name:10}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}")
        
        # Log loss for probability calibration
        try:
            logloss = log_loss(y_result, result_proba)
            result_results['log_loss'] = logloss
            print(f"Log Loss: {logloss:.3f}")
        except:
            pass
        
        # Confusion Matrix
        cm = confusion_matrix(y_result, result_pred)
        self._plot_confusion_matrix(cm, [result_map[c] for c in classes])
        
        # Cross-validation
        cv_scores = cross_val_score(self.result_model, X, y_result, cv=5, scoring='accuracy')
        result_results['cv_accuracy'] = cv_scores.mean()
        result_results['cv_accuracy_std'] = cv_scores.std()
        print(f"5-Fold CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Analyze prediction confidence
        self._analyze_prediction_confidence(y_result, result_pred, result_proba, classes)
        
        return result_results, result_pred, result_proba
    
    def _analyze_goal_errors(self, y_goals, goals_pred, data):
        """Analyze patterns in goal prediction errors"""
        home_errors = np.abs(y_goals[:, 0] - goals_pred[:, 0])
        away_errors = np.abs(y_goals[:, 1] - goals_pred[:, 1])
        
        # Create error analysis plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Home goals: actual vs predicted
        axes[0,0].scatter(y_goals[:, 0], goals_pred[:, 0], alpha=0.6)
        axes[0,0].plot([0, max(y_goals[:, 0])], [0, max(y_goals[:, 0])], 'r--')
        axes[0,0].set_xlabel('Actual Home Goals')
        axes[0,0].set_ylabel('Predicted Home Goals')
        axes[0,0].set_title('Home Goals: Actual vs Predicted')
        
        # Away goals: actual vs predicted
        axes[0,1].scatter(y_goals[:, 1], goals_pred[:, 1], alpha=0.6)
        axes[0,1].plot([0, max(y_goals[:, 1])], [0, max(y_goals[:, 1])], 'r--')
        axes[0,1].set_xlabel('Actual Away Goals')
        axes[0,1].set_ylabel('Predicted Away Goals')
        axes[0,1].set_title('Away Goals: Actual vs Predicted')
        
        # Error distribution
        axes[1,0].hist(home_errors, bins=20, alpha=0.7, label='Home')
        axes[1,0].hist(away_errors, bins=20, alpha=0.7, label='Away')
        axes[1,0].set_xlabel('Absolute Error')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Error Distribution')
        axes[1,0].legend()
        
        # Errors by total goals
        total_goals = y_goals[:, 0] + y_goals[:, 1]
        total_errors = home_errors + away_errors
        axes[1,1].scatter(total_goals, total_errors, alpha=0.6)
        axes[1,1].set_xlabel('Total Actual Goals')
        axes[1,1].set_ylabel('Total Absolute Error')
        axes[1,1].set_title('Error vs Total Goals')
        
        plt.tight_layout()
        os.makedirs('evaluation', exist_ok=True)
        plt.savefig('evaluation/goal_prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Error statistics by score ranges
        print("\nError Analysis by Score Ranges:")
        score_ranges = [(0, 1), (2, 3), (4, 5), (6, 20)]
        for low, high in score_ranges:
            mask = (total_goals >= low) & (total_goals <= high)
            if mask.sum() > 0:
                avg_error = total_errors[mask].mean()
                print(f"  {low}-{high} goals: {avg_error:.3f} average error ({mask.sum()} matches)")
    
    def _plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Result Prediction')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Add percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j+0.5, i+0.7, f'({cm_norm[i,j]:.1%})', 
                        ha='center', va='center', color='red', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('evaluation/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _analyze_prediction_confidence(self, y_true, y_pred, y_proba, classes):
        """Analyze prediction confidence and calibration"""
        # Confidence distribution
        max_proba = np.max(y_proba, axis=1)
        correct_predictions = (y_true == y_pred)
        
        print("\nPrediction Confidence Analysis:")
        confidence_bins = [0.4, 0.6, 0.8, 0.9, 1.0]
        for i in range(len(confidence_bins)-1):
            low, high = confidence_bins[i], confidence_bins[i+1]
            mask = (max_proba >= low) & (max_proba < high)
            if mask.sum() > 0:
                accuracy = correct_predictions[mask].mean()
                count = mask.sum()
                print(f"  Confidence {low:.1f}-{high:.1f}: {accuracy:.3f} accuracy ({count} predictions)")
        
        # Plot confidence vs accuracy
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(max_proba[correct_predictions], bins=20, alpha=0.7, label='Correct', density=True)
        plt.hist(max_proba[~correct_predictions], bins=20, alpha=0.7, label='Incorrect', density=True)
        plt.xlabel('Maximum Predicted Probability')
        plt.ylabel('Density')
        plt.title('Confidence Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        # Calibration plot
        bins = np.linspace(0.3, 1.0, 8)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        accuracies = []
        for i in range(len(bins)-1):
            mask = (max_proba >= bins[i]) & (max_proba < bins[i+1])
            if mask.sum() > 0:
                accuracies.append(correct_predictions[mask].mean())
            else:
                accuracies.append(0)
        
        plt.plot(bin_centers, accuracies, 'bo-', label='Actual')
        plt.plot([0.3, 1.0], [0.3, 1.0], 'r--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('evaluation/confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_performance_by_factors(self, X, y_goals, y_result, data, goals_pred, result_pred):
        """Analyze model performance by various factors"""
        print("\n" + "="*50)
        print("PERFORMANCE BY FACTORS ANALYSIS")
        print("="*50)
        
        factor_analysis = {}
        
        # Performance by season
        if 'season_year' in data.columns:
            seasons = sorted(data['season_year'].unique())
            season_goal_mae = []
            season_result_acc = []
            
            for season in seasons:
                season_mask = data['season_year'] == season
                if season_mask.sum() > 0:
                    # Goal prediction performance
                    season_goal_mae.append(
                        (mean_absolute_error(y_goals[season_mask, 0], goals_pred[season_mask, 0]) +
                         mean_absolute_error(y_goals[season_mask, 1], goals_pred[season_mask, 1])) / 2
                    )
                    
                    # Result prediction performance
                    season_result_acc.append(
                        accuracy_score(y_result[season_mask], result_pred[season_mask])
                    )
            
            factor_analysis['season_performance'] = {
                'seasons': seasons,
                'goal_mae': season_goal_mae,
                'result_accuracy': season_result_acc
            }
            
            # Plot seasonal performance
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].plot(seasons, season_goal_mae, 'bo-')
            axes[0].set_xlabel('Season')
            axes[0].set_ylabel('Goal Prediction MAE')
            axes[0].set_title('Goal Prediction Performance by Season')
            axes[0].grid(True)
            
            axes[1].plot(seasons, season_result_acc, 'ro-')
            axes[1].set_xlabel('Season')
            axes[1].set_ylabel('Result Prediction Accuracy')
            axes[1].set_title('Result Prediction Performance by Season')
            axes[1].grid(True)
            
            plt.tight_layout()
            plt.savefig('evaluation/performance_by_season.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Best season for goals: {seasons[np.argmin(season_goal_mae)]} (MAE: {min(season_goal_mae):.3f})")
            print(f"Best season for results: {seasons[np.argmax(season_result_acc)]} (Acc: {max(season_result_acc):.3f})")
        
        # Performance by team strength (based on league position)
        if 'home_league_position' in data.columns and 'away_league_position' in data.columns:
            self._analyze_performance_by_team_strength(data, y_goals, y_result, goals_pred, result_pred)
        
        # Performance by match characteristics
        self._analyze_performance_by_match_type(data, y_goals, y_result, goals_pred, result_pred)
        
        return factor_analysis
    
    def _analyze_performance_by_team_strength(self, data, y_goals, y_result, goals_pred, result_pred):
        """Analyze performance based on team strength differences"""
        print("\nPerformance by Team Strength:")
        
        # Calculate team strength difference
        pos_diff = np.abs(data['home_league_position'] - data['away_league_position'])
        
        # Categorize matches
        categories = [
            ('Close (0-3 positions)', (pos_diff <= 3)),
            ('Medium (4-8 positions)', (pos_diff > 3) & (pos_diff <= 8)),
            ('Large (>8 positions)', (pos_diff > 8))
        ]
        
        for cat_name, mask in categories:
            if mask.sum() > 0:
                goal_mae = (mean_absolute_error(y_goals[mask, 0], goals_pred[mask, 0]) +
                           mean_absolute_error(y_goals[mask, 1], goals_pred[mask, 1])) / 2
                result_acc = accuracy_score(y_result[mask], result_pred[mask])
                print(f"  {cat_name}: Goal MAE={goal_mae:.3f}, Result Acc={result_acc:.3f} ({mask.sum()} matches)")
    
    def _analyze_performance_by_match_type(self, data, y_goals, y_result, goals_pred, result_pred):
        """Analyze performance by different match characteristics"""
        print("\nPerformance by Match Type:")
        
        # Weekend vs weekday
        if 'is_weekend' in data.columns:
            for match_type, mask in [('Weekend', data['is_weekend'] == 1), 
                                   ('Weekday', data['is_weekend'] == 0)]:
                if mask.sum() > 0:
                    goal_mae = (mean_absolute_error(y_goals[mask, 0], goals_pred[mask, 0]) +
                               mean_absolute_error(y_goals[mask, 1], goals_pred[mask, 1])) / 2
                    result_acc = accuracy_score(y_result[mask], result_pred[mask])
                    print(f"  {match_type}: Goal MAE={goal_mae:.3f}, Result Acc={result_acc:.3f} ({mask.sum()} matches)")
        
        # High vs low scoring games
        if 'total_goals' in data.columns:
            median_goals = data['total_goals'].median()
            for match_type, mask in [('High Scoring', data['total_goals'] > median_goals), 
                                   ('Low Scoring', data['total_goals'] <= median_goals)]:
                if mask.sum() > 0:
                    result_acc = accuracy_score(y_result[mask], result_pred[mask])
                    print(f"  {match_type}: Result Acc={result_acc:.3f} ({mask.sum()} matches)")
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive evaluation report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION REPORT")
        print("="*60)
        
        try:
            # Load and prepare data
            X, y_goals, y_result, data = self.load_and_prepare_data()
            
            # Evaluate models
            goal_results, goals_pred = self.evaluate_goal_prediction(X, y_goals, data)
            result_results, result_pred, result_proba = self.evaluate_result_prediction(X, y_result, data)
            
            # Additional analysis
            factor_analysis = self.analyze_performance_by_factors(
                X, y_goals, y_result, data, goals_pred, result_pred
            )
            
            # Combine results
            self.evaluation_results = {
                'evaluation_date': datetime.now().isoformat(),
                'dataset_size': len(X),
                'goal_prediction': goal_results,
                'result_prediction': result_results,
                'factor_analysis': factor_analysis
            }
            
            # Save detailed results
            os.makedirs('evaluation', exist_ok=True)
            with open('evaluation/comprehensive_results.json', 'w') as f:
                json.dump(self.evaluation_results, f, indent=2, default=str)
            
            # Save predictions for further analysis
            predictions_df = data.copy()
            predictions_df['pred_home_goals'] = goals_pred[:, 0]
            predictions_df['pred_away_goals'] = goals_pred[:, 1]
            predictions_df['pred_result'] = result_pred
            
            # Add probabilities
            classes = self.result_model.classes_
            for i, class_label in enumerate(classes):
                predictions_df[f'prob_{class_label}'] = result_proba[:, i]
            
            predictions_df.to_csv('evaluation/detailed_predictions.csv', index=False)
            
            print(f"\n✅ Evaluation completed successfully!")
            print(f"📊 Results saved to evaluation/ directory")
            print(f"📈 {len(X)} matches evaluated")
            print(f"🎯 Overall Goal MAE: {goal_results['mae_overall']:.3f}")
            print(f"🏆 Overall Result Accuracy: {result_results['accuracy']:.3f}")
            
            return self.evaluation_results
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            raise

# Example usage
if __name__ == "__main__":
    evaluator = AdvancedModelEvaluator(
        data_path='data/LaLiga.csv',
        models_path='models/'
    )
    
    results = evaluator.generate_comprehensive_report()
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Goal Prediction MAE: {results['goal_prediction']['mae_overall']:.3f}")
    print(f"Result Prediction Accuracy: {results['result_prediction']['accuracy']:.3f}")
    print("="*60)
"""XGBoost model trainer."""

import xgboost as xgb
import numpy as np
from typing import Dict, Any
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..config import config
from ..exceptions import ModelTrainingError
from ..utils.logger import LoggerMixin


class XGBoostTrainer(LoggerMixin):
    """XGBoost trainer with hyperparameter tuning."""
    
    def __init__(self, config=None):
        self.config = config or config
        self.model = None
        self.best_params = None
        self.metrics = {}
    
    def create_model(self) -> xgb.XGBRegressor:
        """Create XGBoost model."""
        self.model = xgb.XGBRegressor(
            n_estimators=self.config.model.n_estimators,
            max_depth=self.config.model.max_depth,
            learning_rate=self.config.model.learning_rate,
            subsample=self.config.model.subsample,
            colsample_bytree=self.config.model.colsample_bytree,
            min_child_weight=self.config.model.min_child_weight,
            random_state=self.config.data.random_state,
            objective='reg:squarederror',
            early_stopping_rounds=self.config.model.early_stopping_rounds
        )
        self.log_info("XGBoost model created")
        return self.model
    
    def tune_hyperparameters(self, X_train, y_train, cv=5):
        """Hyperparameter tuning with GridSearchCV."""
        self.log_info(f"Starting hyperparameter tuning with {cv}-fold CV...")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
        }
        
        try:
            base_model = xgb.XGBRegressor(
                random_state=self.config.data.random_state,
                objective='reg:squarederror'
            )
            
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            self.best_params = grid_search.best_params_
            self.model = grid_search.best_estimator_
            
            self.log_info(f"Best params: {self.best_params}")
            return self.model, self.best_params
            
        except Exception as e:
            raise ModelTrainingError(f"Hyperparameter tuning failed: {e}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model."""
        if self.model is None:
            self.create_model()
        
        self.log_info("Training model...")
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        try:
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            self.log_info("Model training complete")
        except Exception as e:
            raise ModelTrainingError(f"Training failed: {e}")
    
    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        """Evaluate model performance."""
        if self.model is None:
            raise ModelTrainingError("No trained model")
        
        self.log_info("Evaluating model...")
        
        y_pred = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        accuracy = np.mean(np.abs((y_test - y_pred) / y_test) < 0.05) * 100
        
        self.metrics = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R² Score': float(r2),
            'MAPE': float(mape),
            'Accuracy (5%)': float(accuracy)
        }
        
        self.log_info(f"Evaluation complete: {self.metrics}")
        return self.metrics
    
    def predict(self, X) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ModelTrainingError("No trained model")
        return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """Save model to file."""
        import joblib
        joblib.dump(self.model, filepath)
        self.log_info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file."""
        import joblib
        try:
            self.model = joblib.load(filepath)
            self.log_info(f"Model loaded from {filepath}")
            return self.model
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")
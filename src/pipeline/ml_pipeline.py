"""End-to-end ML pipeline."""

import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path
import joblib

from ..config import config, Config
from ..utils.logger import LoggerMixin
from ..data.fetcher import CryptoDataFetcher
from ..data.processor import FeatureEngineer
from ..models.trainer import XGBoostTrainer


class MLPipeline(LoggerMixin):
    """End-to-end ML pipeline."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or config
        self.fetcher = CryptoDataFetcher(
            symbol=self.config.data.symbol,
            period=self.config.data.period
        )
        self.engineer = FeatureEngineer(self.config)
        self.trainer = XGBoostTrainer(self.config)
        
        self.model_dir = Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_info(f"Pipeline initialized for {self.config.data.symbol}")
    
    def run_training(self, tune: bool = True, save_model: bool = True) -> Dict[str, Any]:
        """Run complete training pipeline."""
        self.log_info("Starting training pipeline...")
        
        # 1. Fetch data
        raw_data = self.fetcher.fetch_data()
        
        # 2. Feature engineering
        processed_data = self.engineer.process_data(raw_data)
        
        # 3. Prepare features
        X, y, features = self.engineer.prepare_features(
            processed_data,
            lookback_days=self.config.data.lookback_days
        )
        
        # 4. Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.data.test_size,
            shuffle=False, random_state=self.config.data.random_state
        )
        
        # 5. Scale features
        X_train_scaled, X_test_scaled = self.engineer.scale_features(X_train, X_test)
        
        # 6. Train model
        if tune:
            self.trainer.tune_hyperparameters(X_train_scaled, y_train)
        else:
            self.trainer.create_model()
            self.trainer.train(X_train_scaled, y_train)
        
        # 7. Evaluate
        metrics = self.trainer.evaluate(X_test_scaled, y_test)
        
        # 8. Save model
        if save_model:
            model_path = self.model_dir / f"{self.config.data.symbol}_model.pkl"
            self.trainer.save_model(str(model_path))
            
            # Save scaler
            scaler_path = self.model_dir / f"{self.config.data.symbol}_scaler.pkl"
            joblib.dump(self.engineer.scaler, scaler_path)
        
        self.log_info("Training pipeline complete")
        return {'metrics': metrics, 'features': features}
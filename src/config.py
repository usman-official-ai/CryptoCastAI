"""
Configuration management module for CryptoCastAI.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import os
from pydantic import BaseModel, Field, validator

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


class DataConfig(BaseModel):
    """Data configuration."""
    symbol: str = Field(default="BTC-USD")
    period: str = Field(default="2y")
    interval: str = Field(default="1d")
    lookback_days: int = Field(default=3)
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    random_state: int = Field(default=42)


class ModelConfig(BaseModel):
    """Model configuration."""
    n_estimators: int = Field(default=300, ge=100, le=1000)
    max_depth: int = Field(default=6, ge=3, le=15)
    learning_rate: float = Field(default=0.05, ge=0.001, le=0.5)
    subsample: float = Field(default=0.8, ge=0.5, le=1.0)
    colsample_bytree: float = Field(default=0.8, ge=0.5, le=1.0)
    min_child_weight: int = Field(default=1, ge=1, le=10)
    early_stopping_rounds: int = Field(default=50)
    cv_folds: int = Field(default=5)


class Config(BaseModel):
    """Main configuration."""
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    debug: bool = False
    log_level: str = "INFO"


config = Config()
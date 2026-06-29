"""Feature engineering module."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from sklearn.preprocessing import StandardScaler

from ..config import config
from ..utils.logger import LoggerMixin


class FeatureEngineer(LoggerMixin):
    """Feature engineering for cryptocurrency data."""
    
    def __init__(self, config=None):
        self.config = config or config
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        data = df.copy()
        
        # Moving Averages
        data['SMA_10'] = data['Close'].rolling(10).mean()
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Diff'] = data['MACD'] - data['MACD_Signal']
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(20).mean()
        bb_std = data['Close'].rolling(20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        data['BB_Percent'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
        
        # Volatility
        data['Volatility_10'] = data['Close'].rolling(10).std()
        data['Volatility_20'] = data['Close'].rolling(20).std()
        data['Volatility_30'] = data['Close'].rolling(30).std()
        
        # Price ratios
        data['Price_SMA_10'] = data['Close'] / data['SMA_10']
        data['Price_SMA_20'] = data['Close'] / data['SMA_20']
        data['Price_SMA_50'] = data['Close'] / data['SMA_50']
        
        # Volume
        data['Volume_SMA_10'] = data['Volume'].rolling(10).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_10']
        
        # Momentum
        data['Momentum'] = data['Close'] - data['Close'].shift(10)
        data['Price_Change_1d'] = data['Close'].pct_change()
        data['Price_Change_3d'] = data['Close'].pct_change(periods=3)
        data['Price_Change_7d'] = data['Close'].pct_change(periods=7)
        
        # Lag features
        for lag in [1, 3, 7]:
            data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
            data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            data[f'Close_Min_{window}'] = data['Close'].rolling(window).min()
            data[f'Close_Max_{window}'] = data['Close'].rolling(window).max()
            data[f'Close_Mean_{window}'] = data['Close'].rolling(window).mean()
        
        # Additional
        data['High_Low_Ratio'] = data['High'] / data['Low']
        data['Close_Open_Ratio'] = data['Close'] / data['Open']
        data['Daily_Range'] = data['High'] - data['Low']
        
        data = data.dropna()
        return data
    
    def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Process raw data."""
        self.log_info("Processing data...")
        processed = self.calculate_technical_indicators(raw_data)
        self.log_info(f"Processed {len(processed)} rows")
        return processed
    
    def prepare_features(
        self,
        data: pd.DataFrame,
        target_col: str = 'Close',
        lookback_days: int = 3
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare features for training."""
        self.log_info("Preparing features...")
        
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        feature_cols.extend(['Close', 'Volume'])
        feature_cols = list(set(feature_cols))
        
        X = data[feature_cols].copy()
        y = data[target_col].copy()
        
        if lookback_days > 0:
            X = X.shift(lookback_days)
            y = y.shift(-lookback_days)
            X = X.dropna()
            y = y.loc[X.index]
        
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        self.feature_names = X.columns.tolist()
        self.log_info(f"Prepared {X.shape[1]} features, {X.shape[0]} samples")
        return X, y, self.feature_names
    
    def scale_features(self, X_train, X_test):
        """Scale features."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
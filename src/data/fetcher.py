"""Data fetcher module."""

import yfinance as yf
import pandas as pd
from typing import Optional, Dict, Any
import time
import hashlib
import json
from pathlib import Path

from ..config import config
from ..exceptions import DataFetchError
from ..utils.logger import LoggerMixin


class CryptoDataFetcher(LoggerMixin):
    """Fetches cryptocurrency data from Yahoo Finance."""
    
    def __init__(
        self,
        symbol: Optional[str] = None,
        period: Optional[str] = None,
        interval: str = "1d",
        max_retries: int = 3,
        cache_enabled: bool = True
    ):
        self.symbol = symbol or config.data.symbol
        self.period = period or config.data.period
        self.interval = interval
        self.max_retries = max_retries
        self.cache_enabled = cache_enabled
        
        self.cache_dir = Path("cache") / "data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_info(f"DataFetcher initialized for {self.symbol}")
    
    def _get_cache_key(self) -> str:
        params = {
            'symbol': self.symbol,
            'period': self.period,
            'interval': self.interval
        }
        key_string = json.dumps(params, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def fetch_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch cryptocurrency data."""
        
        # Try cache
        cache_key = self._get_cache_key()
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        
        if not force_refresh and cache_file.exists():
            try:
                data = pd.read_parquet(cache_file)
                self.log_info(f"Loaded {len(data)} records from cache")
                return data
            except Exception as e:
                self.log_warning(f"Cache load failed: {e}")
        
        # Fetch with retries
        for attempt in range(self.max_retries):
            try:
                self.log_info(f"Fetching data (attempt {attempt + 1}/{self.max_retries})")
                
                ticker = yf.Ticker(self.symbol)
                data = ticker.history(period=self.period, interval=self.interval)
                
                if data.empty:
                    raise DataFetchError(self.symbol, "No data returned")
                
                if data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                
                data['Symbol'] = self.symbol
                
                if self.cache_enabled:
                    data.to_parquet(cache_file)
                
                self.log_info(f"Fetched {len(data)} records for {self.symbol}")
                return data
                
            except Exception as e:
                wait_time = 1.0 * (2 ** attempt)
                self.log_warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise DataFetchError(self.symbol, f"All attempts failed: {e}")
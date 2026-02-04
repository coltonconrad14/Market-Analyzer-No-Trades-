"""
Data fetcher module for retrieving market data from various sources.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List


class DataFetcher:
    """Fetches market data for stocks and cryptocurrencies."""
    
    def __init__(self):
        """Initialize the data fetcher."""
        self.data_cache = {}
    
    def fetch_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            period: Data period (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
            interval: Data interval (e.g., '1m', '5m', '15m', '1h', '1d', '1wk', '1mo')
        
        Returns:
            DataFrame with OHLCV data or None if fetch fails
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"No data found for symbol: {symbol}")
                return None
            
            # Cache the data
            cache_key = f"{symbol}_{period}_{interval}"
            self.data_cache[cache_key] = data
            
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def fetch_crypto_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Fetch cryptocurrency data from Yahoo Finance.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC-USD', 'ETH-USD')
            period: Data period
            interval: Data interval
        
        Returns:
            DataFrame with OHLCV data or None if fetch fails
        """
        # Ensure crypto symbols have -USD suffix
        if not symbol.endswith('-USD'):
            symbol = f"{symbol}-USD"
        
        return self.fetch_stock_data(symbol, period, interval)
    
    def fetch_multiple_assets(self, symbols: List[str], period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple assets.
        
        Args:
            symbols: List of asset symbols
            period: Data period
            interval: Data interval
        
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        results = {}
        for symbol in symbols:
            data = self.fetch_stock_data(symbol, period, interval)
            if data is not None:
                results[symbol] = data
        return results
    
    def get_asset_info(self, symbol: str) -> Optional[Dict]:
        """
        Get detailed information about an asset.
        
        Args:
            symbol: Asset symbol
        
        Returns:
            Dictionary with asset information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
        except Exception as e:
            print(f"Error fetching info for {symbol}: {str(e)}")
            return None

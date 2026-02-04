"""
Technical indicators module for calculating various market indicators.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class TechnicalIndicators:
    """Calculate technical analysis indicators."""
    
    @staticmethod
    def calculate_sma(data: pd.DataFrame, period: int = 20, column: str = 'Close') -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            data: DataFrame with price data
            period: Number of periods for SMA
            column: Column to use for calculation
        
        Returns:
            Series with SMA values
        """
        return data[column].rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data: pd.DataFrame, period: int = 20, column: str = 'Close') -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            data: DataFrame with price data
            period: Number of periods for EMA
            column: Column to use for calculation
        
        Returns:
            Series with EMA values
        """
        return data[column].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14, column: str = 'Close') -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            data: DataFrame with price data
            period: Number of periods for RSI
            column: Column to use for calculation
        
        Returns:
            Series with RSI values (0-100)
        """
        delta = data[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, 
                       signal: int = 9, column: str = 'Close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: DataFrame with price data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            column: Column to use for calculation
        
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = data[column].ewm(span=fast, adjust=False).mean()
        ema_slow = data[column].ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, 
                                   std_dev: float = 2.0, column: str = 'Close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: DataFrame with price data
            period: Number of periods for moving average
            std_dev: Number of standard deviations for bands
            column: Column to use for calculation
        
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle_band = data[column].rolling(window=period).mean()
        std = data[column].rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            data: DataFrame with OHLC data
            period: Number of periods for ATR
        
        Returns:
            Series with ATR values
        """
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_obv(data: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume.
        
        Args:
            data: DataFrame with price and volume data
        
        Returns:
            Series with OBV values
        """
        obv = [0]
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        return pd.Series(obv, index=data.index)
    
    @staticmethod
    def calculate_stochastic(data: pd.DataFrame, k_period: int = 14, 
                             d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            data: DataFrame with OHLC data
            k_period: Period for %K calculation
            d_period: Period for %D calculation (signal line)
        
        Returns:
            Tuple of (%K, %D)
        """
        low_min = data['Low'].rolling(window=k_period).min()
        high_max = data['High'].rolling(window=k_period).max()
        
        k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    @staticmethod
    def add_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the dataframe.
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            DataFrame with all indicators added
        """
        df = data.copy()
        
        # Moving Averages
        df['SMA_20'] = TechnicalIndicators.calculate_sma(df, 20)
        df['SMA_50'] = TechnicalIndicators.calculate_sma(df, 50)
        df['SMA_200'] = TechnicalIndicators.calculate_sma(df, 200)
        df['EMA_12'] = TechnicalIndicators.calculate_ema(df, 12)
        df['EMA_26'] = TechnicalIndicators.calculate_ema(df, 26)
        
        # RSI
        df['RSI'] = TechnicalIndicators.calculate_rsi(df, 14)
        
        # MACD
        macd, signal, histogram = TechnicalIndicators.calculate_macd(df)
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Histogram'] = histogram
        
        # Bollinger Bands
        upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(df)
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower
        
        # ATR
        df['ATR'] = TechnicalIndicators.calculate_atr(df, 14)
        
        # OBV
        df['OBV'] = TechnicalIndicators.calculate_obv(df)
        
        # Stochastic
        k, d = TechnicalIndicators.calculate_stochastic(df)
        df['Stoch_K'] = k
        df['Stoch_D'] = d
        
        return df

"""
Trend analyzer module for identifying market trends.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


class TrendAnalyzer:
    """Analyze market trends and patterns."""
    
    @staticmethod
    def identify_trend(data: pd.DataFrame, sma_short: int = 50, sma_long: int = 200) -> str:
        """
        Identify the overall market trend.
        
        Args:
            data: DataFrame with price data and indicators
            sma_short: Short-term SMA period
            sma_long: Long-term SMA period
        
        Returns:
            Trend direction: 'BULLISH', 'BEARISH', or 'SIDEWAYS'
        """
        if len(data) < sma_long:
            return "INSUFFICIENT_DATA"
        
        # Calculate SMAs if not already present
        if f'SMA_{sma_short}' not in data.columns:
            data[f'SMA_{sma_short}'] = data['Close'].rolling(window=sma_short).mean()
        if f'SMA_{sma_long}' not in data.columns:
            data[f'SMA_{sma_long}'] = data['Close'].rolling(window=sma_long).mean()
        
        latest_data = data.iloc[-1]
        current_price = latest_data['Close']
        sma_short_val = latest_data[f'SMA_{sma_short}']
        sma_long_val = latest_data[f'SMA_{sma_long}']
        
        # Determine trend
        if pd.isna(sma_short_val) or pd.isna(sma_long_val):
            return "INSUFFICIENT_DATA"
        
        if sma_short_val > sma_long_val and current_price > sma_short_val:
            return "BULLISH"
        elif sma_short_val < sma_long_val and current_price < sma_short_val:
            return "BEARISH"
        else:
            return "SIDEWAYS"
    
    @staticmethod
    def calculate_trend_strength(data: pd.DataFrame) -> float:
        """
        Calculate trend strength (0-100).
        
        Args:
            data: DataFrame with price data
        
        Returns:
            Trend strength percentage
        """
        if len(data) < 20:
            return 0.0
        
        # Calculate price momentum
        price_change = (data['Close'].iloc[-1] - data['Close'].iloc[-20]) / data['Close'].iloc[-20]
        
        # Calculate ADX-like metric using price ranges
        high_low = data['High'] - data['Low']
        avg_range = high_low.tail(14).mean()
        current_range = high_low.iloc[-1]
        
        # Normalize to 0-100 scale
        strength = min(abs(price_change) * 100, 100)
        
        return round(strength, 2)
    
    @staticmethod
    def identify_support_resistance(data: pd.DataFrame, window: int = 20) -> Dict[str, float]:
        """
        Identify support and resistance levels.
        
        Args:
            data: DataFrame with OHLC data
            window: Lookback window for identifying levels
        
        Returns:
            Dictionary with support and resistance levels
        """
        recent_data = data.tail(window)
        
        support = recent_data['Low'].min()
        resistance = recent_data['High'].max()
        
        return {
            'support': round(support, 2),
            'resistance': round(resistance, 2),
            'current': round(data['Close'].iloc[-1], 2)
        }
    
    @staticmethod
    def detect_crossover(data: pd.DataFrame, fast_col: str, slow_col: str) -> Optional[str]:
        """
        Detect crossover signals.
        
        Args:
            data: DataFrame with indicator data
            fast_col: Fast indicator column name
            slow_col: Slow indicator column name
        
        Returns:
            'BULLISH' for golden cross, 'BEARISH' for death cross, None otherwise
        """
        if len(data) < 2:
            return None
        
        if fast_col not in data.columns or slow_col not in data.columns:
            return None
        
        current_fast = data[fast_col].iloc[-1]
        current_slow = data[slow_col].iloc[-1]
        prev_fast = data[fast_col].iloc[-2]
        prev_slow = data[slow_col].iloc[-2]
        
        if pd.isna(current_fast) or pd.isna(current_slow) or pd.isna(prev_fast) or pd.isna(prev_slow):
            return None
        
        # Bullish crossover (golden cross)
        if prev_fast <= prev_slow and current_fast > current_slow:
            return "BULLISH"
        # Bearish crossover (death cross)
        elif prev_fast >= prev_slow and current_fast < current_slow:
            return "BEARISH"
        
        return None
    
    @staticmethod
    def analyze_volume_trend(data: pd.DataFrame, period: int = 20) -> Dict[str, any]:
        """
        Analyze volume trends.
        
        Args:
            data: DataFrame with volume data
            period: Period for volume analysis
        
        Returns:
            Dictionary with volume analysis
        """
        if 'Volume' not in data.columns or len(data) < period:
            return {'status': 'INSUFFICIENT_DATA'}
        
        recent_volume = data['Volume'].tail(period)
        avg_volume = recent_volume.mean()
        current_volume = data['Volume'].iloc[-1]
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        if volume_ratio > 1.5:
            status = "HIGH_VOLUME"
        elif volume_ratio < 0.5:
            status = "LOW_VOLUME"
        else:
            status = "NORMAL_VOLUME"
        
        return {
            'status': status,
            'current_volume': int(current_volume),
            'avg_volume': int(avg_volume),
            'ratio': round(volume_ratio, 2)
        }
    
    @staticmethod
    def detect_divergence(data: pd.DataFrame, price_col: str = 'Close', 
                          indicator_col: str = 'RSI', lookback: int = 14) -> Optional[str]:
        """
        Detect bullish or bearish divergence.
        
        Args:
            data: DataFrame with price and indicator data
            price_col: Price column name
            indicator_col: Indicator column name
            lookback: Lookback period
        
        Returns:
            'BULLISH_DIVERGENCE', 'BEARISH_DIVERGENCE', or None
        """
        if len(data) < lookback or indicator_col not in data.columns:
            return None
        
        recent_data = data.tail(lookback)
        
        price_trend = recent_data[price_col].iloc[-1] - recent_data[price_col].iloc[0]
        indicator_trend = recent_data[indicator_col].iloc[-1] - recent_data[indicator_col].iloc[0]
        
        # Bullish divergence: price making lower lows, indicator making higher lows
        if price_trend < 0 and indicator_trend > 0:
            return "BULLISH_DIVERGENCE"
        # Bearish divergence: price making higher highs, indicator making lower highs
        elif price_trend > 0 and indicator_trend < 0:
            return "BEARISH_DIVERGENCE"
        
        return None

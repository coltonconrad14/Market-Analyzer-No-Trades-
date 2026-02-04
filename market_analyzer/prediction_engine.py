"""
Prediction engine module for generating market predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .technical_indicators import TechnicalIndicators
from .trend_analyzer import TrendAnalyzer


class PredictionEngine:
    """Generate predictions and recommendations based on technical analysis."""
    
    def __init__(self):
        """Initialize the prediction engine."""
        self.indicators = TechnicalIndicators()
        self.trend_analyzer = TrendAnalyzer()
    
    def analyze_indicators(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze technical indicators for signals.
        
        Args:
            data: DataFrame with price data and indicators
        
        Returns:
            Dictionary with indicator analysis
        """
        if len(data) < 50:
            return {'error': 'Insufficient data for analysis'}
        
        latest = data.iloc[-1]
        signals = {}
        
        # RSI Analysis
        if 'RSI' in data.columns and not pd.isna(latest['RSI']):
            rsi_val = latest['RSI']
            if rsi_val < 30:
                signals['RSI'] = {'value': round(rsi_val, 2), 'signal': 'OVERSOLD', 'strength': 'STRONG_BUY'}
            elif rsi_val < 40:
                signals['RSI'] = {'value': round(rsi_val, 2), 'signal': 'WEAK_OVERSOLD', 'strength': 'BUY'}
            elif rsi_val > 70:
                signals['RSI'] = {'value': round(rsi_val, 2), 'signal': 'OVERBOUGHT', 'strength': 'STRONG_SELL'}
            elif rsi_val > 60:
                signals['RSI'] = {'value': round(rsi_val, 2), 'signal': 'WEAK_OVERBOUGHT', 'strength': 'SELL'}
            else:
                signals['RSI'] = {'value': round(rsi_val, 2), 'signal': 'NEUTRAL', 'strength': 'HOLD'}
        
        # MACD Analysis
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            macd_val = latest['MACD']
            signal_val = latest['MACD_Signal']
            if not pd.isna(macd_val) and not pd.isna(signal_val):
                if macd_val > signal_val:
                    signals['MACD'] = {'signal': 'BULLISH', 'strength': 'BUY'}
                else:
                    signals['MACD'] = {'signal': 'BEARISH', 'strength': 'SELL'}
        
        # Bollinger Bands Analysis
        if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'Close']):
            price = latest['Close']
            bb_upper = latest['BB_Upper']
            bb_lower = latest['BB_Lower']
            
            if not pd.isna(bb_upper) and not pd.isna(bb_lower):
                if price >= bb_upper:
                    signals['Bollinger'] = {'signal': 'OVERBOUGHT', 'strength': 'SELL'}
                elif price <= bb_lower:
                    signals['Bollinger'] = {'signal': 'OVERSOLD', 'strength': 'BUY'}
                else:
                    signals['Bollinger'] = {'signal': 'NEUTRAL', 'strength': 'HOLD'}
        
        # Moving Average Analysis
        if 'SMA_50' in data.columns and 'SMA_200' in data.columns:
            sma_50 = latest['SMA_50']
            sma_200 = latest['SMA_200']
            
            if not pd.isna(sma_50) and not pd.isna(sma_200):
                if sma_50 > sma_200:
                    signals['MA_Trend'] = {'signal': 'GOLDEN_CROSS', 'strength': 'BUY'}
                else:
                    signals['MA_Trend'] = {'signal': 'DEATH_CROSS', 'strength': 'SELL'}
        
        # Stochastic Analysis
        if 'Stoch_K' in data.columns and 'Stoch_D' in data.columns:
            stoch_k = latest['Stoch_K']
            stoch_d = latest['Stoch_D']
            
            if not pd.isna(stoch_k) and not pd.isna(stoch_d):
                if stoch_k < 20:
                    signals['Stochastic'] = {'value': round(stoch_k, 2), 'signal': 'OVERSOLD', 'strength': 'BUY'}
                elif stoch_k > 80:
                    signals['Stochastic'] = {'value': round(stoch_k, 2), 'signal': 'OVERBOUGHT', 'strength': 'SELL'}
                else:
                    signals['Stochastic'] = {'value': round(stoch_k, 2), 'signal': 'NEUTRAL', 'strength': 'HOLD'}
        
        return signals
    
    def calculate_probability_score(self, signals: Dict[str, any]) -> Dict[str, float]:
        """
        Calculate probability scores for buy/sell/hold.
        
        Args:
            signals: Dictionary of technical indicator signals
        
        Returns:
            Dictionary with probability scores
        """
        buy_score = 0
        sell_score = 0
        hold_score = 0
        total_signals = 0
        
        for indicator, signal_data in signals.items():
            if 'strength' not in signal_data:
                continue
            
            strength = signal_data['strength']
            total_signals += 1
            
            if 'BUY' in strength:
                if 'STRONG' in strength:
                    buy_score += 2
                else:
                    buy_score += 1
            elif 'SELL' in strength:
                if 'STRONG' in strength:
                    sell_score += 2
                else:
                    sell_score += 1
            else:
                hold_score += 1
        
        if total_signals == 0:
            return {'buy': 33.33, 'sell': 33.33, 'hold': 33.33}
        
        # Calculate percentages
        total_score = buy_score + sell_score + hold_score
        buy_prob = (buy_score / total_score) * 100 if total_score > 0 else 0
        sell_prob = (sell_score / total_score) * 100 if total_score > 0 else 0
        hold_prob = (hold_score / total_score) * 100 if total_score > 0 else 0
        
        return {
            'buy': round(buy_prob, 2),
            'sell': round(sell_prob, 2),
            'hold': round(hold_prob, 2)
        }
    
    def generate_recommendation(self, probabilities: Dict[str, float], 
                                trend: str, trend_strength: float) -> Dict[str, any]:
        """
        Generate final recommendation.
        
        Args:
            probabilities: Probability scores
            trend: Overall trend direction
            trend_strength: Trend strength percentage
        
        Returns:
            Dictionary with recommendation details
        """
        # Determine primary recommendation
        max_prob = max(probabilities.values())
        recommendation = [k for k, v in probabilities.items() if v == max_prob][0].upper()
        
        # Adjust confidence based on trend
        confidence = max_prob
        if trend == 'BULLISH' and recommendation == 'BUY':
            confidence = min(confidence * 1.2, 100)
        elif trend == 'BEARISH' and recommendation == 'SELL':
            confidence = min(confidence * 1.2, 100)
        elif trend == 'SIDEWAYS':
            confidence = confidence * 0.9
        
        # Determine risk level
        if confidence > 70:
            risk_level = "LOW"
        elif confidence > 50:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        return {
            'recommendation': recommendation,
            'confidence': round(confidence, 2),
            'risk_level': risk_level,
            'trend': trend,
            'trend_strength': trend_strength,
            'probabilities': probabilities
        }
    
    def predict(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Generate complete prediction for an asset.
        
        Args:
            data: DataFrame with price data and indicators
        
        Returns:
            Complete prediction dictionary
        """
        # Analyze indicators
        signals = self.analyze_indicators(data)
        
        if 'error' in signals:
            return signals
        
        # Calculate probabilities
        probabilities = self.calculate_probability_score(signals)
        
        # Get trend information
        trend = self.trend_analyzer.identify_trend(data)
        trend_strength = self.trend_analyzer.calculate_trend_strength(data)
        
        # Detect crossovers
        crossover = self.trend_analyzer.detect_crossover(data, 'SMA_50', 'SMA_200')
        
        # Get support/resistance
        levels = self.trend_analyzer.identify_support_resistance(data)
        
        # Volume analysis
        volume_analysis = self.trend_analyzer.analyze_volume_trend(data)
        
        # Generate recommendation
        recommendation = self.generate_recommendation(probabilities, trend, trend_strength)
        
        return {
            'recommendation': recommendation,
            'technical_signals': signals,
            'trend_analysis': {
                'trend': trend,
                'strength': trend_strength,
                'crossover': crossover,
                'support_resistance': levels
            },
            'volume_analysis': volume_analysis,
            'timestamp': data.index[-1].strftime('%Y-%m-%d %H:%M:%S') if hasattr(data.index[-1], 'strftime') else str(data.index[-1])
        }

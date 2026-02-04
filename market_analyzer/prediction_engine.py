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
        Analyze technical indicators for signals with enhanced accuracy.
        
        Args:
            data: DataFrame with price data and indicators
        
        Returns:
            Dictionary with indicator analysis
        """
        if len(data) < 50:
            return {'error': 'Insufficient data for analysis'}
        
        latest = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else latest
        signals = {}
        
        # Enhanced RSI Analysis with divergence detection
        if 'RSI' in data.columns and not pd.isna(latest['RSI']):
            rsi_val = latest['RSI']
            rsi_prev = prev['RSI'] if not pd.isna(prev['RSI']) else rsi_val
            rsi_momentum = rsi_val - rsi_prev
            
            # More nuanced RSI signals
            if rsi_val < 25:
                signals['RSI'] = {'value': round(rsi_val, 2), 'signal': 'EXTREME_OVERSOLD', 'strength': 'STRONG_BUY', 'weight': 2.5}
            elif rsi_val < 30:
                signals['RSI'] = {'value': round(rsi_val, 2), 'signal': 'OVERSOLD', 'strength': 'STRONG_BUY', 'weight': 2.0}
            elif rsi_val < 40:
                signals['RSI'] = {'value': round(rsi_val, 2), 'signal': 'WEAK_OVERSOLD', 'strength': 'BUY', 'weight': 1.5}
            elif rsi_val > 75:
                signals['RSI'] = {'value': round(rsi_val, 2), 'signal': 'EXTREME_OVERBOUGHT', 'strength': 'STRONG_SELL', 'weight': 2.5}
            elif rsi_val > 70:
                signals['RSI'] = {'value': round(rsi_val, 2), 'signal': 'OVERBOUGHT', 'strength': 'STRONG_SELL', 'weight': 2.0}
            elif rsi_val > 60:
                signals['RSI'] = {'value': round(rsi_val, 2), 'signal': 'WEAK_OVERBOUGHT', 'strength': 'SELL', 'weight': 1.5}
            else:
                # Check momentum for neutral zone
                if rsi_momentum > 5:
                    signals['RSI'] = {'value': round(rsi_val, 2), 'signal': 'NEUTRAL_BULLISH', 'strength': 'BUY', 'weight': 1.0}
                elif rsi_momentum < -5:
                    signals['RSI'] = {'value': round(rsi_val, 2), 'signal': 'NEUTRAL_BEARISH', 'strength': 'SELL', 'weight': 1.0}
                else:
                    signals['RSI'] = {'value': round(rsi_val, 2), 'signal': 'NEUTRAL', 'strength': 'HOLD', 'weight': 0.5}
        
        # Enhanced MACD Analysis with histogram and crossover detection
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns and 'MACD_Hist' in data.columns:
            macd_val = latest['MACD']
            signal_val = latest['MACD_Signal']
            hist_val = latest['MACD_Hist']
            prev_hist = prev['MACD_Hist'] if 'MACD_Hist' in data.columns and not pd.isna(prev['MACD_Hist']) else hist_val
            
            if not pd.isna(macd_val) and not pd.isna(signal_val) and not pd.isna(hist_val):
                # Detect crossovers
                crossover_detected = (hist_val > 0 and prev_hist < 0) or (hist_val < 0 and prev_hist > 0)
                
                if macd_val > signal_val:
                    if crossover_detected and hist_val > 0:
                        signals['MACD'] = {'signal': 'BULLISH_CROSSOVER', 'strength': 'STRONG_BUY', 'weight': 2.5}
                    elif hist_val > 0.5:
                        signals['MACD'] = {'signal': 'STRONG_BULLISH', 'strength': 'BUY', 'weight': 2.0}
                    else:
                        signals['MACD'] = {'signal': 'BULLISH', 'strength': 'BUY', 'weight': 1.5}
                else:
                    if crossover_detected and hist_val < 0:
                        signals['MACD'] = {'signal': 'BEARISH_CROSSOVER', 'strength': 'STRONG_SELL', 'weight': 2.5}
                    elif hist_val < -0.5:
                        signals['MACD'] = {'signal': 'STRONG_BEARISH', 'strength': 'SELL', 'weight': 2.0}
                    else:
                        signals['MACD'] = {'signal': 'BEARISH', 'strength': 'SELL', 'weight': 1.5}
        
        # Enhanced Bollinger Bands Analysis with band width
        if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle', 'Close']):
            price = latest['Close']
            bb_upper = latest['BB_Upper']
            bb_lower = latest['BB_Lower']
            bb_middle = latest['BB_Middle']
            
            if not pd.isna(bb_upper) and not pd.isna(bb_lower) and not pd.isna(bb_middle):
                band_width = ((bb_upper - bb_lower) / bb_middle) * 100
                position = ((price - bb_lower) / (bb_upper - bb_lower)) * 100
                
                if price >= bb_upper:
                    signals['Bollinger'] = {'signal': 'UPPER_BAND_TOUCH', 'strength': 'SELL', 'weight': 1.8}
                elif price <= bb_lower:
                    signals['Bollinger'] = {'signal': 'LOWER_BAND_TOUCH', 'strength': 'BUY', 'weight': 1.8}
                elif position > 75:
                    signals['Bollinger'] = {'signal': 'NEAR_UPPER', 'strength': 'SELL', 'weight': 1.2}
                elif position < 25:
                    signals['Bollinger'] = {'signal': 'NEAR_LOWER', 'strength': 'BUY', 'weight': 1.2}
                else:
                    signals['Bollinger'] = {'signal': 'MID_RANGE', 'strength': 'HOLD', 'weight': 0.5}
        
        # Enhanced Moving Average Analysis with multiple timeframes
        price = latest['Close']
        ma_signals = []
        
        if 'SMA_20' in data.columns and not pd.isna(latest['SMA_20']):
            if price > latest['SMA_20']:
                ma_signals.append('BUY')
            else:
                ma_signals.append('SELL')
        
        if 'SMA_50' in data.columns and not pd.isna(latest['SMA_50']):
            if price > latest['SMA_50']:
                ma_signals.append('BUY')
            else:
                ma_signals.append('SELL')
        
        # Golden/Death Cross with enhanced detection
        if 'SMA_50' in data.columns and 'SMA_200' in data.columns:
            sma_50 = latest['SMA_50']
            sma_200 = latest['SMA_200']
            prev_sma_50 = prev['SMA_50'] if not pd.isna(prev['SMA_50']) else sma_50
            prev_sma_200 = prev['SMA_200'] if not pd.isna(prev['SMA_200']) else sma_200
            
            if not pd.isna(sma_50) and not pd.isna(sma_200):
                # Check for recent crossover
                crossover_occurred = (sma_50 > sma_200 and prev_sma_50 <= prev_sma_200) or \
                                   (sma_50 < sma_200 and prev_sma_50 >= prev_sma_200)
                
                if sma_50 > sma_200:
                    if crossover_occurred:
                        signals['MA_Trend'] = {'signal': 'GOLDEN_CROSS_CONFIRMED', 'strength': 'STRONG_BUY', 'weight': 3.0}
                    else:
                        distance = ((sma_50 - sma_200) / sma_200) * 100
                        if distance > 5:
                            signals['MA_Trend'] = {'signal': 'STRONG_UPTREND', 'strength': 'BUY', 'weight': 2.0}
                        else:
                            signals['MA_Trend'] = {'signal': 'UPTREND', 'strength': 'BUY', 'weight': 1.5}
                    ma_signals.append('BUY')
                else:
                    if crossover_occurred:
                        signals['MA_Trend'] = {'signal': 'DEATH_CROSS_CONFIRMED', 'strength': 'STRONG_SELL', 'weight': 3.0}
                    else:
                        distance = ((sma_200 - sma_50) / sma_200) * 100
                        if distance > 5:
                            signals['MA_Trend'] = {'signal': 'STRONG_DOWNTREND', 'strength': 'SELL', 'weight': 2.0}
                        else:
                            signals['MA_Trend'] = {'signal': 'DOWNTREND', 'strength': 'SELL', 'weight': 1.5}
                    ma_signals.append('SELL')
        
        # Overall MA consensus
        if ma_signals:
            buy_count = ma_signals.count('BUY')
            sell_count = ma_signals.count('SELL')
            if buy_count > sell_count:
                signals['MA_Consensus'] = {'signal': 'BULLISH_MAS', 'strength': 'BUY', 'weight': 1.5}
            elif sell_count > buy_count:
                signals['MA_Consensus'] = {'signal': 'BEARISH_MAS', 'strength': 'SELL', 'weight': 1.5}
        
        # Enhanced Stochastic Analysis with crossovers
        if 'Stoch_K' in data.columns and 'Stoch_D' in data.columns:
            stoch_k = latest['Stoch_K']
            stoch_d = latest['Stoch_D']
            prev_k = prev['Stoch_K'] if not pd.isna(prev['Stoch_K']) else stoch_k
            prev_d = prev['Stoch_D'] if not pd.isna(prev['Stoch_D']) else stoch_d
            
            if not pd.isna(stoch_k) and not pd.isna(stoch_d):
                # Detect K/D crossover
                bullish_cross = stoch_k > stoch_d and prev_k <= prev_d
                bearish_cross = stoch_k < stoch_d and prev_k >= prev_d
                
                if stoch_k < 20:
                    if bullish_cross:
                        signals['Stochastic'] = {'value': round(stoch_k, 2), 'signal': 'OVERSOLD_CROSSOVER', 'strength': 'STRONG_BUY', 'weight': 2.5}
                    else:
                        signals['Stochastic'] = {'value': round(stoch_k, 2), 'signal': 'OVERSOLD', 'strength': 'BUY', 'weight': 2.0}
                elif stoch_k > 80:
                    if bearish_cross:
                        signals['Stochastic'] = {'value': round(stoch_k, 2), 'signal': 'OVERBOUGHT_CROSSOVER', 'strength': 'STRONG_SELL', 'weight': 2.5}
                    else:
                        signals['Stochastic'] = {'value': round(stoch_k, 2), 'signal': 'OVERBOUGHT', 'strength': 'SELL', 'weight': 2.0}
                else:
                    if bullish_cross:
                        signals['Stochastic'] = {'value': round(stoch_k, 2), 'signal': 'BULLISH_CROSSOVER', 'strength': 'BUY', 'weight': 1.5}
                    elif bearish_cross:
                        signals['Stochastic'] = {'value': round(stoch_k, 2), 'signal': 'BEARISH_CROSSOVER', 'strength': 'SELL', 'weight': 1.5}
                    else:
                        signals['Stochastic'] = {'value': round(stoch_k, 2), 'signal': 'NEUTRAL', 'strength': 'HOLD', 'weight': 0.5}
        
        # Volume confirmation
        if 'Volume' in data.columns and len(data) > 20:
            avg_volume = data['Volume'].tail(20).mean()
            current_volume = latest['Volume']
            
            if current_volume > avg_volume * 1.5:
                signals['Volume'] = {'signal': 'HIGH_VOLUME', 'strength': 'CONFIRMATION', 'weight': 1.5}
            elif current_volume < avg_volume * 0.5:
                signals['Volume'] = {'signal': 'LOW_VOLUME', 'strength': 'WEAK_SIGNAL', 'weight': 0.5}
        
        return signals
    
    def calculate_probability_score(self, signals: Dict[str, any]) -> Dict[str, float]:
        """
        Calculate weighted probability scores for buy/sell/hold with improved accuracy.
        
        Args:
            signals: Dictionary of technical indicator signals with weights
        
        Returns:
            Dictionary with probability scores and confidence metrics
        """
        buy_score = 0
        sell_score = 0
        hold_score = 0
        total_weight = 0
        signal_alignment = 0  # How well signals agree
        
        for indicator, signal_data in signals.items():
            if 'strength' not in signal_data:
                continue
            
            strength = signal_data['strength']
            weight = signal_data.get('weight', 1.0)  # Use custom weight if provided
            total_weight += weight
            
            if strength == 'CONFIRMATION':
                # Volume confirmation amplifies other signals
                continue
            elif strength == 'WEAK_SIGNAL':
                # Low volume reduces confidence
                total_weight -= 0.5
                continue
            
            if 'BUY' in strength:
                if 'STRONG' in strength:
                    buy_score += 3.0 * weight
                    signal_alignment += 2
                else:
                    buy_score += 1.5 * weight
                    signal_alignment += 1
            elif 'SELL' in strength:
                if 'STRONG' in strength:
                    sell_score += 3.0 * weight
                    signal_alignment += 2
                else:
                    sell_score += 1.5 * weight
                    signal_alignment += 1
            else:
                hold_score += 1.0 * weight
        
        if total_weight == 0:
            return {
                'buy': 33.33,
                'sell': 33.33,
                'hold': 33.33,
                'confidence_multiplier': 0.5,
                'signal_strength': 'WEAK'
            }
        
        # Calculate weighted percentages
        total_score = buy_score + sell_score + hold_score
        buy_prob = (buy_score / total_score) * 100 if total_score > 0 else 0
        sell_prob = (sell_score / total_score) * 100 if total_score > 0 else 0
        hold_prob = (hold_score / total_score) * 100 if total_score > 0 else 0
        
        # Calculate signal alignment (how much signals agree)
        max_prob = max(buy_prob, sell_prob, hold_prob)
        alignment_score = (max_prob - 33.33) / 66.67  # Normalized 0-1
        
        # Confidence multiplier based on signal strength
        if alignment_score > 0.7:
            confidence_multiplier = 1.3
            signal_strength = 'VERY_STRONG'
        elif alignment_score > 0.5:
            confidence_multiplier = 1.15
            signal_strength = 'STRONG'
        elif alignment_score > 0.3:
            confidence_multiplier = 1.0
            signal_strength = 'MODERATE'
        else:
            confidence_multiplier = 0.85
            signal_strength = 'WEAK'
        
        return {
            'buy': round(buy_prob, 2),
            'sell': round(sell_prob, 2),
            'hold': round(hold_prob, 2),
            'confidence_multiplier': confidence_multiplier,
            'signal_strength': signal_strength,
            'total_indicators': len([s for s in signals.values() if 'strength' in s])
        }
    
    def generate_recommendation(self, probabilities: Dict[str, float], 
                                trend: str, trend_strength: float, signals: Dict[str, any]) -> Dict[str, any]:
        """
        Generate final recommendation with enhanced confidence calculation.
        
        Args:
            probabilities: Probability scores with confidence metrics
            trend: Overall trend direction
            trend_strength: Trend strength percentage
            signals: Technical signals for reasoning
        
        Returns:
            Dictionary with detailed recommendation
        """
        # Extract probability values
        buy_prob = probabilities['buy']
        sell_prob = probabilities['sell']
        hold_prob = probabilities['hold']
        confidence_multiplier = probabilities.get('confidence_multiplier', 1.0)
        signal_strength = probabilities.get('signal_strength', 'MODERATE')
        total_indicators = probabilities.get('total_indicators', 0)
        
        # Determine primary recommendation
        max_prob = max(buy_prob, sell_prob, hold_prob)
        
        if buy_prob == max_prob:
            recommendation = 'BUY'
        elif sell_prob == max_prob:
            recommendation = 'SELL'
        else:
            recommendation = 'HOLD'
        
        # Calculate base confidence
        confidence = max_prob
        
        # Apply confidence multiplier from signal alignment
        confidence *= confidence_multiplier
        
        # Adjust confidence based on trend alignment (stronger factor)
        if trend == 'BULLISH' and recommendation == 'BUY':
            confidence = min(confidence * 1.25, 98)
            trend_aligned = True
        elif trend == 'BEARISH' and recommendation == 'SELL':
            confidence = min(confidence * 1.25, 98)
            trend_aligned = True
        elif trend == 'SIDEWAYS' and recommendation == 'HOLD':
            confidence = min(confidence * 1.15, 98)
            trend_aligned = True
        elif trend == 'SIDEWAYS':
            confidence *= 0.85
            trend_aligned = False
        else:
            # Recommendation contradicts trend - reduce confidence significantly
            confidence *= 0.70
            trend_aligned = False
        
        # Adjust for trend strength
        if trend_strength > 75 and trend_aligned:
            confidence = min(confidence * 1.1, 98)
        elif trend_strength < 40:
            confidence *= 0.9
        
        # Boost confidence if we have many confirming indicators
        if total_indicators >= 6 and signal_strength in ['STRONG', 'VERY_STRONG']:
            confidence = min(confidence * 1.05, 98)
        elif total_indicators < 4:
            confidence *= 0.95
        
        # Generate strong buy/sell for very high confidence
        if confidence > 85 and recommendation != 'HOLD':
            recommendation = f'STRONG_{recommendation}'
        
        # Determine risk level with more granular assessment
        if confidence > 80:
            risk_level = "LOW"
        elif confidence > 65:
            risk_level = "MEDIUM-LOW"
        elif confidence > 50:
            risk_level = "MEDIUM"
        elif confidence > 35:
            risk_level = "MEDIUM-HIGH"
        else:
            risk_level = "HIGH"
        
        # Generate reasoning based on signals
        reasoning = self._generate_reasoning(signals, trend, recommendation, trend_aligned)
        
        return {
            'recommendation': recommendation,
            'confidence': round(min(confidence, 98), 2),  # Cap at 98% for realism
            'risk_level': risk_level,
            'trend': trend,
            'trend_strength': trend_strength,
            'trend_aligned': trend_aligned,
            'signal_strength': signal_strength,
            'total_indicators': total_indicators,
            'probabilities': {
                'buy': buy_prob,
                'sell': sell_prob,
                'hold': hold_prob
            },
            'reasoning': reasoning
        }
    
    def _generate_reasoning(self, signals: Dict[str, any], trend: str, 
                           recommendation: str, trend_aligned: bool) -> List[str]:
        """Generate human-readable reasoning for the recommendation."""
        reasoning = []
        
        # Count signal types
        buy_signals = sum(1 for s in signals.values() if 'BUY' in s.get('strength', ''))
        sell_signals = sum(1 for s in signals.values() if 'SELL' in s.get('strength', ''))
        
        # Overall trend reasoning
        if trend_aligned:
            reasoning.append(f"Recommendation aligns with {trend.lower()} market trend")
        else:
            reasoning.append(f"Recommendation goes against {trend.lower()} trend - exercise caution")
        
        # Signal consensus reasoning
        if 'BUY' in recommendation:
            reasoning.append(f"{buy_signals} technical indicators showing bullish signals")
            if 'MA_Trend' in signals and 'GOLDEN_CROSS' in signals['MA_Trend'].get('signal', ''):
                reasoning.append("Golden Cross pattern detected - strong long-term uptrend signal")
            if 'RSI' in signals and 'OVERSOLD' in signals['RSI'].get('signal', ''):
                reasoning.append("RSI indicates oversold conditions - potential bounce expected")
            if 'MACD' in signals and 'BULLISH' in signals['MACD'].get('signal', ''):
                reasoning.append("MACD showing bullish momentum")
        elif 'SELL' in recommendation:
            reasoning.append(f"{sell_signals} technical indicators showing bearish signals")
            if 'MA_Trend' in signals and 'DEATH_CROSS' in signals['MA_Trend'].get('signal', ''):
                reasoning.append("Death Cross pattern detected - strong long-term downtrend signal")
            if 'RSI' in signals and 'OVERBOUGHT' in signals['RSI'].get('signal', ''):
                reasoning.append("RSI indicates overbought conditions - potential correction expected")
            if 'MACD' in signals and 'BEARISH' in signals['MACD'].get('signal', ''):
                reasoning.append("MACD showing bearish momentum")
        else:
            reasoning.append("Mixed signals suggest waiting for clearer direction")
        
        # Volume reasoning
        if 'Volume' in signals:
            if signals['Volume'].get('strength') == 'CONFIRMATION':
                reasoning.append("Above-average volume confirms the signal strength")
            elif signals['Volume'].get('strength') == 'WEAK_SIGNAL':
                reasoning.append("Low volume suggests weak conviction - be cautious")
        
        return reasoning
    
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
        recommendation = self.generate_recommendation(probabilities, trend, trend_strength, signals)
        
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

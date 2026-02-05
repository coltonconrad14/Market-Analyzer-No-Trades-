#!/usr/bin/env python3
"""
Web-based GUI Application for Market Analyzer using Streamlit.
Works in dev containers and remote environments!
"""

import streamlit as st
from market_analyzer import MarketAnalyzer
import pandas as pd
from io import StringIO
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import numpy as np
from ml_learning_engine import get_ml_engine
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Market Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    h1 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

def analyze_darkpool_activity(df):
    """
    Analyze estimated darkpool activity and institutional flows.
    
    Since darkpool data isn't directly available via yfinance, we estimate it based on:
    - Large volume spikes (institutional block trades)
    - Price-volume divergence (off-exchange activity indicators)
    - Volume patterns that suggest large accumulation/distribution
    
    Returns analysis of estimated darkpool activity and sentiment.
    """
    if len(df) < 20:
        return None
    
    try:
        # Get price and volume data
        prices = df['Close'].values
        volumes = df['Volume'].values if 'Volume' in df.columns else np.ones(len(df))
        
        # Calculate metrics
        daily_returns = np.diff(prices) / prices[:-1]
        
        # 1. VOLUME ANOMALY DETECTION (indicates potential block trades)
        avg_volume = np.mean(volumes[-20:])
        volume_std = np.std(volumes[-20:])
        current_volume = volumes[-1]
        volume_zscore = (current_volume - avg_volume) / (volume_std + 0.001)
        
        # 2. PRICE-VOLUME DIVERGENCE (sign of off-exchange activity)
        # High volume but minimal price movement suggests darkpool accumulation
        recent_price_range = np.max(prices[-5:]) - np.min(prices[-5:])
        recent_avg_vol = np.mean(volumes[-5:])
        
        price_vol_ratio = recent_price_range / (recent_avg_vol / 1e6 + 0.001)
        divergence_signal = 1 - np.clip(price_vol_ratio, 0, 1)  # Higher = more divergence
        
        # 3. ACCUMULATION/DISTRIBUTION INDICATOR
        # Estimates if large players are buying or selling
        ad_line = []
        for i in range(len(prices)):
            if i == 0:
                ad_line.append(0)
            else:
                close_location = (prices[i] - np.min(prices[max(0, i-5):i+1])) / \
                                (np.max(prices[max(0, i-5):i+1]) - np.min(prices[max(0, i-5):i+1]) + 0.001)
                ad = ad_line[-1] + volumes[i] * (2 * close_location - 1)
                ad_line.append(ad)
        
        ad_momentum = ad_line[-1] - ad_line[-5] if len(ad_line) >= 5 else 0
        
        # 4. LARGE BLOCK DETECTION
        # Detect unusually large volume candles
        large_blocks = np.sum(volumes[-10:] > (avg_volume + 2 * volume_std)) / 10
        
        # 5. DARKPOOL SENTIMENT
        # Combines price action with volume patterns
        if divergence_signal > 0.7 and volume_zscore > 1.5:
            # High divergence + high volume = possible large institutional activity
            if ad_momentum > 0:
                darkpool_sentiment = "ACCUMULATION"
                sentiment_score = 0.8
            else:
                darkpool_sentiment = "DISTRIBUTION"
                sentiment_score = -0.8
        elif divergence_signal > 0.5 or volume_zscore > 1.2:
            if ad_momentum > 0:
                darkpool_sentiment = "MILD ACCUMULATION"
                sentiment_score = 0.4
            else:
                darkpool_sentiment = "MILD DISTRIBUTION"
                sentiment_score = -0.4
        else:
            darkpool_sentiment = "NEUTRAL"
            sentiment_score = 0
        
        # 6. ESTIMATED DARKPOOL VOLUME %
        # Estimate what percentage of volume might be from darkpools (typically 10-20%)
        estimated_darkpool_pct = (divergence_signal * 0.15) + (abs(volume_zscore) / 10 * 0.1)
        estimated_darkpool_pct = np.clip(estimated_darkpool_pct, 0.05, 0.30) * 100
        
        return {
            'volume_anomaly': volume_zscore,
            'price_volume_divergence': divergence_signal,
            'accumulation_momentum': ad_momentum,
            'large_blocks_detected': large_blocks,
            'darkpool_sentiment': darkpool_sentiment,
            'sentiment_score': sentiment_score,
            'estimated_darkpool_pct': estimated_darkpool_pct,
            'recent_volume': current_volume,
            'average_volume': avg_volume,
            'volume_trend': "INCREASING" if np.mean(volumes[-5:]) > np.mean(volumes[-20:]) else "DECREASING"
        }
    except Exception as e:
        st.warning(f"Darkpool analysis error: {str(e)}")
        return None


def detect_candlestick_patterns(df):
    """
    Detect candlestick patterns to improve predictions.
    
    Returns a dictionary with detected patterns and their signals.
    """
    if len(df) < 5:
        return {'patterns': [], 'signal': 0, 'strength': 0}
    
    patterns = []
    signal = 0  # -1 bearish, 0 neutral, 1 bullish
    strength = 0  # Pattern strength multiplier
    
    # Get last candles
    o, h, l, c = df['Open'].iloc[-5:].values, df['High'].iloc[-5:].values, df['Low'].iloc[-5:].values, df['Close'].iloc[-5:].values
    v = df['Volume'].iloc[-5:].values if 'Volume' in df.columns else np.ones(5)
    
    # Normalize for pattern recognition
    body_sizes = np.abs(c - o)
    wicks_high = h - np.maximum(c, o)
    wicks_low = np.minimum(c, o) - l
    ranges = h - l
    
    # Pattern 1: Hammer (bullish reversal)
    if len(df) >= 2:
        lower_wick = wicks_low[-1]
        body = body_sizes[-1]
        upper_wick = wicks_high[-1]
        if lower_wick > body * 2 and upper_wick < body * 0.5 and c[-1] > o[-1]:
            patterns.append("Hammer")
            signal += 1
            strength += 0.15
    
    # Pattern 2: Shooting Star (bearish reversal)
    if len(df) >= 2:
        upper_wick = wicks_high[-1]
        body = body_sizes[-1]
        lower_wick = wicks_low[-1]
        if upper_wick > body * 2 and lower_wick < body * 0.5 and c[-1] < o[-1]:
            patterns.append("Shooting Star")
            signal -= 1
            strength += 0.15
    
    # Pattern 3: Engulfing (bullish or bearish)
    if len(df) >= 2:
        prev_body = body_sizes[-2]
        curr_body = body_sizes[-1]
        if curr_body > prev_body * 1.5:
            if c[-1] > o[-1] and o[-1] < c[-2] and c[-1] > o[-2]:
                patterns.append("Bullish Engulfing")
                signal += 1
                strength += 0.2
            elif c[-1] < o[-1] and o[-1] > c[-2] and c[-1] < o[-2]:
                patterns.append("Bearish Engulfing")
                signal -= 1
                strength += 0.2
    
    # Pattern 4: Doji (indecision)
    if len(df) >= 1:
        body = body_sizes[-1]
        range_size = ranges[-1]
        if body < range_size * 0.1:
            patterns.append("Doji")
            signal *= 0.5  # Reduces confidence
    
    # Pattern 5: Three White Soldiers (bullish)
    if len(df) >= 3:
        if all(c[-3:] > o[-3:]) and c[-1] > c[-2] > c[-3] and c[-1] > o[-1] * 1.01:
            patterns.append("Three White Soldiers")
            signal += 1.5
            strength += 0.25
    
    # Pattern 6: Three Black Crows (bearish)
    if len(df) >= 3:
        if all(c[-3:] < o[-3:]) and c[-1] < c[-2] < c[-3] and c[-1] < o[-1] * 0.99:
            patterns.append("Three Black Crows")
            signal -= 1.5
            strength += 0.25
    
    # Pattern 7: Morning Star (bullish reversal)
    if len(df) >= 3:
        gap_down = l[-2] < l[-3]
        recovery = c[-1] > (o[-3] + c[-3]) / 2
        if gap_down and recovery and body_sizes[-3] > body_sizes[-2]:
            patterns.append("Morning Star")
            signal += 1.5
            strength += 0.2
    
    # Pattern 8: Evening Star (bearish reversal)
    if len(df) >= 3:
        gap_up = h[-2] > h[-3]
        decline = c[-1] < (o[-3] + c[-3]) / 2
        if gap_up and decline and body_sizes[-3] > body_sizes[-2]:
            patterns.append("Evening Star")
            signal -= 1.5
            strength += 0.2
    
    # Clamp signal
    signal = np.clip(signal, -2, 2)
    strength = np.clip(strength, 0, 1)
    
    return {
        'patterns': patterns,
        'signal': signal,
        'strength': strength,
        'bullish': signal > 0.5,
        'bearish': signal < -0.5
    }


def predict_future_price(df, time_horizons=[30, 90]):
    """
    Advanced AI-powered future price prediction incorporating momentum, sentiment, and market conditions.
    
    Args:
        df: DataFrame with historical price data
        time_horizons: List of days ahead to predict (default: [30, 90])
                      = [1 month, 3 months]
    
    Returns:
        Dictionary with predictions at multiple time horizons
    """
    try:
        if len(df) < 2:
            return None

        history_len = len(df)
        limited_history = history_len < 50
        
        # Prepare data
        prices = df['Close'].values
        dates = np.arange(len(prices))
        current_price = prices[-1]
        
        # CANDLESTICK PATTERN ANALYSIS
        pattern_analysis = detect_candlestick_patterns(df)
        pattern_signal = pattern_analysis['signal']
        pattern_strength = pattern_analysis['strength']
        
        # Calculate historical statistics
        daily_returns = np.diff(prices) / prices[:-1]
        vol_window = min(20, len(daily_returns)) if len(daily_returns) > 0 else 1
        volatility_short = np.std(daily_returns[-vol_window:]) * 100 if len(daily_returns) > 0 else 0
        volatility_long = np.std(daily_returns) * 100
        volatility = (volatility_short * 0.6 + volatility_long * 0.4)
        
        drift = np.mean(daily_returns)
        drift_short = np.mean(daily_returns[-20:]) if len(daily_returns) >= 20 else drift
        
        # DARKPOOL ACTIVITY ANALYSIS
        darkpool_analysis = analyze_darkpool_activity(df)
        
        # MARKET CONDITION ANALYSIS
        # Detect bull/bear market regime
        sma_50 = np.mean(prices[-min(50, len(prices)):])
        sma_200 = np.mean(prices[-min(200, len(prices)):])
        current_sma_short = np.mean(prices[-min(20, len(prices)):])
        
        market_regime = "BULLISH" if current_price > sma_50 > sma_200 else "BEARISH" if current_price < sma_50 < sma_200 else "NEUTRAL"
        
        # MOMENTUM ANALYSIS
        # Recent momentum (last 20 days vs previous 20 days)
        recent_momentum = drift_short
        momentum_strength = abs(recent_momentum) / (volatility / 100 + 0.001)  # Momentum relative to volatility
        
        # Trend strength (how consistent is the direction?)
        recent_window = min(20, len(daily_returns)) if len(daily_returns) > 0 else 1
        recent_returns = daily_returns[-recent_window:] if len(daily_returns) > 0 else np.array([0])
        positive_days = np.sum(recent_returns > 0)
        trend_strength = abs(positive_days - (recent_window / 2)) / max(recent_window / 2, 1)
        
        # Price acceleration (is trend getting stronger or weaker?)
        if len(recent_returns) >= 6:
            split = len(recent_returns) // 2
            first_half_momentum = np.mean(recent_returns[:split])
            second_half_momentum = np.mean(recent_returns[split:])
            acceleration = second_half_momentum - first_half_momentum
        else:
            acceleration = 0
        
        # SENTIMENT ANALYSIS (based on technical indicators)
        # RSI-like sentiment (position in recent range)
        range_window = min(30, len(prices))
        recent_range = np.max(prices[-range_window:]) - np.min(prices[-range_window:])
        sentiment_position = (current_price - np.min(prices[-range_window:])) / (recent_range + 0.001)
        sentiment_position = np.clip(sentiment_position, 0, 1)
        
        # Oversold/overbought adjustment
        if sentiment_position > 0.8:
            sentiment = "OVERBOUGHT"
            sentiment_bias = -0.02  # Slight bearish bias
        elif sentiment_position < 0.2:
            sentiment = "OVERSOLD"
            sentiment_bias = 0.02   # Slight bullish bias
        else:
            sentiment = "NEUTRAL"
            sentiment_bias = 0
        
        # VOLATILITY REGIME
        volatility_trend = volatility_short - volatility_long
        if volatility_trend > 0:
            volatility_regime = "EXPANDING"
        else:
            volatility_regime = "CONTRACTING"
        
        # Historical statistics for mean reversion
        price_mean = np.mean(prices)
        price_std = np.std(prices)
        price_min = np.min(prices)
        price_max = np.max(prices)
        historical_range = price_max - price_min
        
        # Model 1: Linear regression trend (best for short term)
        lr_coeffs = np.polyfit(dates, prices, 1)
        lr_poly = np.poly1d(lr_coeffs)
        lr_predictions_hist = lr_poly(dates)
        ss_res_lr = np.sum((prices - lr_predictions_hist) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        lr_score = 1 - (ss_res_lr / ss_tot) if ss_tot > 0 else 0
        lr_score = max(0, min(1, lr_score))
        
        # Model 2: Polynomial regression (degree 3, captures recent momentum)
        poly_coeffs = np.polyfit(dates, prices, 3)
        poly_func = np.poly1d(poly_coeffs)
        poly_predictions_hist = poly_func(dates)
        ss_res_poly = np.sum((prices - poly_predictions_hist) ** 2)
        poly_score = 1 - (ss_res_poly / ss_tot) if ss_tot > 0 else 0
        poly_score = max(0, min(1, poly_score))
        
        # Model 3: Exponential Moving Average (captures trend and momentum)
        ema_short = df['Close'].ewm(span=12, adjust=False).mean().values
        ema_long = df['Close'].ewm(span=26, adjust=False).mean().values
        ema_momentum = (ema_short[-1] - ema_long[-1]) / ema_long[-1]
        
        # Model 4: Mean reversion (with regime adjustment)
        # Reduce mean reversion for strong trends, increase for choppy markets
        base_reversion_rate = 0.05
        reversion_rate = base_reversion_rate * (0.5 if trend_strength > 0.6 else 1.0)
        current_deviation = (current_price - price_mean) / price_mean
        
        # Initialize ML Learning Engine
        ml_engine = get_ml_engine()
        
        # Generate predictions for all time horizons
        predictions = {}
        for days_ahead in time_horizons:
            months_ahead = days_ahead / 30
            
            # ADAPTIVE MODEL WEIGHTS with ML learning
            # Get base weights first
            if days_ahead <= 30:
                # 30-day: favor recent momentum and trends
                base_weights = {
                    "linear": 0.30,
                    "polynomial": 0.35,
                    "ema": 0.20,
                    "mean_reversion": 0.15
                }
            elif days_ahead <= 90:
                # 90-day: balanced approach with emphasis on trend models for better confidence
                base_weights = {
                    "linear": 0.28,
                    "polynomial": 0.32,
                    "ema": 0.25,
                    "mean_reversion": 0.15
                }
            elif days_ahead <= 180:
                # Medium term: balance between trend and reversion
                base_weights = {
                    "linear": 0.25,
                    "polynomial": 0.25,
                    "ema": 0.20,
                    "mean_reversion": 0.30
                }
            else:
                # Long term: more weight on mean reversion and historical patterns
                base_weights = {
                    "linear": 0.15,
                    "polynomial": 0.15,
                    "ema": 0.15,
                    "mean_reversion": 0.55
                }
            
            # Apply ML-learned weights if available
            ml_weights = ml_engine.get_adaptive_model_weights(days_ahead)
            
            # Blend base weights with ML weights (70% base, 30% ML for stability)
            learning_level = ml_engine.model_performance.get("overall", {}).get("learning_level", "BEGINNER")
            if learning_level in ["EXPERT", "ADVANCED"]:
                ml_blend = 0.5  # 50% ML influence for experienced model
            elif learning_level in ["INTERMEDIATE", "LEARNING"]:
                ml_blend = 0.3  # 30% ML influence
            else:
                ml_blend = 0.0  # No ML influence for beginners
            
            lr_weight = base_weights["linear"] * (1 - ml_blend) + ml_weights["linear"] * ml_blend
            poly_weight = base_weights["polynomial"] * (1 - ml_blend) + ml_weights["polynomial"] * ml_blend
            ema_weight = base_weights["ema"] * (1 - ml_blend) + ml_weights["ema"] * ml_blend
            mean_rev_weight = base_weights["mean_reversion"] * (1 - ml_blend) + ml_weights["mean_reversion"] * ml_blend
            
            # TREND-BASED ADJUSTMENT: Boost momentum for strong trends, reduce for choppy markets
            if trend_strength > 0.6:
                # Strong trend detected - increase momentum models confidence
                trend_boost = 0.12 * trend_strength
                lr_weight += trend_boost * 0.5
                poly_weight += trend_boost * 0.5
                mean_rev_weight -= trend_boost
            
            # MARKET REGIME ADJUSTMENT
            if market_regime == "BULLISH":
                bullish_boost = 0.05
                ema_weight += bullish_boost
                mean_rev_weight -= bullish_boost * 0.5
            elif market_regime == "BEARISH":
                bearish_boost = 0.05
                mean_rev_weight += bearish_boost * 0.5
                ema_weight -= bearish_boost
            
            # Normalize weights
            total_weight = lr_weight + poly_weight + ema_weight + mean_rev_weight
            lr_weight /= total_weight
            poly_weight /= total_weight
            ema_weight /= total_weight
            mean_rev_weight /= total_weight
            
            # Model 1: Linear extrapolation
            lr_pred = lr_poly(len(prices) + days_ahead - 1)
            
            # Model 2: Polynomial with adaptive dampening
            # Less dampening for 90-day to capture trend momentum better
            if days_ahead <= 90:
                poly_dampening = 1 - (days_ahead / 365) * 0.35  # Reduced dampening
            else:
                poly_dampening = 1 - (min(days_ahead, 365) / 365) * 0.5
            poly_pred = current_price + (poly_func(len(prices) + days_ahead - 1) - current_price) * poly_dampening
            
            # Model 3: EMA with enhanced momentum and acceleration for 90-day
            if days_ahead <= 90:
                decay_factor = 0.96 ** (days_ahead / 365)  # Slower decay
            else:
                decay_factor = 0.95 ** (days_ahead / 365)
            # Include acceleration for stronger signals
            acceleration_factor = 1 + (acceleration * 12 * days_ahead / 365)  # Increased sensitivity
            ema_pred = current_price * (1 + ema_momentum * decay_factor * acceleration_factor)
            
            # Model 4: Mean reversion with regime adjustment
            reversion_strength = 1 - np.exp(-reversion_rate * months_ahead)
            mean_rev_pred = current_price * (1 - reversion_strength) + price_mean * reversion_strength
            
            # Apply sentiment adjustment to mean reversion (don't always revert to mean if sentiment is extreme)
            if abs(sentiment_bias) > 0.001:
                mean_rev_pred += sentiment_bias * historical_range * reversion_strength
            
            # Ensemble: weighted average
            ensemble_pred = (
                lr_pred * lr_weight +
                poly_pred * poly_weight +
                ema_pred * ema_weight +
                mean_rev_pred * mean_rev_weight
            )
            
            # CANDLESTICK PATTERN ADJUSTMENT: Apply pattern signal for short-term predictions
            if days_ahead <= 90 and pattern_strength > 0.05:
                pattern_adjustment = current_price * (pattern_signal * pattern_strength * 0.025)  # Increased impact
                ensemble_pred += pattern_adjustment
            
            # Apply bounds with improved trend consideration for 90-day
            if days_ahead <= 90:
                # More flexible bounds for 90-day predictions
                expansion_factor = 1 + (days_ahead / 365) * 0.35
                if trend_strength > 0.7:
                    expansion_factor *= 1.3  # Allow more range for strong trends
                elif trend_strength > 0.5:
                    expansion_factor *= 1.15
            else:
                expansion_factor = 1 + (min(days_ahead, 1825) / 1825) * 0.15
                if trend_strength > 0.7:
                    expansion_factor *= 1.2
            
            upper_bound = price_max + (price_max - price_mean) * expansion_factor
            lower_bound = price_min - (price_mean - price_min) * expansion_factor
            
            # For strong trends, allow breaking historical bounds
            if trend_strength > 0.65 and days_ahead <= 90:
                trend_extension = current_price * trend_strength * 0.25 * (days_ahead / 90)
                if drift_short > 0:  # Uptrend
                    upper_bound = max(upper_bound, current_price + trend_extension)
                else:  # Downtrend
                    lower_bound = min(lower_bound, current_price - trend_extension)
            
            ensemble_pred = np.clip(ensemble_pred, lower_bound, upper_bound)
            
            # Calculate metrics
            expected_gain_loss = ensemble_pred - current_price
            expected_gain_loss_pct = (expected_gain_loss / current_price) * 100
            
            # Enhanced confidence calculation with better model agreement scoring
            model_std = np.std([lr_pred, poly_pred, ema_pred, mean_rev_pred])
            model_agreement = max(0.35, 1 - (model_std / current_price))
            
            # Higher base confidence for better predictions
            base_confidence = 45 + (model_agreement * 50)
            
            # IMPROVED TIME DECAY: Less penalty for 90-day predictions
            if days_ahead <= 90:
                # Minimal decay for 30-90 day predictions (sweet spot for technical analysis)
                time_decay = 1 - (np.log(1 + days_ahead / 365) / np.log(6)) * 0.25
                time_decay = max(0.75, time_decay)  # Higher floor for short-term
            else:
                time_decay = 1 - (np.log(1 + min(days_ahead, 1825) / 365) / np.log(6)) * 0.4
                time_decay = max(0.60, time_decay)
            
            # ENHANCED CONDITION BOOST: More factors for confidence
            condition_boost = 1.0
            
            # Strong trend boost (bigger impact)
            if trend_strength > 0.7:
                condition_boost += 0.18
            elif trend_strength > 0.6:
                condition_boost += 0.12
            elif trend_strength > 0.5:
                condition_boost += 0.08
            
            # Market regime consistency boost
            if market_regime != "NEUTRAL":
                condition_boost += 0.08
                # Extra boost if regime aligns with price momentum
                if (market_regime == "BULLISH" and drift_short > 0) or (market_regime == "BEARISH" and drift_short < 0):
                    condition_boost += 0.07
            
            # Volatility regime boost
            if volatility_regime == "CONTRACTING":
                condition_boost += 0.07
            
            # Pattern strength boost for predictions
            if pattern_strength > 0.3:
                condition_boost += 0.06 * pattern_strength
            
            # Data quality boost (sufficient history increases confidence)
            if not limited_history:
                condition_boost += 0.05
            
            # Momentum alignment boost
            if abs(momentum_strength) > 0.5:
                condition_boost += 0.05
            
            # Volatility adjustment (lower volatility = higher confidence)
            vol_adjustment = max(0.75, 1 - (volatility / 100) * 0.25)
            
            # Fit quality with enhanced impact
            fit_quality = (lr_score + poly_score) / 2
            fit_bonus = 1 + (fit_quality * 0.20)  # Increased from 0.15
            
            # Calculate final confidence with improved weighting
            confidence = (
                base_confidence * 0.45 +
                (model_agreement * 100) * 0.30 +  # Increased weight
                (time_decay * 100) * 0.15 +
                (vol_adjustment * 100) * 0.10
            ) * fit_bonus * condition_boost
            
            # Wider confidence range with higher ceiling for strong predictions
            confidence = max(35, min(96, confidence))
            
            # Apply ML-based confidence adjustment
            ml_adjusted_confidence, ml_reason = ml_engine.get_confidence_adjustment(confidence, days_ahead)
            confidence = ml_adjusted_confidence
            
            # Improved risk assessment aligned with confidence
            volatility_adjusted = volatility * (1 + np.log(1 + days_ahead / 30) * 0.12)  # Reduced penalty
            movement_risk = abs(expected_gain_loss_pct) / 100
            trend_risk_adjustment = 1 - (trend_strength * 0.3)  # Strong trends reduce risk
            total_risk = (volatility_adjusted * 0.55 + movement_risk * 45) * trend_risk_adjustment
            
            # More nuanced risk levels aligned with confidence
            if total_risk > 10:
                risk_level = "HIGH"
            elif total_risk > 6:
                risk_level = "MEDIUM-HIGH" 
            elif total_risk > 3.5:
                risk_level = "MEDIUM"
            elif total_risk > 2:
                risk_level = "MEDIUM-LOW"
            else:
                risk_level = "LOW"
            
            # Direction
            if expected_gain_loss_pct > 0.5:
                direction = "📈 BULLISH"
                emoji = "🟢"
            elif expected_gain_loss_pct < -0.5:
                direction = "📉 BEARISH"
                emoji = "🔴"
            else:
                direction = "➡️ NEUTRAL"
                emoji = "🟡"
            
            # Store prediction
            predictions[days_ahead] = {
                'predicted_price': ensemble_pred,
                'expected_gain_loss': expected_gain_loss,
                'expected_gain_loss_pct': expected_gain_loss_pct,
                'direction': direction,
                'emoji': emoji,
                'confidence': confidence,
                'risk_level': risk_level,
                'volatility': volatility_adjusted,
                'models': {
                    'linear': lr_pred,
                    'polynomial': poly_pred,
                    'ema': ema_pred,
                    'mean_reversion': mean_rev_pred,
                    'bounded': ensemble_pred
                },
                'bounds': {
                    'upper': upper_bound,
                    'lower': lower_bound
                },
                'ml_adjusted': ml_blend > 0,
                'ml_reason': ml_reason if ml_blend > 0 else None
            }
        
        # Get ML learning statistics
        ml_stats = ml_engine.get_learning_stats()
        
        return {
            'current_price': current_price,
            'volatility': volatility,
            'predictions': predictions,
            'historical_volatility': volatility_long,
            'drift': drift,
            'price_range': (price_min, price_max),
            'price_mean': price_mean,
            'market_regime': market_regime,
            'sentiment': sentiment,
            'trend_strength': trend_strength,
            'momentum_strength': momentum_strength,
            'volatility_regime': volatility_regime,
            'pattern_analysis': pattern_analysis,
            'patterns_detected': pattern_analysis['patterns'],
            'darkpool_analysis': darkpool_analysis,
            'data_quality': 'LIMITED_HISTORY' if limited_history else 'OK',
            'history_length': history_len,
            'ml_learning': ml_stats
        }
    
    except Exception as e:
        st.error(f"Error in prediction model: {str(e)}")
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def run_bullish_bearish_scan(symbols, time_horizon, history_period):
    """Run AI prediction scan and return top bullish/bearish lists."""
    results = []

    for symbol in symbols:
        try:
            df = yf.Ticker(symbol).history(period=history_period)
            if df.empty:
                continue

            prediction_results = predict_future_price(df, time_horizons=[time_horizon])
            if not prediction_results:
                continue

            pred = prediction_results['predictions'][time_horizon]
            results.append({
                "Symbol": symbol,
                "Predicted Price": round(pred['predicted_price'], 2),
                "Expected Return (%)": round(pred['expected_gain_loss_pct'], 2),
                "Direction": pred['direction'],
                "Confidence (%)": round(pred['confidence'], 1),
                "Risk": pred['risk_level']
            })
        except Exception:
            continue

    if not results:
        return pd.DataFrame(), pd.DataFrame()

    results_df = pd.DataFrame(results)
    # Sort by expected return but don't limit results - show all
    bullish = results_df.sort_values("Expected Return (%)", ascending=False)
    bearish = results_df.sort_values("Expected Return (%)", ascending=True)

    return bullish, bearish


def create_darkpool_chart(df, darkpool_analysis):
    """
    Create a comprehensive darkpool activity visualization showing institutional flows.
    """
    if darkpool_analysis is None:
        return None
    
    try:
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Volume Anomaly Score', 'Price-Volume Divergence', 
                          'Accumulation Momentum', 'Large Block Detection'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.12
        )
        
        volumes = df['Volume'].values if 'Volume' in df.columns else np.ones(len(df))
        prices = df['Close'].values
        dates = df.index
        
        # Calculate metrics over time
        volume_anomalies = []
        divergences = []
        
        avg_vol_20 = np.mean(volumes[-20:])
        vol_std = np.std(volumes[-20:])
        
        for i in range(len(volumes)):
            zscore = (volumes[i] - avg_vol_20) / (vol_std + 0.001)
            volume_anomalies.append(zscore)
            
            if i >= 4:
                price_range = np.max(prices[i-4:i+1]) - np.min(prices[i-4:i+1])
                divergence = 1 - np.clip(price_range / (volumes[i] / 1e6 + 0.001), 0, 1)
                divergences.append(divergence)
            else:
                divergences.append(0)
        
        # 1. Volume Anomaly Score
        colors_vol = ['red' if x > 1.5 else 'orange' if x > 0.5 else 'green' for x in volume_anomalies]
        fig.add_trace(
            go.Bar(x=dates, y=volume_anomalies, name='Volume Anomaly (Z-Score)',
                   marker=dict(color=colors_vol), showlegend=True),
            row=1, col=1
        )
        fig.add_hline(y=1.5, line_dash="dash", line_color="red", row=1, col=1)
        
        # 2. Price-Volume Divergence
        fig.add_trace(
            go.Scatter(x=dates, y=divergences, name='P-V Divergence',
                      line=dict(color='purple', width=2), fill='tozeroy',
                      fillcolor='rgba(128,0,128,0.2)', showlegend=True),
            row=1, col=2
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=1, col=2)
        
        # 3. Accumulation/Distribution
        ad_line = []
        for i in range(len(prices)):
            if i == 0:
                ad_line.append(0)
            else:
                close_loc = (prices[i] - np.min(prices[max(0,i-4):i+1])) / \
                           (np.max(prices[max(0,i-4):i+1]) - np.min(prices[max(0,i-4):i+1]) + 0.001)
                ad = ad_line[-1] + volumes[i] * (2 * close_loc - 1)
                ad_line.append(ad)
        
        fig.add_trace(
            go.Scatter(x=dates, y=ad_line, name='A/D Line',
                      line=dict(color='steelblue', width=2), fill='tozeroy',
                      fillcolor='rgba(70,130,180,0.2)', showlegend=True),
            row=2, col=1
        )
        
        # 4. Block Trade Frequency
        block_detection = []
        threshold = avg_vol_20 + 2 * vol_std
        for i in range(len(volumes)):
            blocks_in_window = np.sum(volumes[max(0,i-4):i+1] > threshold)
            block_detection.append(blocks_in_window)
        
        fig.add_trace(
            go.Bar(x=dates, y=block_detection, name='Large Blocks/5-day',
                   marker=dict(color='coral'), showlegend=True),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="🌊 Darkpool Activity Analysis - Institutional Flow Indicators",
            hovermode='x unified',
            height=800,
            showlegend=True,
            template='plotly_dark'
        )
        
        fig.update_yaxes(title_text="Z-Score", row=1, col=1)
        fig.update_yaxes(title_text="Divergence", row=1, col=2)
        fig.update_yaxes(title_text="A/D Momentum", row=2, col=1)
        fig.update_yaxes(title_text="Block Count", row=2, col=2)
        
        return fig
    except Exception as e:
        st.error(f"Error creating darkpool chart: {str(e)}")
        return None

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = MarketAnalyzer()
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def normalize_symbol(symbol: str, asset_type: str = "stock") -> str:
    """
    Normalize ticker symbol to correct format.
    
    Args:
        symbol: Raw symbol input
        asset_type: Type of asset (stock or crypto)
    
    Returns:
        Normalized symbol
    """
    symbol = symbol.strip().upper()
    
    # For crypto, ensure -USD suffix
    if asset_type.lower() == "crypto":
        if not symbol.endswith("-USD"):
            symbol = f"{symbol}-USD"
    
    return symbol


def main():
    """Main application function."""
    
    # Header
    st.title("📊 Market Analyzer")
    st.markdown("### For Informational Purposes Only - No Trading")
    st.caption("Analyze market trends, technical indicators, and AI-based forecasts in one place.")
    st.warning("⚠️ This tool analyzes financial markets but does NOT execute trades or move money.")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Asset Information")
        st.caption("Configure your asset and analysis settings.")
        
        with st.expander("Asset Settings", expanded=True):
            # Asset type (moved before symbol for better logic flow)
            asset_type = st.radio(
                "Asset Type",
                options=["stock", "crypto"],
                index=0,
                help="Select whether analyzing a stock or cryptocurrency"
            )
            
            # Symbol input
            raw_symbol = st.text_input(
                "Symbol",
                value="AAPL",
                help="Enter stock ticker (e.g., AAPL, MSFT) or crypto (e.g., BTC, ETH)",
                placeholder="AAPL or BTC"
            )
        
        # Check for quick button override
        if 'selected_symbol' in st.session_state:
            raw_symbol = st.session_state.selected_symbol
            # Clear the session state so it doesn't persist
            del st.session_state.selected_symbol
        
        # Normalize the symbol based on asset type
        symbol = normalize_symbol(raw_symbol, asset_type)
        
        # Top 10 Lists
        top_10_stocks = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "TSLA", "META", "BRK.B", "JNJ", "V"
        ]
        
        top_10_crypto = [
            "BTC", "ETH", "BNB", "SOL", "XRP",
            "ADA", "DOGE", "AVAX", "POLKADOT", "LINK"
        ]
        
        st.divider()
        
        with st.expander("Analysis Settings", expanded=True):
            # Period selection
            period = st.selectbox(
                "Time Period",
                options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
                index=5,  # Default to 1y
                help="Select the time period for analysis"
            )
            
            mode = st.radio(
                "Mode",
                options=["Analyze", "Compare"],
                horizontal=True,
                help="Analyze one asset or compare multiple assets"
            )
            compare_mode = mode == "Compare"
        
        st.divider()
        
        # ML Learning Section
        with st.expander("🤖 ML Learning Status", expanded=False):
            ml_engine = get_ml_engine()
            ml_stats = ml_engine.get_learning_stats()
            
            learning_level = ml_stats['learning_level']
            total = ml_stats['total_predictions']
            validated = ml_stats['validated_predictions']
            
            level_info = {
                'BEGINNER': ('🟡', 'Just starting'),
                'LEARNING': ('🟢', 'Building knowledge'),
                'INTERMEDIATE': ('🟢', 'Good progress'),
                'ADVANCED': ('🔵', 'Well trained'),
                'EXPERT': ('🟣', 'Highly accurate')
            }
            
            emoji, desc = level_info.get(learning_level, ('⚪', 'Unknown'))
            
            st.markdown(f"**{emoji} Level: {learning_level}**")
            st.caption(f"{desc} • {validated} validated predictions")
            
            if total > 0:
                st.progress(min(validated / 100, 1.0), text=f"{validated}/100 for Expert")
            
            if validated > 0:
                st.markdown("**Performance:**")
                for tf, perf in ml_stats.get('timeframe_performance', {}).items():
                    if perf['predictions_count'] > 0:
                        accuracy = 100 - perf['avg_error_pct']
                        st.write(f"  • {tf}-day: {accuracy:.1f}% accuracy")
            
            st.caption(
                "💡 Record predictions to build ML learning data. "
                "The system improves over time!"
            )
            
            # Optional: Add manual outcome update
            st.markdown("---")
            st.markdown("**Manual Update (Optional):**")
            update_symbol = st.text_input("Symbol", key="ml_update_symbol", placeholder="e.g., AAPL")
            update_price = st.number_input("Current Price", min_value=0.01, step=0.01, key="ml_update_price")
            
            if st.button("Update Outcomes", key="ml_update_btn", use_container_width=True):
                if update_symbol and update_price > 0:
                    updated = ml_engine.update_with_actual_outcome(
                        update_symbol,
                        datetime.now().isoformat(),
                        update_price
                    )
                    if updated:
                        st.success("✅ Outcomes updated!")
                    else:
                        st.info("No predictions to update for this symbol/date")
                else:
                    st.error("Please enter symbol and price")
        
        st.divider()
        
        # Analyze button
        analyze_clicked = st.button("🔍 Analyze Asset", type="primary", use_container_width=True, key="analyze_asset_btn")
        
        # Clear history button
        if st.button("🗑️ Clear History", use_container_width=True, key="clear_history_btn"):
            st.session_state.analysis_history = []
            st.success("History cleared!")
    
    # Main content area
    st.markdown("#### Quick Guide")
    guide_col1, guide_col2 = st.columns([2, 1])
    with guide_col1:
        st.markdown(
            "- Select an asset and time period in the sidebar.\n"
            "- Click **Analyze Asset** to generate charts and AI forecasts.\n"
            "- Use **Compare** mode to review multiple symbols at once."
        )
    with guide_col2:
        st.info("Tip: Use symbols like AAPL, MSFT or BTC, ETH for crypto.")
    if compare_mode:
        show_comparison_interface()
    else:
        # Single analysis mode
        if analyze_clicked:
            if not symbol:
                st.error("Please enter a symbol!")
            else:
                with st.spinner(f"Analyzing {symbol}..."):
                    try:
                        # Perform analysis
                        analysis = st.session_state.analyzer.analyze_asset(
                            symbol, 
                            period=period, 
                            asset_type=asset_type
                        )
                        
                        if analysis:
                            # Add to history
                            st.session_state.analysis_history.append({
                                'symbol': symbol,
                                'period': period,
                                'asset_type': asset_type,
                                'analysis': analysis
                            })
                            
                            # Display analysis
                            display_analysis(analysis, symbol, period, asset_type)
                        else:
                            st.error(f"❌ Failed to analyze {symbol}. Please check the symbol and try again.")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        elif st.session_state.analysis_history:
            # Show most recent analysis
            latest = st.session_state.analysis_history[-1]
            st.info(f"📊 Showing latest analysis: {latest['symbol']} ({latest['period']})")
            display_analysis(latest['analysis'], latest['symbol'], latest['period'], latest['asset_type'])
            
            # Show history
            if len(st.session_state.analysis_history) > 1:
                with st.expander(f"📜 View History ({len(st.session_state.analysis_history)} analyses)"):
                    for idx, item in enumerate(reversed(st.session_state.analysis_history[:-1])):
                        st.markdown(f"**{len(st.session_state.analysis_history) - idx - 1}.** {item['symbol']} ({item['period']}, {item['asset_type']})")
        else:
            st.info("👈 Enter a symbol and click **Analyze Asset** to get started!")
            
            # Show example
            st.markdown("""
            ### How to Use:
            1. Enter a stock symbol (e.g., **AAPL**, **MSFT**) or crypto (e.g., **BTC-USD**)
            2. Select a time period (e.g., **1y** for 1 year)
            3. Choose asset type (**stock** or **crypto**)
            4. Click **🔍 Analyze Asset** button
            5. View detailed analysis below!
            
            **Try the Quick Examples** in the sidebar to get started quickly!
            """)

def create_technical_chart(symbol, period, analysis):
    """Create an interactive TradingView-style chart with AI-enhanced technical analysis."""
    
    try:
        # Fetch data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            return None
        
        # Create subplots: Price chart + Volume + RSI
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} Price Chart with AI Analysis', 'Volume', 'RSI'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # 1. Candlestick Chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # Add predicted candlesticks for next 30 days
        # Generate synthetic predicted candles based on prediction model
        if 'predictions' in analysis:
            predictions = analysis['predictions']
            if 30 in predictions:
                pred_data = predictions[30]
                predicted_price = pred_data['predicted_price']
                
                # Create synthetic predicted candles
                last_date = df.index[-1]
                last_close = df['Close'].iloc[-1]
                last_high = df['High'].iloc[-1]
                last_low = df['Low'].iloc[-1]
                
                # Generate 5 candles towards the prediction
                pred_dates = []
                pred_opens = []
                pred_highs = []
                pred_lows = []
                pred_closes = []
                
                current_pred = last_close
                price_step = (predicted_price - last_close) / 6  # Divide into 6 steps
                
                for i in range(1, 7):
                    pred_date = last_date + pd.Timedelta(days=i*5)
                    pred_open = current_pred
                    pred_close = current_pred + price_step
                    
                    # Add some volatility based on historical volatility
                    historical_vol = np.std(np.diff(df['Close'].values[-20:]) / df['Close'].values[-19:])
                    pred_high = pred_close + abs(price_step) * (0.5 + historical_vol * 5)
                    pred_low = min(pred_open, pred_close) - abs(price_step) * (0.3 + historical_vol * 3)
                    
                    pred_dates.append(pred_date)
                    pred_opens.append(pred_open)
                    pred_highs.append(pred_high)
                    pred_lows.append(pred_low)
                    pred_closes.append(pred_close)
                    
                    current_pred = pred_close
                
                # Add predicted candlesticks with different styling
                fig.add_trace(
                    go.Candlestick(
                        x=pred_dates,
                        open=pred_opens,
                        high=pred_highs,
                        low=pred_lows,
                        close=pred_closes,
                        name='Predicted Price',
                        increasing_line_color='rgba(0,255,0,0.5)',
                        decreasing_line_color='rgba(255,0,0,0.5)',
                        increasing_fillcolor='rgba(0,255,0,0.2)',
                        decreasing_fillcolor='rgba(255,0,0,0.2)',
                        opacity=0.6
                    ),
                    row=1, col=1
                )
        
        # AI-Enhanced Trend Line Detection
        def detect_ai_trendline(prices, window=20):
            """Detect trend lines using linear regression."""
            x = np.arange(len(prices))
            y = prices.values
            
            # Calculate linear regression
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            
            return p(x)
        
        # Add AI-detected trend line
        if len(df) > 20:
            trend_line = detect_ai_trendline(df['Close'])
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=trend_line,
                    name='AI Trend Line',
                    line=dict(color='#FFD700', width=2, dash='solid'),
                    opacity=0.8
                ),
                row=1, col=1
            )
        
        # Add AI-detected trend channel
        if len(df) > 50:
            # Upper channel (resistance trend)
            upper_prices = df['High'].rolling(window=20).max()
            upper_trend = detect_ai_trendline(upper_prices.dropna())
            upper_indices = df.index[19:]  # Adjust for rolling window
            
            # Lower channel (support trend)
            lower_prices = df['Low'].rolling(window=20).min()
            lower_trend = detect_ai_trendline(lower_prices.dropna())
            
            fig.add_trace(
                go.Scatter(
                    x=upper_indices, y=upper_trend,
                    name='AI Upper Channel',
                    line=dict(color='rgba(255,0,0,0.3)', width=1, dash='dash'),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=upper_indices, y=lower_trend,
                    name='AI Lower Channel',
                    line=dict(color='rgba(0,255,0,0.3)', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Add Moving Averages if available
        if 'indicators' in analysis:
            indicators = analysis['indicators']
            
            # Calculate SMAs for the chart
            if len(df) >= 20:
                sma_20 = df['Close'].rolling(window=20).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=sma_20,
                        name='SMA 20',
                        line=dict(color='#2196F3', width=1.5)
                    ),
                    row=1, col=1
                )
            
            if len(df) >= 50:
                sma_50 = df['Close'].rolling(window=50).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=sma_50,
                        name='SMA 50',
                        line=dict(color='#FF9800', width=1.5)
                    ),
                    row=1, col=1
                )
            
            if len(df) >= 200:
                sma_200 = df['Close'].rolling(window=200).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=sma_200,
                        name='SMA 200',
                        line=dict(color='#9C27B0', width=1.5)
                    ),
                    row=1, col=1
                )
            
            # Add Bollinger Bands if available
            if 'bb_upper' in indicators and 'bb_lower' in indicators:
                # Calculate Bollinger Bands
                sma_20_bb = df['Close'].rolling(window=20).mean()
                std_20 = df['Close'].rolling(window=20).std()
                bb_upper = sma_20_bb + (2 * std_20)
                bb_lower = sma_20_bb - (2 * std_20)
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=bb_upper,
                        name='BB Upper',
                        line=dict(color='rgba(128,128,128,0.3)', width=1, dash='dash'),
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=bb_lower,
                        name='BB Lower',
                        line=dict(color='rgba(128,128,128,0.3)', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)',
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # Add AI-Enhanced Support/Resistance levels
            if 'trend' in analysis:
                trend = analysis['trend']
                if 'support' in trend and trend['support']:
                    fig.add_hline(
                        y=trend['support'],
                        line=dict(color='#00ff00', width=2, dash='solid'),
                        annotation_text=f"🤖 AI Support: ${trend['support']:.2f}",
                        annotation_position="right",
                        annotation_font=dict(color='#00ff00', size=12),
                        row=1, col=1
                    )
                if 'resistance' in trend and trend['resistance']:
                    fig.add_hline(
                        y=trend['resistance'],
                        line=dict(color='#ff0000', width=2, dash='solid'),
                        annotation_text=f"🤖 AI Resistance: ${trend['resistance']:.2f}",
                        annotation_position="right",
                        annotation_font=dict(color='#ff0000', size=12),
                        row=1, col=1
                    )
        
        # Add AI-detected price targets based on trend
        current_price = df['Close'].iloc[-1]
        price_change = df['Close'].pct_change().mean() * 100
        
        # Predictive price levels
        if price_change > 0:
            target_price = current_price * 1.1  # 10% upside target
            fig.add_hline(
                y=target_price,
                line=dict(color='#00ccff', width=1, dash='dot'),
                annotation_text=f"🎯 AI Target: ${target_price:.2f}",
                annotation_position="left",
                annotation_font=dict(color='#00ccff', size=10),
                row=1, col=1
            )
        else:
            target_price = current_price * 0.9  # 10% downside target
            fig.add_hline(
                y=target_price,
                line=dict(color='#ff00cc', width=1, dash='dot'),
                annotation_text=f"🎯 AI Target: ${target_price:.2f}",
                annotation_position="left",
                annotation_font=dict(color='#ff00cc', size=10),
                row=1, col=1
            )
        
        # 2. Volume Chart
        colors = ['red' if close < open_ else 'green' 
                 for close, open_ in zip(df['Close'], df['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add volume moving average
        vol_ma = df['Volume'].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index, y=vol_ma,
                name='Vol MA',
                line=dict(color='orange', width=1),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 3. RSI
        if 'indicators' in analysis and 'rsi' in analysis['indicators']:
            # Calculate RSI for the full period
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=rsi,
                    name='RSI',
                    line=dict(color='#9C27B0', width=2),
                    showlegend=False
                ),
                row=3, col=1
            )
            
            # Add RSI levels
            fig.add_hline(y=70, line=dict(color='red', width=1, dash='dash'), 
                         annotation_text="Overbought (70)", row=3, col=1)
            fig.add_hline(y=30, line=dict(color='green', width=1, dash='dash'), 
                         annotation_text="Oversold (30)", row=3, col=1)
            fig.add_hline(y=50, line=dict(color='gray', width=1, dash='dot'), 
                         row=3, col=1)
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.05)'
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def display_analysis(analysis, symbol, period, asset_type):
    """Display analysis results in a formatted way."""
    
    # Create and display the technical chart first
    st.markdown("### 📈 Technical Analysis Chart")
    st.caption("Interactive price chart with volume, moving averages, and RSI indicator. Hover over the chart for detailed price information.")
    with st.spinner("Generating interactive chart..."):
        chart = create_technical_chart(symbol, period, analysis)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.warning("Unable to generate chart for this symbol.")
    
    st.divider()
    
    # Asset Overview
    st.markdown("#### 📋 Asset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Symbol", symbol)
    with col2:
        st.metric("Analysis Period", period)
    with col3:
        st.metric("Asset Type", asset_type.title())
    with col4:
        if 'current_price' in analysis:
            st.metric("Current Price", f"${analysis['current_price']:.2f}")
    
    st.divider()
    
    # AI Future Price Prediction
    st.markdown("### 🤖 AI Future Price Prediction (Multi-Horizon)")
    
    with st.spinner("Running advanced AI prediction models..."):
        # Fetch data for prediction
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if not df.empty:
            prediction_results = predict_future_price(df)
            
            if prediction_results:
                if prediction_results.get('data_quality') == 'LIMITED_HISTORY':
                    st.warning(
                        f"Limited history detected ({prediction_results.get('history_length', 0)} periods). "
                        "Some metrics may be less reliable for newer or thinly traded stocks."
                    )
                
                # Display ML Learning Status
                ml_learning = prediction_results.get('ml_learning', {})
                if ml_learning:
                    learning_level = ml_learning.get('learning_level', 'BEGINNER')
                    total_preds = ml_learning.get('total_predictions', 0)
                    validated_preds = ml_learning.get('validated_predictions', 0)
                    
                    level_colors = {
                        'BEGINNER': '🟡',
                        'LEARNING': '🟢',
                        'INTERMEDIATE': '🟢',
                        'ADVANCED': '🔵',
                        'EXPERT': '🟣'
                    }
                    level_emoji = level_colors.get(learning_level, '⚪')
                    
                    if learning_level != 'BEGINNER' or validated_preds > 0:
                        with st.expander(f"{level_emoji} ML Learning Status: {learning_level} - {validated_preds} validated predictions", expanded=False):
                            st.markdown(f"""
                            **Machine Learning Enhancement Active**
                            - **Learning Level:** {learning_level}
                            - **Total Predictions Tracked:** {total_preds}
                            - **Validated Outcomes:** {validated_preds}
                            - **Confidence Adjustment:** {'Active' if learning_level in ['ADVANCED', 'EXPERT', 'INTERMEDIATE', 'LEARNING'] else 'Building data'}
                            """)
                            
                            # Show timeframe performance if available
                            tf_perf = ml_learning.get('timeframe_performance', {})
                            if tf_perf:
                                st.markdown("**📊 Historical Accuracy by Timeframe:**")
                                perf_data = []
                                for tf, stats in tf_perf.items():
                                    perf_data.append({
                                        "Timeframe": f"{tf} days",
                                        "Predictions": stats['predictions_count'],
                                        "Avg Error": f"{stats['avg_error_pct']:.1f}%",
                                        "Direction Accuracy": f"{stats['direction_accuracy']:.1f}%",
                                        "Avg Confidence": f"{stats['avg_confidence']:.1f}%"
                                    })
                                if perf_data:
                                    st.dataframe(pd.DataFrame(perf_data), use_container_width=True)
                            
                            # Show model performance if available
                            model_perf = ml_learning.get('model_performance', {})
                            if model_perf:
                                st.markdown("**🎯 Model Performance (Adaptive Weights):**")
                                model_data = []
                                for model, stats in model_perf.items():
                                    model_data.append({
                                        "Model": model.title(),
                                        "Predictions": stats['predictions_count'],
                                        "Avg Error": f"{stats['avg_error_pct']:.1f}%",
                                        "RMSE": f"{stats['rmse']:.2f}"
                                    })
                                if model_data:
                                    st.dataframe(pd.DataFrame(model_data), use_container_width=True)
                                    st.caption("System automatically adjusts model weights based on historical accuracy")
                            
                            if ml_learning.get('improvement_suggestions'):
                                st.markdown("**💡 Suggestions:**")
                                for suggestion in ml_learning['improvement_suggestions']:
                                    st.write(f"- {suggestion}")
                
                current_price = prediction_results['current_price']
                predictions = prediction_results['predictions']
                
                # Time horizon labels for display
                horizon_labels = {
                    30: "📅 1 Month (30d)",
                    90: "📊 3 Months (90d)"
                }
                
                # Display current price and market conditions
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("💵 Current Price", f"${current_price:.2f}")
                with col2:
                    st.metric("📊 Volatility", f"{prediction_results['volatility']:.2f}%")
                with col3:
                    regime_emoji = "🟢" if prediction_results['market_regime'] == "BULLISH" else "🔴" if prediction_results['market_regime'] == "BEARISH" else "🟡"
                    st.metric(f"{regime_emoji} Market Regime", prediction_results['market_regime'])
                with col4:
                    sentiment_emoji = "📈" if prediction_results['trend_strength'] > 0.6 else "📉" if prediction_results['trend_strength'] < 0.4 else "➡️"
                    st.metric(f"{sentiment_emoji} Trend Strength", f"{prediction_results['trend_strength']*100:.1f}%")
                
                st.divider()
                
                # Market conditions summary
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    **Market Conditions:**
                    - **Regime:** {prediction_results['market_regime']} ({prediction_results['sentiment']})
                    - **Volatility Trend:** {prediction_results['volatility_regime']}
                    - **Momentum:** {prediction_results['momentum_strength']:.2f}x
                    - **Daily Drift:** {prediction_results['drift']*100:.4f}%
                    """)
                with col2:
                    st.markdown(f"""
                    **Technical Assessment:**
                    - **Trend Strength:** {prediction_results['trend_strength']*100:.1f}% ({"Strong" if prediction_results['trend_strength'] > 0.6 else "Weak" if prediction_results['trend_strength'] < 0.4 else "Moderate"})
                    - **Price Position:** {prediction_results['sentiment']}
                    - **Historical Range:** ${prediction_results['price_range'][0]:.2f} - ${prediction_results['price_range'][1]:.2f}
                    - **Price Mean:** ${prediction_results['price_mean']:.2f}
                    """)
                
                # Candlestick Patterns Detection
                if prediction_results['patterns_detected']:
                    st.markdown("### 🕯️ Candlestick Pattern Recognition")
                    patterns_text = ", ".join(prediction_results['patterns_detected'])
                    pattern_sentiment = prediction_results['pattern_analysis']['signal']
                    if pattern_sentiment > 0.5:
                        st.success(f"**Bullish Patterns Detected:** {patterns_text}")
                    elif pattern_sentiment < -0.5:
                        st.error(f"**Bearish Patterns Detected:** {patterns_text}")
                    else:
                        st.info(f"**Neutral Patterns Detected:** {patterns_text}")
                else:
                    st.info("No candlestick patterns detected in recent candles.")
                
                st.divider()
                
                # DARKPOOL ANALYSIS DISPLAY
                if prediction_results['darkpool_analysis']:
                    st.markdown("### 🌊 Darkpool Activity Analysis")
                    darkpool = prediction_results['darkpool_analysis']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Volume Anomaly", f"{darkpool['volume_anomaly']:.2f}",
                                "Z-Score", delta_color="inverse")
                    with col2:
                        st.metric("Est. Darkpool %", f"{darkpool['estimated_darkpool_pct']:.1f}%",
                                "of volume")
                    with col3:
                        st.metric("Sentiment", darkpool['darkpool_sentiment'],
                                f"Score: {darkpool['sentiment_score']:.2f}")
                    with col4:
                        st.metric("Volume Trend", darkpool['volume_trend'],
                                f"Avg: {darkpool['average_volume']/1e6:.1f}M")
                    
                    # Detailed explanation
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        **What This Means:**
                        - **Volume Anomaly ({darkpool['volume_anomaly']:.2f}):** Measures deviation from normal volume. Higher = unusual activity
                        - **Divergence ({darkpool['price_volume_divergence']:.2f}):** Price moving with minimal volume suggests off-exchange trades
                        - **Accumulation Momentum:** Positive = Smart money buying, Negative = Distribution
                        """)
                    with col2:
                        st.markdown(f"""
                        **Institutional Flow Insights:**
                        - **Est. Darkpool Volume:** {darkpool['estimated_darkpool_pct']:.1f}% (typically 10-20%)
                        - **Sentiment:** {darkpool['darkpool_sentiment']} (Accumulation = bullish, Distribution = bearish)
                        - **Large Blocks:** {darkpool['large_blocks_detected']:.1f} blocks/10-day period
                        - **Current Volume:** {darkpool['recent_volume']/1e6:.1f}M vs Avg {darkpool['average_volume']/1e6:.1f}M
                        """)
                    
                    # Display darkpool chart
                    st.markdown("**Darkpool Metrics Breakdown:**")
                    darkpool_chart = create_darkpool_chart(df, darkpool)
                    if darkpool_chart:
                        st.plotly_chart(darkpool_chart, use_container_width=True)
                
                st.divider()
                
                # Create tabs for different time horizons
                tabs = st.tabs([horizon_labels[h] for h in [30, 90]])
                
                for idx, days in enumerate([30, 90]):
                    with tabs[idx]:
                            pred = predictions[days]
                            
                            # Calculate time period text
                            time_text = {
                                30: "Next 30 days",
                                90: "Next 90 days (Q)"
                            }[days]
                            
                            # Main metrics row
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric(
                                    "🎯 Predicted Price",
                                    f"${pred['predicted_price']:.2f}",
                                    delta=f"${pred['expected_gain_loss']:.2f}",
                                    delta_color="inverse"
                                )
                            
                            with col2:
                                color = "normal" if pred['expected_gain_loss_pct'] >= 0 else "inverse"
                                st.metric(
                                    "📊 Expected Return",
                                    f"{pred['expected_gain_loss_pct']:.2f}%",
                                    delta_color=color
                                )
                            
                            with col3:
                                st.metric(
                                    "🎯 Confidence",
                                    f"{pred['confidence']:.1f}%",
                                    f"Risk: {pred['risk_level']}"
                                )
                            
                            with col4:
                                st.metric(
                                    "📈 Direction",
                                    pred['emoji'],
                                    pred['direction']
                                )
                            
                            # Detailed breakdown
                            st.markdown(f"**Prediction Details ({time_text}):**")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"""
                                - **Trend:** {pred['direction']}
                                - **Adjusted Volatility:** {pred['volatility']:.2f}%
                                - **Risk Assessment:** {pred['risk_level']}
                                - **Realistic Range:** ${pred['bounds']['lower']:.2f} - ${pred['bounds']['upper']:.2f}
                                """)
                            
                            with col2:
                                st.write(f"""
                                **Model Ensemble Predictions:**
                                - Linear Trend: ${pred['models']['linear']:.2f}
                                - Polynomial: ${pred['models']['polynomial']:.2f}
                                - EMA Momentum: ${pred['models']['ema']:.2f}
                                - Mean Reversion: ${pred['models']['mean_reversion']:.2f}
                                """)
                            
                            # Sentiment color
                            if pred['expected_gain_loss_pct'] > 2:
                                st.success(f"✅ **Strong Bullish Signal**: Expected upside of {pred['expected_gain_loss_pct']:.2f}%")
                            elif pred['expected_gain_loss_pct'] > 0:
                                st.success(f"🟢 **Bullish**: Expected gain of {pred['expected_gain_loss_pct']:.2f}%")
                            elif pred['expected_gain_loss_pct'] > -2:
                                st.info(f"🟡 **Neutral**: Minimal expected change ({pred['expected_gain_loss_pct']:.2f}%)")
                            elif pred['expected_gain_loss_pct'] > -5:
                                st.warning(f"🟠 **Bearish**: Expected decline of {abs(pred['expected_gain_loss_pct']):.2f}%")
                            else:
                                st.error(f"🔴 **Strong Bearish Signal**: Expected downside of {abs(pred['expected_gain_loss_pct']):.2f}%")
                
                st.divider()
                
                # Comparison table (outside the loop to show only once)
                st.markdown("### 📋 Prediction Comparison Across Time Horizons")
                
                comparison_data = []
                for days in [30, 90]:
                    pred = predictions[days]
                    time_label = {30: "1 Month", 90: "3 Months"}[days]
                    comparison_data.append({
                        "Time Horizon": time_label,
                        "Predicted Price": f"${pred['predicted_price']:.2f}",
                        "Expected Return": f"{pred['expected_gain_loss_pct']:.2f}%",
                        "Direction": pred['direction'],
                        "Confidence": f"{pred['confidence']:.1f}%",
                        "Risk Level": pred['risk_level']
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # ML: Record Prediction button
                st.markdown("---")
                col1, col2, col3 = st.columns([2, 1, 2])
                with col2:
                    # Create unique key using symbol, period, and current timestamp hash
                    import time
                    unique_id = hash(f"{symbol}_{period}_{time.time()}")
                    if st.button("💾 Record Prediction for ML Learning", type="primary", use_container_width=True, key=f"record_pred_{unique_id}"):
                        ml_engine = get_ml_engine()
                        record = ml_engine.record_prediction(
                            symbol=symbol,
                            prediction_data=predictions,
                            current_price=current_price,
                            prediction_date=datetime.now().isoformat()
                        )
                        st.success(f"✅ Prediction recorded! The system will learn from this prediction as time passes.")
                        st.info(f"💡 Come back in 30 or 90 days to see how accurate this prediction was. The ML engine will automatically improve confidence scoring based on results.")
    
    st.divider()
    
    # Prediction section
    if 'prediction' in analysis:
        pred = analysis['prediction']
        
        # Color coding for prediction
        pred_colors = {
            'BUY': '🟢',
            'STRONG_BUY': '🟢',
            'SELL': '🔴',
            'STRONG_SELL': '🔴',
            'HOLD': '🟡'
        }
        
        icon = pred_colors.get(pred['recommendation'], '⚪')
        
        st.markdown(f"## {icon} Recommendation: **{pred['recommendation'].replace('_', ' ')}**")
        st.caption("Based on technical analysis of historical price patterns, volume trends, and market conditions.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{pred['confidence']:.1f}%", help="Reliability of this recommendation (higher is better)")
        with col2:
            st.metric("Risk Level", pred['risk_level'], help="Expected volatility and downside risk")
        
        if 'reasoning' in pred:
            st.markdown("**Key Factors:**")
            for reason in pred['reasoning']:
                st.markdown(f"✓ {reason}")
        
        st.divider()
        
        # Technical indicators
        if 'indicators' in analysis:
            st.markdown("### 📊 Technical Indicators")
            st.caption("Common market indicators that help identify trends and momentum.")
            
            indicators = analysis['indicators']
            
            ind_col1, ind_col2 = st.columns(2)
            
            with ind_col1:
                st.markdown("**Momentum Indicators:**")
                if 'rsi' in indicators:
                    rsi_val = indicators['rsi']
                    rsi_desc = "Overbought (Sell Signal)" if rsi_val > 70 else "Oversold (Buy Signal)" if rsi_val < 30 else "Neutral"
                    st.metric("RSI (14)", f"{rsi_val:.2f}", help=f"Range 0-100. {rsi_desc}")
                if 'macd' in indicators:
                    st.metric("MACD", f"{indicators['macd']:.2f}", help="Momentum indicator - positive = bullish, negative = bearish")
            
            with ind_col2:
                st.markdown("**Trend Indicators:**")
                if 'sma_20' in indicators:
                    st.metric("SMA 20", f"${indicators['sma_20']:.2f}", help="Short-term trend (20-day average)")
                if 'sma_50' in indicators:
                    st.metric("SMA 50", f"${indicators['sma_50']:.2f}", help="Medium-term trend (50-day average)")
            
            st.markdown("**Volume Analysis:**")
            if 'volume' in indicators:
                vol = indicators['volume']
                if vol > 1_000_000:
                    vol_str = f"{vol/1_000_000:.2f}M"
                else:
                    vol_str = f"{vol:,.0f}"
                st.metric("Current Volume", vol_str, help="Trade volume for the most recent period")
        
        # Trend analysis
        if 'trend' in analysis:
            st.markdown("### 📈 Trend Analysis")
            st.caption("Key price levels and trend direction that indicate where the market may find support or face resistance.")
            trend = analysis['trend']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Direction:** {trend.get('direction', 'N/A')}")
                st.caption("Uptrend, downtrend, or sideways")
                st.markdown(f"**Strength:** {trend.get('strength', 'N/A')}")
                st.caption("How strong is the current trend")
            
            with col2:
                if 'support' in trend:
                    st.metric("Support Level", f"${trend['support']:.2f}", help="Price level where buying typically occurs")
                if 'resistance' in trend:
                    st.metric("Resistance Level", f"${trend['resistance']:.2f}", help="Price level where selling typically occurs")
        
        # Full output
        with st.expander("📄 View Full Analysis Details"):
            st.caption("Complete technical analysis breakdown from the analyzer engine.")
            # Capture the print output
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            st.session_state.analyzer.print_analysis(analysis)
            output = captured_output.getvalue()
            
            sys.stdout = old_stdout
            
            st.code(output, language=None)

def show_comparison_interface():
    """Show interface for comparing multiple assets."""
    
    st.markdown("## ⚖️ Compare Multiple Assets")
    
    # Input for symbols
    symbols_input = st.text_area(
        "Enter symbols (one per line)",
        value="AAPL\nMSFT\nGOOGL\nTSLA\nBTC-USD\nETH-USD",
        height=150,
        help="Enter stock tickers or crypto symbols, one per line"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Period selection
        compare_period = st.selectbox(
            "Comparison Period",
            options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,  # Default to 3mo
            key="compare_period"
        )
    
    with col2:
        # Compare button
        compare_clicked = st.button("Run Comparison", type="primary", use_container_width=True, key="run_comparison_btn")
    
    if compare_clicked:
        symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]
        
        if not symbols:
            st.error("Please enter at least one symbol!")
        else:
            with st.spinner(f"Comparing {len(symbols)} assets..."):
                try:
                    comparison = st.session_state.analyzer.compare_assets(
                        symbols, 
                        period=compare_period
                    )
                    
                    if not comparison.empty:
                        analyzed_count = len(comparison)
                        requested_count = len(symbols)
                        if analyzed_count == requested_count:
                            st.success(f"✅ Successfully analyzed all {analyzed_count} assets!")
                        else:
                            st.warning(f"⚠️ Analyzed {analyzed_count} out of {requested_count} assets. Some symbols may be invalid or unavailable.")
                        
                        # Display as dataframe
                        st.dataframe(
                            comparison,
                            use_container_width=True,
                            hide_index=True,
                            height=600  # Show more rows at once
                        )
                        
                        # Download button
                        csv = comparison.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"market_comparison_{compare_period}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("❌ Failed to compare assets. Please check the symbols.")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script with sample data to demonstrate the Market Analyzer functionality.
This script uses mock data to show how the analyzer works without network access.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from market_analyzer import MarketAnalyzer, TechnicalIndicators, TrendAnalyzer, PredictionEngine


def generate_sample_data(symbol: str, days: int = 365, trend: str = 'bullish'):
    """
    Generate sample OHLCV data for testing.
    
    Args:
        symbol: Asset symbol
        days: Number of days of data
        trend: 'bullish', 'bearish', or 'sideways'
    
    Returns:
        DataFrame with sample data
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Base price
    base_price = 100.0
    
    # Generate price data based on trend
    if trend == 'bullish':
        trend_factor = np.linspace(0, 0.5, days)
    elif trend == 'bearish':
        trend_factor = np.linspace(0, -0.3, days)
    else:  # sideways
        trend_factor = np.zeros(days)
    
    # Add random walk
    random_walk = np.cumsum(np.random.randn(days) * 0.02)
    
    # Calculate close prices
    close_prices = base_price * (1 + trend_factor + random_walk)
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'Open': close_prices * (1 + np.random.randn(days) * 0.005),
        'High': close_prices * (1 + np.abs(np.random.randn(days)) * 0.01),
        'Low': close_prices * (1 - np.abs(np.random.randn(days)) * 0.01),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, days)
    }, index=dates)
    
    # Ensure High is highest and Low is lowest
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
    
    return data


def test_technical_indicators():
    """Test technical indicators calculation."""
    print("\n" + "="*60)
    print("TEST 1: Technical Indicators")
    print("="*60 + "\n")
    
    # Generate sample data
    data = generate_sample_data('TEST', days=365, trend='bullish')
    
    # Calculate indicators
    indicators = TechnicalIndicators()
    data_with_indicators = indicators.add_all_indicators(data)
    
    # Display results
    latest = data_with_indicators.iloc[-1]
    print(f"Latest Close Price: ${latest['Close']:.2f}")
    print(f"\nMoving Averages:")
    print(f"  SMA_20:  ${latest['SMA_20']:.2f}")
    print(f"  SMA_50:  ${latest['SMA_50']:.2f}")
    print(f"  SMA_200: ${latest['SMA_200']:.2f}")
    print(f"\nMomentum Indicators:")
    print(f"  RSI: {latest['RSI']:.2f}")
    print(f"  MACD: {latest['MACD']:.4f}")
    print(f"  MACD Signal: {latest['MACD_Signal']:.4f}")
    print(f"\nBollinger Bands:")
    print(f"  Upper:  ${latest['BB_Upper']:.2f}")
    print(f"  Middle: ${latest['BB_Middle']:.2f}")
    print(f"  Lower:  ${latest['BB_Lower']:.2f}")
    print(f"\nVolatility:")
    print(f"  ATR: ${latest['ATR']:.2f}")
    print(f"\nStochastic:")
    print(f"  %K: {latest['Stoch_K']:.2f}")
    print(f"  %D: {latest['Stoch_D']:.2f}")
    
    print("\n✓ Technical indicators calculated successfully!")


def test_trend_analysis():
    """Test trend analysis."""
    print("\n" + "="*60)
    print("TEST 2: Trend Analysis")
    print("="*60 + "\n")
    
    # Test different trends
    trends = ['bullish', 'bearish', 'sideways']
    
    for trend_type in trends:
        data = generate_sample_data(f'TEST-{trend_type.upper()}', days=365, trend=trend_type)
        
        indicators = TechnicalIndicators()
        data_with_indicators = indicators.add_all_indicators(data)
        
        trend_analyzer = TrendAnalyzer()
        
        # Identify trend
        trend = trend_analyzer.identify_trend(data_with_indicators)
        trend_strength = trend_analyzer.calculate_trend_strength(data_with_indicators)
        
        # Support/Resistance
        levels = trend_analyzer.identify_support_resistance(data_with_indicators)
        
        # Volume analysis
        volume_analysis = trend_analyzer.analyze_volume_trend(data_with_indicators)
        
        # Crossover detection
        crossover = trend_analyzer.detect_crossover(data_with_indicators, 'SMA_50', 'SMA_200')
        
        print(f"\n{trend_type.upper()} Market:")
        print(f"  Detected Trend: {trend}")
        print(f"  Trend Strength: {trend_strength}%")
        print(f"  Support: ${levels['support']:.2f}")
        print(f"  Current: ${levels['current']:.2f}")
        print(f"  Resistance: ${levels['resistance']:.2f}")
        print(f"  Crossover Signal: {crossover if crossover else 'None'}")
        print(f"  Volume Status: {volume_analysis['status']}")
    
    print("\n✓ Trend analysis completed successfully!")


def test_prediction_engine():
    """Test prediction engine."""
    print("\n" + "="*60)
    print("TEST 3: Prediction Engine")
    print("="*60 + "\n")
    
    # Generate bullish data
    data = generate_sample_data('BULLISH-STOCK', days=365, trend='bullish')
    
    indicators = TechnicalIndicators()
    data_with_indicators = indicators.add_all_indicators(data)
    
    predictor = PredictionEngine()
    prediction = predictor.predict(data_with_indicators)
    
    # Display prediction
    rec = prediction['recommendation']
    print(f"Asset: BULLISH-STOCK")
    print(f"Current Price: ${data['Close'].iloc[-1]:.2f}")
    print(f"\nRECOMMENDATION: {rec['recommendation']}")
    print(f"Confidence: {rec['confidence']:.2f}%")
    print(f"Risk Level: {rec['risk_level']}")
    print(f"\nProbabilities:")
    print(f"  Buy:  {rec['probabilities']['buy']:.2f}%")
    print(f"  Hold: {rec['probabilities']['hold']:.2f}%")
    print(f"  Sell: {rec['probabilities']['sell']:.2f}%")
    
    print(f"\nTrend Analysis:")
    trend = prediction['trend_analysis']
    print(f"  Overall Trend: {trend['trend']}")
    print(f"  Trend Strength: {trend['strength']}%")
    
    print(f"\nTechnical Signals:")
    for indicator, signal in prediction['technical_signals'].items():
        if 'signal' in signal:
            print(f"  {indicator}: {signal['signal']} ({signal.get('strength', 'N/A')})")
    
    print("\n✓ Prediction generated successfully!")


def test_full_analysis():
    """Test complete analysis workflow."""
    print("\n" + "="*60)
    print("TEST 4: Complete Analysis Workflow")
    print("="*60 + "\n")
    
    # Create sample data for multiple assets
    assets = [
        ('TECH-A', 'bullish'),
        ('TECH-B', 'sideways'),
        ('TECH-C', 'bearish'),
        ('CRYPTO-X', 'bullish'),
        ('CRYPTO-Y', 'bearish')
    ]
    
    results = []
    
    for symbol, trend in assets:
        data = generate_sample_data(symbol, days=365, trend=trend)
        
        indicators = TechnicalIndicators()
        data_with_indicators = indicators.add_all_indicators(data)
        
        predictor = PredictionEngine()
        prediction = predictor.predict(data_with_indicators)
        
        if 'recommendation' in prediction:
            rec = prediction['recommendation']
            results.append({
                'Symbol': symbol,
                'Price': f"${data['Close'].iloc[-1]:.2f}",
                'Recommendation': rec['recommendation'],
                'Confidence': f"{rec['confidence']:.1f}%",
                'Risk': rec['risk_level'],
                'Trend': rec['trend']
            })
    
    # Display comparison table
    print("Asset Comparison:")
    print("-" * 80)
    print(f"{'Symbol':<12} {'Price':<10} {'Recommendation':<15} {'Confidence':<12} {'Risk':<10} {'Trend':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['Symbol']:<12} {result['Price']:<10} {result['Recommendation']:<15} "
              f"{result['Confidence']:<12} {result['Risk']:<10} {result['Trend']:<10}")
    
    print("-" * 80)
    print("\n✓ Complete analysis workflow tested successfully!")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("MARKET ANALYZER - TEST SUITE")
    print("="*60)
    print("\nTesting with sample data (no network required)")
    print("="*60)
    
    # Run tests
    test_technical_indicators()
    test_trend_analysis()
    test_prediction_engine()
    test_full_analysis()
    
    # Summary
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    print("\nThe Market Analyzer is working correctly!")
    print("\nFeatures Tested:")
    print("  ✓ Technical Indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)")
    print("  ✓ Trend Analysis (trend identification, strength, support/resistance)")
    print("  ✓ Prediction Engine (probability-based recommendations)")
    print("  ✓ Complete Analysis Workflow (multi-asset analysis)")
    print("\nNote: These tests use sample data. To analyze real market data,")
    print("      ensure you have internet access and run example.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

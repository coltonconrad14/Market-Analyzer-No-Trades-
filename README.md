# Market Analyzer (No Trades)

A comprehensive market analysis tool that identifies trends in different financial markets using technical analysis. This analyzer provides predictions based on probability and likelihood for stocks and cryptocurrencies **without any trading functionality** - it's purely for information and prediction purposes.

**🚀 NEW: Easy-to-use GUI Application!** Run `python app.py` for a simple point-and-click interface. See [APP_OPTIONS.md](APP_OPTIONS.md) for all the ways to use this tool.

## Features

### Technical Analysis Indicators
- **Moving Averages**: SMA (20, 50, 200 periods), EMA (12, 26 periods)
- **RSI (Relative Strength Index)**: Identifies overbought/oversold conditions
- **MACD (Moving Average Convergence Divergence)**: Trend following momentum indicator
- **Bollinger Bands**: Volatility and price level indicators
- **ATR (Average True Range)**: Volatility measurement
- **OBV (On-Balance Volume)**: Volume-based momentum indicator
- **Stochastic Oscillator**: Momentum indicator comparing closing price to price range

### Trend Analysis
- Overall trend identification (Bullish, Bearish, Sideways)
- Trend strength calculation
- Support and resistance level identification
- Crossover detection (Golden Cross, Death Cross)
- Volume trend analysis
- Divergence detection

### Prediction Engine
- Probability-based recommendations (Buy/Sell/Hold)
- Confidence scores for predictions
- Risk level assessment
- Multi-indicator signal aggregation
- Comprehensive market analysis reports

### Supported Assets
- **Stocks**: Any ticker available on Yahoo Finance (e.g., AAPL, MSFT, GOOGL, TSLA)
- **Cryptocurrencies**: Major cryptocurrencies (e.g., BTC-USD, ETH-USD, ADA-USD)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/coltonconrad14/Market-Analyzer-No-Trades-.git
cd Market-Analyzer-No-Trades-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 🖥️ GUI Application (Easiest Method)

Launch the easy-to-use graphical interface:

```bash
python app.py
```

The GUI provides:
- **Simple Input Form**: Enter symbol, select period and asset type
- **Quick Examples**: Pre-loaded buttons for popular stocks and crypto
- **Real-time Analysis**: View results in formatted display
- **Asset Comparison**: Compare multiple assets side-by-side
- **User-Friendly**: No command-line knowledge required!

### 💻 Interactive CLI

Use the command-line interface for quick analysis:

```bash
python analyze.py
```

### 📋 Quick Start - Run Example

Run the example script to see the analyzer in action:

```bash
python example.py
```

This will demonstrate:
1. Single stock analysis (Apple - AAPL)
2. Cryptocurrency analysis (Bitcoin - BTC)
3. Comparison of multiple assets
4. Top buy recommendations from a portfolio

### Basic Usage

```python
from market_analyzer import MarketAnalyzer

# Initialize the analyzer
analyzer = MarketAnalyzer()

# Analyze a stock
stock_analysis = analyzer.analyze_asset('AAPL', period='1y', asset_type='stock')
analyzer.print_analysis(stock_analysis)

# Analyze a cryptocurrency
crypto_analysis = analyzer.analyze_asset('BTC', period='6mo', asset_type='crypto')
analyzer.print_analysis(crypto_analysis)
```

### Advanced Usage

#### Compare Multiple Assets

```python
from market_analyzer import MarketAnalyzer

analyzer = MarketAnalyzer()

# Compare stocks and crypto
symbols = ['AAPL', 'MSFT', 'GOOGL', 'BTC-USD', 'ETH-USD']
comparison = analyzer.compare_assets(symbols, period='3mo')
print(comparison)
```

#### Get Top Recommendations

```python
from market_analyzer import MarketAnalyzer

analyzer = MarketAnalyzer()

# Get top 5 buy recommendations
portfolio = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'BTC-USD', 'ETH-USD']
top_buys = analyzer.get_top_recommendations(portfolio, top_n=5, recommendation_type='BUY')
print(top_buys)
```

#### Individual Component Usage

```python
from market_analyzer import TechnicalIndicators, TrendAnalyzer, PredictionEngine
from market_analyzer.data_fetcher import DataFetcher
import pandas as pd

# Fetch data
fetcher = DataFetcher()
data = fetcher.fetch_stock_data('AAPL', period='1y')

# Calculate indicators
indicators = TechnicalIndicators()
data_with_indicators = indicators.add_all_indicators(data)

# Analyze trend
trend_analyzer = TrendAnalyzer()
trend = trend_analyzer.identify_trend(data_with_indicators)
support_resistance = trend_analyzer.identify_support_resistance(data_with_indicators)

# Generate prediction
predictor = PredictionEngine()
prediction = predictor.predict(data_with_indicators)
```

## Analysis Output

The analyzer provides detailed reports including:

### Recommendation Section
- **Recommendation**: BUY, SELL, or HOLD
- **Confidence**: Percentage confidence in the recommendation
- **Risk Level**: LOW, MEDIUM, or HIGH
- **Probabilities**: Individual probabilities for buy, sell, and hold actions

### Trend Analysis
- **Overall Trend**: Market direction (BULLISH, BEARISH, SIDEWAYS)
- **Trend Strength**: Percentage strength of the trend
- **Crossover Signals**: Golden cross or death cross detection
- **Support/Resistance Levels**: Key price levels

### Technical Indicators
- **RSI**: Current value and signal (oversold/overbought/neutral)
- **MACD**: Trend direction signal
- **Bollinger Bands**: Price position relative to bands
- **Moving Averages**: Trend signals from MA crossovers
- **Stochastic**: Momentum signals

### Volume Analysis
- **Volume Status**: HIGH, NORMAL, or LOW volume
- **Volume Ratio**: Current volume compared to average
- **Volume Trend**: Supporting or contradicting price action

## Example Output

```
============================================================
MARKET ANALYSIS REPORT
============================================================

Symbol: AAPL
Asset Type: STOCK
Current Price: $178.25
Analysis Date: 2024-02-04

============================================================
RECOMMENDATION: BUY
============================================================
Confidence: 72.5%
Risk Level: MEDIUM

Probabilities:
  Buy:  65.0%
  Hold: 20.0%
  Sell: 15.0%

============================================================
TREND ANALYSIS
============================================================
Overall Trend: BULLISH
Trend Strength: 45.8%
Crossover Signal: BULLISH

Key Levels:
  Resistance: $185.50
  Current:    $178.25
  Support:    $165.00

============================================================
TECHNICAL INDICATORS
============================================================

RSI:
  value: 58.45
  signal: NEUTRAL
  strength: HOLD

MACD:
  signal: BULLISH
  strength: BUY

Bollinger:
  signal: NEUTRAL
  strength: HOLD
...
```

## API Reference

### MarketAnalyzer

Main class for market analysis.

**Methods:**
- `analyze_asset(symbol, period, interval, asset_type)`: Analyze a single asset
- `analyze_multiple_assets(symbols, period, interval)`: Analyze multiple assets
- `compare_assets(symbols, period)`: Compare and rank multiple assets
- `print_analysis(analysis)`: Print formatted analysis report
- `get_top_recommendations(symbols, top_n, recommendation_type)`: Get top N recommendations

### TechnicalIndicators

Calculate technical analysis indicators.

**Methods:**
- `calculate_sma(data, period, column)`: Simple Moving Average
- `calculate_ema(data, period, column)`: Exponential Moving Average
- `calculate_rsi(data, period, column)`: Relative Strength Index
- `calculate_macd(data, fast, slow, signal, column)`: MACD
- `calculate_bollinger_bands(data, period, std_dev, column)`: Bollinger Bands
- `calculate_atr(data, period)`: Average True Range
- `calculate_obv(data)`: On-Balance Volume
- `calculate_stochastic(data, k_period, d_period)`: Stochastic Oscillator
- `add_all_indicators(data)`: Add all indicators to DataFrame

### TrendAnalyzer

Analyze market trends and patterns.

**Methods:**
- `identify_trend(data, sma_short, sma_long)`: Identify overall trend
- `calculate_trend_strength(data)`: Calculate trend strength
- `identify_support_resistance(data, window)`: Find support/resistance levels
- `detect_crossover(data, fast_col, slow_col)`: Detect MA crossovers
- `analyze_volume_trend(data, period)`: Analyze volume trends
- `detect_divergence(data, price_col, indicator_col, lookback)`: Detect divergences

### PredictionEngine

Generate predictions and recommendations.

**Methods:**
- `analyze_indicators(data)`: Analyze technical indicator signals
- `calculate_probability_score(signals)`: Calculate buy/sell/hold probabilities
- `generate_recommendation(probabilities, trend, trend_strength)`: Generate final recommendation
- `predict(data)`: Generate complete prediction

## Dependencies

- `yfinance>=0.2.18`: Market data fetching
- `pandas>=1.5.3`: Data manipulation
- `numpy>=1.24.0`: Numerical computations
- `requests>=2.28.0`: HTTP requests
- `matplotlib>=3.7.0`: Data visualization (optional)

## Important Disclaimers

⚠️ **NO TRADING FUNCTIONALITY**: This tool does not execute trades or move money. It is purely for analysis and educational purposes.

⚠️ **NOT FINANCIAL ADVICE**: The predictions and recommendations provided by this tool are based on technical analysis algorithms and should not be considered as financial advice. Always conduct your own research and consult with financial professionals before making investment decisions.

⚠️ **PAST PERFORMANCE**: Historical data and technical indicators do not guarantee future results. Markets are unpredictable and involve substantial risk.

⚠️ **USE AT YOUR OWN RISK**: The creators and contributors of this project are not responsible for any financial losses incurred from using this tool.

## Architecture

The project is organized into modular components:

```
Market-Analyzer-No-Trades-/
├── market_analyzer/
│   ├── __init__.py              # Package initialization
│   ├── analyzer.py              # Main MarketAnalyzer class
│   ├── data_fetcher.py          # Data fetching from Yahoo Finance
│   ├── technical_indicators.py  # Technical indicator calculations
│   ├── trend_analyzer.py        # Trend analysis and pattern detection
│   └── prediction_engine.py     # Prediction and recommendation engine
├── example.py                   # Example usage script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

## License

This project is provided as-is for educational and informational purposes.

## Acknowledgments

- Market data provided by Yahoo Finance via yfinance library
- Technical analysis methodologies based on standard financial analysis practices
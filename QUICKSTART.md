# Quick Start Guide

This is a quick reference guide to get started with the Market Analyzer.

## Installation

```bash
# Clone the repository
git clone https://github.com/coltonconrad14/Market-Analyzer-No-Trades-.git
cd Market-Analyzer-No-Trades-

# Install dependencies
pip install -r requirements.txt
```

## Usage Options

### 1. Interactive CLI (Easiest)

Analyze assets one at a time interactively:

```bash
python analyze.py
```

You'll be prompted to enter:
- Symbol (e.g., AAPL, BTC-USD)
- Time period (e.g., 1y, 6mo, 3mo)
- Asset type (stock or crypto)

### 2. Example Script

Run predefined examples:

```bash
python example.py
```

This demonstrates:
- Single stock analysis (AAPL)
- Cryptocurrency analysis (BTC)
- Multiple asset comparison
- Top buy recommendations

### 3. Test with Sample Data

Test the analyzer without internet access:

```bash
python test_analyzer.py
```

Uses simulated data to verify all features work correctly.

### 4. Python API

Use in your own scripts:

```python
from market_analyzer import MarketAnalyzer

# Initialize
analyzer = MarketAnalyzer()

# Analyze a stock
analysis = analyzer.analyze_asset('AAPL', period='1y', asset_type='stock')
analyzer.print_analysis(analysis)

# Compare multiple assets
comparison = analyzer.compare_assets(['AAPL', 'MSFT', 'GOOGL'])
print(comparison)

# Get top recommendations
top_buys = analyzer.get_top_recommendations(
    ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
    top_n=3,
    recommendation_type='BUY'
)
print(top_buys)
```

## Common Symbols

### Popular Stocks
- AAPL (Apple)
- MSFT (Microsoft)
- GOOGL (Google/Alphabet)
- AMZN (Amazon)
- TSLA (Tesla)
- NVDA (NVIDIA)
- META (Meta/Facebook)

### Cryptocurrencies
- BTC-USD (Bitcoin)
- ETH-USD (Ethereum)
- ADA-USD (Cardano)
- SOL-USD (Solana)
- DOGE-USD (Dogecoin)

## Time Periods

- `1d` - 1 day
- `5d` - 5 days
- `1mo` - 1 month
- `3mo` - 3 months
- `6mo` - 6 months
- `1y` - 1 year (default)
- `2y` - 2 years
- `5y` - 5 years
- `max` - Maximum available

## Understanding the Output

### Recommendation
- **BUY**: Technical indicators suggest upward potential
- **HOLD**: Mixed signals or neutral conditions
- **SELL**: Technical indicators suggest downward pressure

### Confidence
- Higher % = Stronger agreement among indicators
- 70%+ = High confidence
- 50-70% = Medium confidence
- <50% = Low confidence

### Risk Level
- **LOW**: Strong, clear signals
- **MEDIUM**: Moderate agreement among indicators
- **HIGH**: Conflicting or weak signals

### Technical Indicators

#### RSI (Relative Strength Index)
- < 30: Oversold (potential buy)
- 30-70: Neutral
- > 70: Overbought (potential sell)

#### MACD
- BULLISH: Fast line above slow line (buy signal)
- BEARISH: Fast line below slow line (sell signal)

#### Bollinger Bands
- Price at upper band: Overbought
- Price at lower band: Oversold
- Price in middle: Neutral

#### Moving Averages
- GOLDEN_CROSS: 50-day MA crosses above 200-day MA (bullish)
- DEATH_CROSS: 50-day MA crosses below 200-day MA (bearish)

## Important Notes

⚠️ **This tool does not trade or move money**

⚠️ **Not financial advice** - Always do your own research

⚠️ **Past performance ≠ future results**

⚠️ **Markets are unpredictable** - Use at your own risk

## Need Help?

See the full README.md for detailed documentation, API reference, and advanced usage examples.

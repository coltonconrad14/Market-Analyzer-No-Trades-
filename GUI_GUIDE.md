# 🚀 Quick Start Guide - GUI App

Welcome to the Market Analyzer GUI! This is the easiest way to analyze stocks and cryptocurrencies.

## Starting the App

### Method 1: Using the Launcher (Recommended)
```bash
./run_app.sh
```

### Method 2: Direct Python Command
```bash
python app.py
```

## Using the GUI

### 1. Analyze a Single Asset

1. **Enter Symbol**: Type the stock ticker or crypto symbol
   - Examples: `AAPL`, `MSFT`, `BTC-USD`, `ETH-USD`

2. **Select Period**: Choose timeframe from dropdown
   - Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y

3. **Choose Asset Type**: 
   - Stock (for traditional stocks)
   - Cryptocurrency (for crypto assets)

4. **Click "🔍 Analyze Asset"**

5. View the detailed analysis in the results panel!

### 2. Quick Examples

Use the **Quick Examples** dropdown to instantly load popular assets:
- AAPL - Apple
- MSFT - Microsoft  
- GOOGL - Google
- TSLA - Tesla
- BTC-USD - Bitcoin
- ETH-USD - Ethereum

Just select one and click analyze!

### 3. Compare Multiple Assets

1. Click **"⚖️  Compare Assets"** button

2. A new window opens - enter symbols (one per line):
   ```
   AAPL
   MSFT
   GOOGL
   TSLA
   BTC-USD
   ```

3. Select the period for comparison

4. Click **"Compare"**

5. View side-by-side comparison table!

### 4. Understanding the Results

The analysis includes:
- **Current Price** and price changes
- **Technical Indicators** (RSI, MACD, Bollinger Bands, etc.)
- **Trend Analysis** (Bullish/Bearish/Sideways)
- **Support/Resistance Levels**
- **Prediction** with confidence score
- **Risk Assessment**

## Tips

✅ **DO:**
- Try different time periods to see different trends
- Use comparison to evaluate multiple options
- Check the analysis regularly (markets change!)
- Read the full analysis including risk levels

⚠️ **REMEMBER:**
- This is for **informational purposes only**
- No trading or money movement occurs
- Always do your own research
- Past performance doesn't guarantee future results

## Troubleshooting

### App won't start?
- Make sure Python is installed: `python --version`
- Install requirements: `pip install -r requirements.txt`

### Symbol not found?
- Check the spelling of the ticker symbol
- For crypto, use the format: `BTC-USD`, `ETH-USD`
- Some assets may not be available in Yahoo Finance

### Analysis takes too long?
- Large time periods (5y) take longer to process
- Check your internet connection
- Try a shorter period (1mo, 3mo)

## Need More Help?

- Check the main [README.md](README.md) for detailed information
- Review the [QUICKSTART.md](QUICKSTART.md) guide
- Run `python example.py` to see code examples

---

**Happy Analyzing! 📊**

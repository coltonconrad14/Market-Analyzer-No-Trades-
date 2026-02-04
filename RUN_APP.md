# 🚀 RUN THE APP - Quick Reference

## Fastest Way to Start

```bash
python app.py
```

That's it! The GUI will open and you can start analyzing assets immediately.

---

## All Methods

### 🖥️ Method 1: Direct Python (Recommended)
```bash
python app.py
```
**Pros:** Simple, works everywhere
**Cons:** Need to type the command

---

### 🔵 Method 2: Shell Script (Linux/Mac)
```bash
./run_app.sh
```
**Pros:** Shorter command, auto-detects Python
**Cons:** Unix/Mac only

---

### 🟦 Method 3: Batch File (Windows)
```
Double-click: run_app.bat
```
**Pros:** No typing needed
**Cons:** Windows only

---

### 🎯 Method 4: Desktop Shortcut
```bash
python create_shortcut.py
```
Then click the desktop icon!

**Pros:** Easiest access, permanent shortcut
**Cons:** One-time setup required

---

## First Time Setup

### 1. Make sure Python is installed
```bash
python --version
```
Should show Python 3.6 or higher

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

---

## Quick Test

Want to test if everything works? Try this:

```bash
# Navigate to project directory
cd Market-Analyzer-No-Trades-

# Run the GUI
python app.py
```

In the GUI:
1. Leave the default "AAPL" symbol
2. Click "🔍 Analyze Asset"
3. Wait a few seconds
4. See the analysis results!

---

## Troubleshooting

### App won't start?

**Check Python:**
```bash
python --version
```
Need Python 3.6+

**Install requirements:**
```bash
pip install -r requirements.txt
```

**Try alternative Python command:**
```bash
python3 app.py
```

---

### "No module named 'tkinter'"?

**Linux/Ubuntu:**
```bash
sudo apt-get install python3-tk
```

**Mac:**
Tkinter should be included with Python

**Windows:**
Reinstall Python with "tcl/tk" option checked

---

### "No module named 'market_analyzer'"?

Make sure you're in the project directory:
```bash
cd Market-Analyzer-No-Trades-
python app.py
```

---

### Analysis shows errors?

- Check your internet connection (needs to fetch market data)
- Verify the symbol is correct (AAPL, not Apple)
- For crypto, use format: BTC-USD, ETH-USD
- Try a different time period

---

## Alternative: CLI Version

Don't want the GUI? Use the command-line version:

```bash
python analyze.py
```

Follow the prompts to analyze assets.

---

## Alternative: Example Script

See a demo of all features:

```bash
python example.py
```

---

## System Requirements

- **OS:** Windows, Linux, or macOS
- **Python:** 3.6 or higher
- **RAM:** 2GB minimum
- **Internet:** Required for fetching market data
- **Display:** Any (GUI scales automatically)

---

## What to Do After Starting

1. **Enter a Symbol**
   - Type: AAPL, MSFT, GOOGL, etc.
   - Or: BTC-USD, ETH-USD, etc.

2. **Select Period**
   - Choose from dropdown
   - Start with "1y" (1 year)

3. **Choose Asset Type**
   - Stock or Cryptocurrency

4. **Click Analyze**
   - Wait for results (5-30 seconds)

5. **Read the Analysis**
   - Scroll through results
   - Check recommendation
   - Note the confidence score

6. **Try More**
   - Click Quick Examples
   - Compare multiple assets
   - Experiment with different periods

---

## Quick Examples to Try

| Symbol | Type | What It Is |
|--------|------|------------|
| AAPL | Stock | Apple Inc. |
| MSFT | Stock | Microsoft |
| GOOGL | Stock | Google |
| TSLA | Stock | Tesla |
| BTC-USD | Crypto | Bitcoin |
| ETH-USD | Crypto | Ethereum |

---

## Getting Help

- **GUI Guide:** [GUI_GUIDE.md](GUI_GUIDE.md)
- **All Options:** [APP_OPTIONS.md](APP_OPTIONS.md)
- **Full Docs:** [README.md](README.md)
- **Quick Start:** [QUICKSTART.md](QUICKSTART.md)

---

## Common Questions

**Q: Does this trade stocks?**
A: No! It only analyzes and provides information.

**Q: Is the data real-time?**
A: Data is from Yahoo Finance, typically 15-20 minute delay.

**Q: How accurate are the predictions?**
A: Use as one of many tools. Always do your own research!

**Q: Can I use this professionally?**
A: It's for informational/educational purposes only.

**Q: What about other exchanges?**
A: Works with any asset on Yahoo Finance.

---

## Summary

**To run:** `python app.py`

**To learn:** Read [GUI_GUIDE.md](GUI_GUIDE.md)

**For help:** Check [APP_OPTIONS.md](APP_OPTIONS.md)

**Have fun analyzing! 📊**

#!/usr/bin/env python3
"""
Example script demonstrating the Market Analyzer capabilities.
"""

from market_analyzer import MarketAnalyzer


def main():
    """Run example market analysis."""
    
    # Initialize the analyzer
    analyzer = MarketAnalyzer()
    
    print("\n" + "="*60)
    print("MARKET ANALYZER - DEMONSTRATION")
    print("="*60)
    print("\nThis tool analyzes financial markets for informational purposes only.")
    print("No trading or money movement functionality is included.\n")
    
    # Example 1: Analyze a single stock
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Stock Analysis")
    print("="*60)
    
    stock_analysis = analyzer.analyze_asset('AAPL', period='6mo', asset_type='stock')
    if stock_analysis:
        analyzer.print_analysis(stock_analysis)
    
    # Example 2: Analyze a cryptocurrency
    print("\n" + "="*60)
    print("EXAMPLE 2: Cryptocurrency Analysis")
    print("="*60)
    
    crypto_analysis = analyzer.analyze_asset('BTC', period='6mo', asset_type='crypto')
    if crypto_analysis:
        analyzer.print_analysis(crypto_analysis)
    
    # Example 3: Compare multiple assets
    print("\n" + "="*60)
    print("EXAMPLE 3: Compare Multiple Assets")
    print("="*60)
    
    symbols_to_compare = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'BTC-USD', 'ETH-USD']
    print(f"\nComparing: {', '.join(symbols_to_compare)}\n")
    
    comparison = analyzer.compare_assets(symbols_to_compare, period='3mo')
    if not comparison.empty:
        print(comparison.to_string(index=False))
    
    # Example 4: Get top buy recommendations
    print("\n" + "="*60)
    print("EXAMPLE 4: Top Buy Recommendations")
    print("="*60)
    
    # Popular stocks and crypto
    portfolio = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        'NVDA', 'META', 'NFLX', 'BTC-USD', 'ETH-USD'
    ]
    
    print(f"\nAnalyzing {len(portfolio)} assets for top buy recommendations...\n")
    
    top_buys = analyzer.get_top_recommendations(portfolio, top_n=5, recommendation_type='BUY')
    if not top_buys.empty:
        print("Top 5 Buy Recommendations:")
        print(top_buys.to_string(index=False))
    else:
        print("No strong buy recommendations found in current market conditions.")
    
    # Example 5: Market Scanner - Top Bullish and Bearish
    print("\n" + "="*60)
    print("EXAMPLE 5: Market Scanner - Bullish & Bearish Signals")
    print("="*60)
    
    # Scan with default watchlist
    print("\nUsing default market watchlist (stocks + crypto)...")
    scan_results = analyzer.scan_market(period='3mo', top_n=5)
    analyzer.print_market_scan(scan_results)
    
    # Alternative: Scan custom list
    print("\n" + "="*60)
    print("EXAMPLE 6: Custom Market Scanner")
    print("="*60)
    
    custom_watchlist = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA',
        'BTC-USD', 'ETH-USD', 'SOL-USD'
    ]
    
    print(f"\nScanning custom watchlist: {', '.join(custom_watchlist)}")
    custom_scan = analyzer.scan_market(symbols=custom_watchlist, period='1mo', top_n=3)
    analyzer.print_market_scan(custom_scan)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print("\nDISCLAIMER: This analysis is for informational purposes only.")
    print("Always conduct your own research before making investment decisions.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

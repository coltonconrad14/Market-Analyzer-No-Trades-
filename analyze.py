#!/usr/bin/env python3
"""
Interactive CLI for Market Analyzer.
Simple command-line interface to analyze individual assets.
"""

from market_analyzer import MarketAnalyzer
import sys


def print_header():
    """Print application header."""
    print("\n" + "="*60)
    print("MARKET ANALYZER - Interactive CLI")
    print("="*60)
    print("\n⚠️  FOR INFORMATIONAL PURPOSES ONLY - NO TRADING")
    print("    This tool does not execute trades or move money.\n")


def get_user_input():
    """Get asset information from user."""
    print("Choose an option:\n")
    print("1. Analyze individual asset")
    print("2. Scan market for top bullish/bearish signals")
    print("3. Quit\n")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '3' or choice.lower() in ['quit', 'exit', 'q']:
        return 'quit', None, None
    
    if choice == '2':
        return 'scan', None, None
    
    # Choice 1 or default - individual asset analysis
    print("\nEnter asset information:\n")
    
    symbol = input("Symbol (e.g., AAPL, MSFT, BTC-USD): ").strip().upper()
    if symbol.lower() == 'quit' or symbol == '':
        return 'quit', None, None
    
    print("\nTime period options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y")
    period = input("Period (default: 1y): ").strip() or "1y"
    
    print("\nAsset type options: stock, crypto")
    asset_type = input("Asset type (default: stock): ").strip().lower() or "stock"
    
    return symbol, period, asset_type


def main():
    """Main interactive loop."""
    print_header()
    
    # Initialize analyzer
    analyzer = MarketAnalyzer()
    
    while True:
        try:
            # Get user input
            symbol, period, asset_type = get_user_input()
            
            if symbol == 'quit':
                print("\nThank you for using Market Analyzer!")
                print("="*60 + "\n")
                break
            
            # Market scan mode
            if symbol == 'scan':
                print("\n" + "="*60)
                print("MARKET SCANNER MODE")
                print("="*60)
                
                # Get scan parameters
                print("\nScan parameters:\n")
                print("Time period options: 1d, 5d, 1mo, 3mo, 6mo, 1y")
                scan_period = input("Period (default: 3mo): ").strip() or "3mo"
                
                print("\nHow many top recommendations to show?")
                try:
                    top_n = int(input("Number (default: 5): ").strip() or "5")
                except ValueError:
                    top_n = 5
                
                print("\nUse default watchlist or custom?")
                print("1. Default watchlist (30+ stocks and crypto)")
                print("2. Custom list")
                watchlist_choice = input("Choice (default: 1): ").strip() or "1"
                
                custom_symbols = None
                if watchlist_choice == '2':
                    print("\nEnter symbols separated by commas (e.g., AAPL,MSFT,BTC-USD):")
                    symbol_input = input("Symbols: ").strip().upper()
                    if symbol_input:
                        custom_symbols = [s.strip() for s in symbol_input.split(',')]
                
                # Run market scan
                print("\nStarting market scan...")
                scan_results = analyzer.scan_market(
                    symbols=custom_symbols, 
                    period=scan_period, 
                    top_n=top_n
                )
                analyzer.print_market_scan(scan_results)
            else:
                # Individual asset analysis mode
                print(f"\nAnalyzing {symbol}...")
                analysis = analyzer.analyze_asset(symbol, period=period, asset_type=asset_type)
                
                if analysis:
                    analyzer.print_analysis(analysis)
                else:
                    print(f"\n❌ Failed to analyze {symbol}")
                    print("    Please check the symbol and try again.\n")
            
            # Ask if user wants to continue
            print("\n" + "-"*60)
            continue_input = input("\nContinue? (y/n): ").strip().lower()
            if continue_input not in ['y', 'yes']:
                print("\nThank you for using Market Analyzer!")
                print("="*60 + "\n")
                break
                
        except KeyboardInterrupt:
            print("\n\nExiting Market Analyzer...")
            print("="*60 + "\n")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()

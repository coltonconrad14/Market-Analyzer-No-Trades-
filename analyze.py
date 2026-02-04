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
    print("Enter asset information (or 'quit' to exit):\n")
    
    symbol = input("Symbol (e.g., AAPL, MSFT, BTC-USD): ").strip().upper()
    if symbol.lower() == 'quit' or symbol == '':
        return None, None, None
    
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
            
            if symbol is None:
                print("\nThank you for using Market Analyzer!")
                print("="*60 + "\n")
                break
            
            # Analyze asset
            print(f"\nAnalyzing {symbol}...")
            analysis = analyzer.analyze_asset(symbol, period=period, asset_type=asset_type)
            
            if analysis:
                analyzer.print_analysis(analysis)
            else:
                print(f"\n❌ Failed to analyze {symbol}")
                print("    Please check the symbol and try again.\n")
            
            # Ask if user wants to continue
            print("\n" + "-"*60)
            continue_input = input("\nAnalyze another asset? (y/n): ").strip().lower()
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

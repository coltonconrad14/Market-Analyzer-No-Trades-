#!/usr/bin/env python3
"""
Market Scanner Script - Quick market scan for top bullish and bearish recommendations.

This script provides a fast way to scan the market for trading opportunities
without using the interactive CLI.
"""

from market_analyzer import MarketAnalyzer
import sys


def print_banner():
    """Print application banner."""
    print("\n" + "="*60)
    print("MARKET SCANNER")
    print("="*60)
    print("\n⚠️  FOR INFORMATIONAL PURPOSES ONLY - NO TRADING")
    print("    This tool analyzes markets but does not execute trades.\n")


def quick_scan():
    """Perform a quick market scan with default settings."""
    print_banner()
    
    # Initialize analyzer
    analyzer = MarketAnalyzer()
    
    print("Running quick market scan with default settings...")
    print("(Default: 30+ stocks & crypto, 3-month period, top 5 each)\n")
    
    # Perform scan
    scan_results = analyzer.scan_market(period='3mo', top_n=5)
    analyzer.print_market_scan(scan_results)


def custom_scan():
    """Perform a custom market scan with user-defined parameters."""
    print_banner()
    
    # Initialize analyzer
    analyzer = MarketAnalyzer()
    
    # Get scan parameters
    print("Configure your market scan:\n")
    
    # Time period
    print("Time period options: 1d, 5d, 1mo, 3mo, 6mo, 1y")
    period = input("Period (default: 3mo): ").strip() or "3mo"
    
    # Number of recommendations
    print("\nHow many top recommendations to show?")
    try:
        top_n = int(input("Number (default: 5): ").strip() or "5")
    except ValueError:
        top_n = 5
    
    # Watchlist choice
    print("\n1. Use default watchlist (30+ stocks and crypto)")
    print("2. Enter custom symbols")
    choice = input("Choice (default: 1): ").strip() or "1"
    
    symbols = None
    if choice == '2':
        print("\nEnter symbols separated by commas (e.g., AAPL,MSFT,BTC-USD):")
        symbol_input = input("Symbols: ").strip().upper()
        if symbol_input:
            symbols = [s.strip() for s in symbol_input.split(',')]
            print(f"\nWill scan: {', '.join(symbols)}")
        else:
            print("\nNo symbols entered. Using default watchlist.")
    
    # Perform scan
    print("\nStarting market scan...\n")
    scan_results = analyzer.scan_market(symbols=symbols, period=period, top_n=top_n)
    analyzer.print_market_scan(scan_results)


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--custom', '-c']:
            custom_scan()
        elif sys.argv[1] in ['--help', '-h']:
            print("\nMarket Scanner Usage:\n")
            print("  python scan_market.py           Run quick scan (default)")
            print("  python scan_market.py --custom  Run custom scan with parameters")
            print("  python scan_market.py --help    Show this help message\n")
        else:
            print(f"\nUnknown option: {sys.argv[1]}")
            print("Use --help to see available options.\n")
    else:
        # Default: quick scan
        quick_scan()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMarket scan cancelled by user.")
        print("="*60 + "\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please try again or use --help for options.\n")
        sys.exit(1)

"""
Main Market Analyzer class that integrates all components.
"""

import pandas as pd
from typing import Dict, List, Optional
from .data_fetcher import DataFetcher
from .technical_indicators import TechnicalIndicators
from .trend_analyzer import TrendAnalyzer
from .prediction_engine import PredictionEngine


class MarketAnalyzer:
    """
    Main class for analyzing financial markets.
    
    This analyzer provides technical analysis and predictions for stocks and cryptocurrencies
    without any trading functionality.
    """
    
    def __init__(self):
        """Initialize the Market Analyzer."""
        self.data_fetcher = DataFetcher()
        self.technical_indicators = TechnicalIndicators()
        self.trend_analyzer = TrendAnalyzer()
        self.prediction_engine = PredictionEngine()
    
    def analyze_asset(self, symbol: str, period: str = "1y", interval: str = "1d", 
                      asset_type: str = "stock") -> Optional[Dict]:
        """
        Perform complete analysis on an asset.
        
        Args:
            symbol: Asset symbol (e.g., 'AAPL', 'BTC-USD')
            period: Time period for analysis
            interval: Data interval
            asset_type: Type of asset ('stock' or 'crypto')
        
        Returns:
            Complete analysis dictionary or None if analysis fails
        """
        print(f"\n{'='*60}")
        print(f"Analyzing {symbol} ({asset_type.upper()})")
        print(f"{'='*60}\n")
        
        # Fetch data
        if asset_type.lower() == 'crypto':
            data = self.data_fetcher.fetch_crypto_data(symbol, period, interval)
        else:
            data = self.data_fetcher.fetch_stock_data(symbol, period, interval)
        
        if data is None or data.empty:
            print(f"Failed to fetch data for {symbol}")
            return None
        
        print(f"✓ Data fetched: {len(data)} periods")
        
        # Add technical indicators
        data_with_indicators = self.technical_indicators.add_all_indicators(data)
        print(f"✓ Technical indicators calculated")
        
        # Generate prediction
        prediction = self.prediction_engine.predict(data_with_indicators)
        print(f"✓ Prediction generated\n")
        
        # Add symbol info
        prediction['symbol'] = symbol
        prediction['asset_type'] = asset_type
        prediction['current_price'] = round(data['Close'].iloc[-1], 2)
        
        return prediction
    
    def analyze_multiple_assets(self, symbols: List[str], period: str = "1y", 
                                interval: str = "1d") -> Dict[str, Dict]:
        """
        Analyze multiple assets and return results.
        
        Args:
            symbols: List of asset symbols
            period: Time period for analysis
            interval: Data interval
        
        Returns:
            Dictionary mapping symbols to their analysis results
        """
        results = {}
        
        for symbol in symbols:
            # Determine if it's crypto or stock based on symbol
            asset_type = 'crypto' if '-USD' in symbol or symbol in ['BTC', 'ETH', 'ADA', 'SOL', 'DOGE'] else 'stock'
            
            analysis = self.analyze_asset(symbol, period, interval, asset_type)
            if analysis:
                results[symbol] = analysis
        
        return results
    
    def compare_assets(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """
        Compare multiple assets and rank them.
        
        Args:
            symbols: List of asset symbols
            period: Time period for analysis
        
        Returns:
            DataFrame with comparison data
        """
        analyses = self.analyze_multiple_assets(symbols, period)
        
        comparison_data = []
        for symbol, analysis in analyses.items():
            if 'recommendation' in analysis:
                rec = analysis['recommendation']
                comparison_data.append({
                    'Symbol': symbol,
                    'Current Price': analysis.get('current_price', 'N/A'),
                    'Recommendation': rec['recommendation'],
                    'Confidence': rec['confidence'],
                    'Risk Level': rec['risk_level'],
                    'Trend': rec['trend'],
                    'Trend Strength': rec['trend_strength'],
                    'Buy Probability': rec['probabilities']['buy'],
                    'Sell Probability': rec['probabilities']['sell']
                })
        
        if not comparison_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison_data)
        # Sort by confidence descending
        df = df.sort_values('Confidence', ascending=False)
        
        return df
    
    def print_analysis(self, analysis: Dict) -> None:
        """
        Print formatted analysis results.
        
        Args:
            analysis: Analysis dictionary from analyze_asset
        """
        if not analysis:
            print("No analysis available")
            return
        
        print(f"\n{'='*60}")
        print(f"MARKET ANALYSIS REPORT")
        print(f"{'='*60}\n")
        
        print(f"Symbol: {analysis['symbol']}")
        print(f"Asset Type: {analysis['asset_type'].upper()}")
        print(f"Current Price: ${analysis['current_price']}")
        print(f"Analysis Date: {analysis['timestamp']}\n")
        
        # Recommendation
        rec = analysis['recommendation']
        print(f"{'='*60}")
        print(f"RECOMMENDATION: {rec['recommendation']}")
        print(f"{'='*60}")
        print(f"Confidence: {rec['confidence']}%")
        print(f"Risk Level: {rec['risk_level']}")
        print(f"\nProbabilities:")
        print(f"  Buy:  {rec['probabilities']['buy']}%")
        print(f"  Hold: {rec['probabilities']['hold']}%")
        print(f"  Sell: {rec['probabilities']['sell']}%\n")
        
        # Trend Analysis
        trend = analysis['trend_analysis']
        print(f"{'='*60}")
        print(f"TREND ANALYSIS")
        print(f"{'='*60}")
        print(f"Overall Trend: {trend['trend']}")
        print(f"Trend Strength: {trend['strength']}%")
        if trend['crossover']:
            print(f"Crossover Signal: {trend['crossover']}")
        
        levels = trend['support_resistance']
        print(f"\nKey Levels:")
        print(f"  Resistance: ${levels['resistance']}")
        print(f"  Current:    ${levels['current']}")
        print(f"  Support:    ${levels['support']}\n")
        
        # Technical Signals
        print(f"{'='*60}")
        print(f"TECHNICAL INDICATORS")
        print(f"{'='*60}")
        for indicator, signal in analysis['technical_signals'].items():
            print(f"\n{indicator}:")
            for key, value in signal.items():
                print(f"  {key}: {value}")
        
        # Volume Analysis
        if 'volume_analysis' in analysis:
            vol = analysis['volume_analysis']
            if 'status' in vol and vol['status'] != 'INSUFFICIENT_DATA':
                print(f"\n{'='*60}")
                print(f"VOLUME ANALYSIS")
                print(f"{'='*60}")
                print(f"Status: {vol['status']}")
                print(f"Current Volume: {vol.get('current_volume', 'N/A'):,}")
                print(f"Average Volume: {vol.get('avg_volume', 'N/A'):,}")
                print(f"Volume Ratio: {vol.get('ratio', 'N/A')}x")
        
        print(f"\n{'='*60}\n")
    
    def get_top_recommendations(self, symbols: List[str], top_n: int = 5, 
                                 recommendation_type: str = 'BUY') -> pd.DataFrame:
        """
        Get top N assets based on recommendation type.
        
        Args:
            symbols: List of asset symbols to analyze
            top_n: Number of top assets to return
            recommendation_type: Type of recommendation ('BUY', 'SELL', or 'HOLD')
        
        Returns:
            DataFrame with top recommendations
        """
        analyses = self.analyze_multiple_assets(symbols)
        
        filtered_data = []
        for symbol, analysis in analyses.items():
            if 'recommendation' in analysis:
                rec = analysis['recommendation']
                if rec['recommendation'] == recommendation_type.upper():
                    filtered_data.append({
                        'Symbol': symbol,
                        'Current Price': analysis.get('current_price', 'N/A'),
                        'Confidence': rec['confidence'],
                        'Risk Level': rec['risk_level'],
                        'Trend': rec['trend'],
                        'Buy Probability': rec['probabilities']['buy']
                    })
        
        if not filtered_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(filtered_data)
        df = df.sort_values('Confidence', ascending=False).head(top_n)
        
        return df
    
    def scan_market(self, symbols: Optional[List[str]] = None, period: str = "3mo", 
                    top_n: int = 5) -> Dict:
        """
        Scan the market for top bullish and bearish recommendations.
        
        This method analyzes multiple assets and returns ranked lists of:
        - Top bullish (BUY) recommendations
        - Top bearish (SELL) recommendations
        
        Args:
            symbols: List of symbols to scan. If None, uses default market watchlist.
            period: Time period for analysis (default: 3mo)
            top_n: Number of top recommendations to return for each category (default: 5)
        
        Returns:
            Dictionary containing:
                - 'bullish': DataFrame of top bullish recommendations
                - 'bearish': DataFrame of top bearish recommendations
                - 'summary': Summary statistics of the scan
        """
        # Default market watchlist if none provided
        if symbols is None:
            symbols = [
                # Tech stocks
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
                # Financial
                'JPM', 'BAC', 'GS', 'V', 'MA',
                # Healthcare
                'JNJ', 'UNH', 'PFE', 'ABBV',
                # Consumer
                'WMT', 'HD', 'MCD', 'NKE', 'SBUX',
                # Energy
                'XOM', 'CVX',
                # Crypto
                'BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD'
            ]
        
        print(f"\n{'='*60}")
        print(f"MARKET SCANNER")
        print(f"{'='*60}")
        print(f"\nScanning {len(symbols)} assets for opportunities...")
        print(f"Period: {period}\n")
        
        # Analyze all symbols
        analyses = self.analyze_multiple_assets(symbols, period)
        
        # Categorize recommendations
        bullish_data = []
        bearish_data = []
        neutral_data = []
        
        for symbol, analysis in analyses.items():
            if 'recommendation' in analysis:
                rec = analysis['recommendation']
                
                data_point = {
                    'Symbol': symbol,
                    'Price': f"${analysis.get('current_price', 'N/A')}",
                    'Recommendation': rec['recommendation'],
                    'Confidence': rec['confidence'],
                    'Risk': rec['risk_level'],
                    'Trend': rec['trend'],
                    'Strength': rec['trend_strength'],
                    'Buy Prob': rec['probabilities']['buy'],
                    'Sell Prob': rec['probabilities']['sell']
                }
                
                if rec['recommendation'] == 'BUY':
                    bullish_data.append(data_point)
                elif rec['recommendation'] == 'SELL':
                    bearish_data.append(data_point)
                else:
                    neutral_data.append(data_point)
        
        # Create DataFrames and sort
        bullish_df = pd.DataFrame(bullish_data) if bullish_data else pd.DataFrame()
        bearish_df = pd.DataFrame(bearish_data) if bearish_data else pd.DataFrame()
        
        # Sort bullish by confidence and buy probability
        if not bullish_df.empty:
            bullish_df = bullish_df.sort_values(
                ['Confidence', 'Buy Prob'], 
                ascending=[False, False]
            ).head(top_n)
        
        # Sort bearish by confidence and sell probability
        if not bearish_df.empty:
            bearish_df = bearish_df.sort_values(
                ['Confidence', 'Sell Prob'], 
                ascending=[False, False]
            ).head(top_n)
        
        # Generate summary statistics
        summary = {
            'total_scanned': len(analyses),
            'bullish_count': len(bullish_data),
            'bearish_count': len(bearish_data),
            'neutral_count': len(neutral_data),
            'bullish_percentage': round(len(bullish_data) / len(analyses) * 100, 1) if analyses else 0,
            'bearish_percentage': round(len(bearish_data) / len(analyses) * 100, 1) if analyses else 0,
            'neutral_percentage': round(len(neutral_data) / len(analyses) * 100, 1) if analyses else 0
        }
        
        return {
            'bullish': bullish_df,
            'bearish': bearish_df,
            'summary': summary
        }
    
    def print_market_scan(self, scan_results: Dict) -> None:
        """
        Print formatted market scan results.
        
        Args:
            scan_results: Results from scan_market()
        """
        summary = scan_results['summary']
        bullish_df = scan_results['bullish']
        bearish_df = scan_results['bearish']
        
        print(f"\n{'='*60}")
        print(f"MARKET SCAN SUMMARY")
        print(f"{'='*60}")
        print(f"Total Assets Scanned: {summary['total_scanned']}")
        print(f"\nMarket Sentiment:")
        print(f"  🟢 Bullish: {summary['bullish_count']} ({summary['bullish_percentage']}%)")
        print(f"  🔴 Bearish: {summary['bearish_count']} ({summary['bearish_percentage']}%)")
        print(f"  ⚪ Neutral:  {summary['neutral_count']} ({summary['neutral_percentage']}%)")
        
        print(f"\n{'='*60}")
        print(f"TOP BULLISH RECOMMENDATIONS 🟢")
        print(f"{'='*60}\n")
        if not bullish_df.empty:
            print(bullish_df.to_string(index=False))
        else:
            print("No strong bullish signals found in current market conditions.")
        
        print(f"\n{'='*60}")
        print(f"TOP BEARISH RECOMMENDATIONS 🔴")
        print(f"{'='*60}\n")
        if not bearish_df.empty:
            print(bearish_df.to_string(index=False))
        else:
            print("No strong bearish signals found in current market conditions.")
        
        print(f"\n{'='*60}")
        print(f"⚠️  DISCLAIMER")
        print(f"{'='*60}")
        print("This analysis is for informational purposes only.")
        print("Always conduct your own research before making investment decisions.")
        print("Past performance does not guarantee future results.")
        print(f"{'='*60}\n")

"""
Market Analyzer - A tool for analyzing financial markets without trading.
This package provides technical analysis and trend prediction capabilities.
"""

from .analyzer import MarketAnalyzer
from .technical_indicators import TechnicalIndicators
from .trend_analyzer import TrendAnalyzer
from .prediction_engine import PredictionEngine

__version__ = "1.0.0"
__all__ = ["MarketAnalyzer", "TechnicalIndicators", "TrendAnalyzer", "PredictionEngine"]

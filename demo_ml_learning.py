#!/usr/bin/env python3
"""
Demo script showing ML learning engine capabilities.
Simulates recording predictions and validating outcomes.
"""

from ml_learning_engine import get_ml_engine
from datetime import datetime, timedelta
import random

def demo_ml_learning():
    """Demonstrate the ML learning system with simulated data."""
    
    print("\n" + "="*80)
    print("🤖 MACHINE LEARNING ENGINE DEMONSTRATION")
    print("="*80 + "\n")
    
    ml_engine = get_ml_engine()
    
    # Show initial state
    print("📊 INITIAL STATE:")
    stats = ml_engine.get_learning_stats()
    print(f"   Learning Level: {stats['learning_level']}")
    print(f"   Total Predictions: {stats['total_predictions']}")
    print(f"   Validated: {stats['validated_predictions']}")
    print()
    
    # Simulate recording predictions
    print("📝 SIMULATING PREDICTIONS...")
    print()
    
    test_assets = [
        ("AAPL", 150.00, "tech stock"),
        ("TSLA", 200.00, "volatile tech"),
        ("BTC-USD", 45000.00, "cryptocurrency"),
        ("MSFT", 380.00, "stable blue chip"),
        ("NVDA", 500.00, "AI leader")
    ]
    
    for symbol, current_price, description in test_assets:
        # Simulate predictions for 30 and 90 days
        predictions = {}
        
        for days in [30, 90]:
            # Simulate realistic prediction with some randomness
            trend = random.choice([1.05, 1.03, 0.97, 0.95])  # +5%, +3%, -3%, -5%
            predicted_price = current_price * trend
            expected_return = ((predicted_price - current_price) / current_price) * 100
            
            predictions[days] = {
                'predicted_price': predicted_price,
                'expected_gain_loss_pct': expected_return,
                'confidence': random.uniform(75, 92),
                'direction': "📈 BULLISH" if expected_return > 0 else "📉 BEARISH",
                'risk_level': random.choice(["LOW", "MEDIUM", "MEDIUM-HIGH"]),
                'models': {
                    'linear': predicted_price * 0.98,
                    'polynomial': predicted_price * 1.01,
                    'ema': predicted_price * 0.99,
                    'mean_reversion': predicted_price * 1.02
                }
            }
        
        # Record the prediction
        pred_date = (datetime.now() - timedelta(days=random.randint(1, 60))).isoformat()
        record = ml_engine.record_prediction(symbol, predictions, current_price, pred_date)
        
        print(f"✅ Recorded: {symbol} ({description})")
        print(f"   Current: ${current_price:.2f}")
        print(f"   30-day: ${predictions[30]['predicted_price']:.2f} ({predictions[30]['expected_gain_loss_pct']:+.1f}%)")
        print(f"   90-day: ${predictions[90]['predicted_price']:.2f} ({predictions[90]['expected_gain_loss_pct']:+.1f}%)")
        print()
    
    print("="*80)
    print()
    
    # Show updated state
    print("📊 AFTER RECORDING PREDICTIONS:")
    stats = ml_engine.get_learning_stats()
    print(f"   Learning Level: {stats['learning_level']}")
    print(f"   Total Predictions: {stats['total_predictions']}")
    print(f"   Validated: {stats['validated_predictions']}")
    print()
    
    # Simulate validating some predictions
    print("🔄 SIMULATING OUTCOME VALIDATION...")
    print()
    
    # Update with simulated actual prices (within reasonable range)
    for symbol, original_price, description in test_assets[:3]:  # Validate first 3
        # Simulate actual outcome (close to prediction with some error)
        actual_price = original_price * random.uniform(0.95, 1.08)
        
        updated = ml_engine.update_with_actual_outcome(
            symbol,
            datetime.now().isoformat(),
            actual_price
        )
        
        if updated:
            print(f"✅ Updated: {symbol}")
            print(f"   Original: ${original_price:.2f}")
            print(f"   Actual: ${actual_price:.2f}")
            print(f"   Change: {((actual_price - original_price) / original_price * 100):+.2f}%")
            print()
    
    print("="*80)
    print()
    
    # Show final state with performance metrics
    print("📈 FINAL STATE WITH METRICS:")
    stats = ml_engine.get_learning_stats()
    print(f"   Learning Level: {stats['learning_level']}")
    print(f"   Total Predictions: {stats['total_predictions']}")
    print(f"   Validated: {stats['validated_predictions']}")
    print()
    
    # Show timeframe performance if available
    if stats['timeframe_performance']:
        print("📊 PERFORMANCE BY TIMEFRAME:")
        for tf, perf in stats['timeframe_performance'].items():
            print(f"\n   {tf}-day predictions:")
            print(f"      Count: {perf['predictions_count']}")
            print(f"      Avg Error: {perf['avg_error_pct']:.2f}%")
            print(f"      Direction Accuracy: {perf['direction_accuracy']:.1f}%")
            print(f"      Avg Confidence: {perf['avg_confidence']:.1f}%")
    
    # Show model performance if available
    if stats['model_performance']:
        print("\n🎯 MODEL PERFORMANCE:")
        for model, perf in stats['model_performance'].items():
            print(f"\n   {model.title()}:")
            print(f"      Predictions: {perf['predictions_count']}")
            print(f"      Avg Error: {perf['avg_error_pct']:.2f}%")
            print(f"      RMSE: {perf['rmse']:.2f}")
    
    print("\n" + "="*80)
    print()
    
    # Demonstrate adaptive weights
    print("⚙️  ADAPTIVE MODEL WEIGHTS:")
    weights = ml_engine.get_adaptive_model_weights(90)
    print(f"   90-day predictions will use:")
    for model, weight in weights.items():
        print(f"      {model.ljust(20)}: {weight*100:.1f}%")
    print()
    
    # Demonstrate confidence adjustment
    print("🎯 CONFIDENCE ADJUSTMENT DEMO:")
    test_confidence = 85.0
    adjusted, reason = ml_engine.get_confidence_adjustment(test_confidence, 90)
    print(f"   Base Confidence: {test_confidence:.1f}%")
    print(f"   Adjusted Confidence: {adjusted:.1f}%")
    print(f"   Reason: {reason}")
    print()
    
    print("="*80)
    print()
    print("✅ DEMONSTRATION COMPLETE!")
    print()
    print("💡 Key Takeaways:")
    print("   • System tracks predictions and validates outcomes")
    print("   • Confidence adjusts based on historical accuracy")
    print("   • Model weights adapt to performance")
    print("   • Learning improves with more data")
    print("   • Progress tracked from BEGINNER to EXPERT")
    print()
    print("🚀 Use the web app to record real predictions and build your ML learning data!")
    print("="*80 + "\n")

if __name__ == "__main__":
    demo_ml_learning()

#!/usr/bin/env python3
"""
Machine Learning Engine for adaptive prediction improvement.
Tracks prediction accuracy and optimizes model performance over time.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict


class MLLearningEngine:
    """
    Learns from prediction history to improve future predictions.
    Tracks actual vs predicted outcomes and adapts model confidence.
    """
    
    def __init__(self, storage_path: str = "ml_prediction_history.json"):
        """Initialize the ML learning engine."""
        self.storage_path = storage_path
        self.prediction_history = self._load_history()
        self.model_performance = self._calculate_model_performance()
        
    def _load_history(self) -> Dict:
        """Load prediction history from storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load prediction history: {e}")
                return {"predictions": [], "metadata": {"version": "1.0"}}
        return {"predictions": [], "metadata": {"version": "1.0"}}
    
    def _save_history(self):
        """Save prediction history to storage."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.prediction_history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save prediction history: {e}")
    
    def record_prediction(self, symbol: str, prediction_data: Dict, 
                         current_price: float, prediction_date: str = None):
        """
        Record a new prediction for future validation.
        
        Args:
            symbol: Asset symbol
            prediction_data: Prediction results including timeframes
            current_price: Current price at prediction time
            prediction_date: Date of prediction (default: now)
        """
        if prediction_date is None:
            prediction_date = datetime.now().isoformat()
        
        record = {
            "symbol": symbol,
            "prediction_date": prediction_date,
            "current_price": current_price,
            "timeframes": {},
            "validated": False
        }
        
        # Store predictions for each timeframe
        for days, pred in prediction_data.items():
            if isinstance(days, int):
                target_date = (datetime.fromisoformat(prediction_date) + 
                             timedelta(days=days)).isoformat()
                record["timeframes"][str(days)] = {
                    "predicted_price": pred.get('predicted_price'),
                    "expected_return_pct": pred.get('expected_gain_loss_pct'),
                    "confidence": pred.get('confidence'),
                    "direction": pred.get('direction'),
                    "target_date": target_date,
                    "models": pred.get('models', {}),
                    "actual_price": None,
                    "actual_return_pct": None,
                    "error_pct": None
                }
        
        self.prediction_history["predictions"].append(record)
        self._save_history()
        
        return record
    
    def update_with_actual_outcome(self, symbol: str, date: str, actual_price: float):
        """
        Update predictions with actual outcomes for validation.
        
        Args:
            symbol: Asset symbol
            date: Current date
            actual_price: Actual price observed
        """
        current_date = datetime.fromisoformat(date)
        updated = False
        
        for pred_record in self.prediction_history["predictions"]:
            if pred_record["symbol"] != symbol or pred_record["validated"]:
                continue
            
            for timeframe, data in pred_record["timeframes"].items():
                target_date = datetime.fromisoformat(data["target_date"])
                
                # Check if we've reached or passed the target date
                if current_date >= target_date and data["actual_price"] is None:
                    data["actual_price"] = actual_price
                    data["actual_return_pct"] = ((actual_price - pred_record["current_price"]) 
                                                / pred_record["current_price"] * 100)
                    
                    # Calculate prediction error
                    predicted_price = data["predicted_price"]
                    data["error_pct"] = abs((actual_price - predicted_price) / actual_price * 100)
                    
                    # Calculate directional accuracy
                    predicted_direction = "UP" if data["expected_return_pct"] > 0 else "DOWN"
                    actual_direction = "UP" if data["actual_return_pct"] > 0 else "DOWN"
                    data["direction_correct"] = (predicted_direction == actual_direction)
                    
                    updated = True
            
            # Mark as validated if all timeframes are complete
            all_validated = all(
                data.get("actual_price") is not None 
                for data in pred_record["timeframes"].values()
            )
            if all_validated:
                pred_record["validated"] = True
        
        if updated:
            self._save_history()
            self.model_performance = self._calculate_model_performance()
        
        return updated
    
    def _calculate_model_performance(self) -> Dict:
        """Calculate performance metrics for each model type."""
        model_stats = defaultdict(lambda: {
            "predictions": 0,
            "total_error": 0,
            "direction_correct": 0,
            "rmse_sum": 0
        })
        
        timeframe_stats = defaultdict(lambda: {
            "predictions": 0,
            "total_error": 0,
            "direction_correct": 0,
            "avg_confidence": 0,
            "confidence_calibration": []
        })
        
        for pred_record in self.prediction_history["predictions"]:
            for timeframe, data in pred_record["timeframes"].items():
                if data.get("actual_price") is None:
                    continue
                
                # Overall timeframe stats
                tf_stats = timeframe_stats[timeframe]
                tf_stats["predictions"] += 1
                tf_stats["total_error"] += data["error_pct"]
                tf_stats["direction_correct"] += (1 if data.get("direction_correct") else 0)
                tf_stats["avg_confidence"] += data["confidence"]
                
                # Confidence calibration (how well confidence matches accuracy)
                accuracy = 100 - data["error_pct"]
                tf_stats["confidence_calibration"].append({
                    "confidence": data["confidence"],
                    "accuracy": accuracy
                })
                
                # Individual model performance
                if "models" in data and data["models"]:
                    actual_price = data["actual_price"]
                    for model_name, model_pred in data["models"].items():
                        if model_pred and model_name != "bounded":
                            stats = model_stats[model_name]
                            stats["predictions"] += 1
                            error = abs((actual_price - model_pred) / actual_price * 100)
                            stats["total_error"] += error
                            stats["rmse_sum"] += error ** 2
        
        # Calculate averages
        performance = {
            "models": {},
            "timeframes": {},
            "overall": {
                "total_predictions": 0,
                "validated_predictions": 0,
                "learning_level": "BEGINNER"
            }
        }
        
        for model, stats in model_stats.items():
            if stats["predictions"] > 0:
                performance["models"][model] = {
                    "avg_error_pct": stats["total_error"] / stats["predictions"],
                    "rmse": np.sqrt(stats["rmse_sum"] / stats["predictions"]),
                    "predictions_count": stats["predictions"]
                }
        
        for timeframe, stats in timeframe_stats.items():
            if stats["predictions"] > 0:
                avg_error = stats["total_error"] / stats["predictions"]
                direction_accuracy = (stats["direction_correct"] / stats["predictions"]) * 100
                avg_conf = stats["avg_confidence"] / stats["predictions"]
                
                # Calculate confidence calibration score
                calibration_error = 0
                if stats["confidence_calibration"]:
                    for cal in stats["confidence_calibration"]:
                        calibration_error += abs(cal["confidence"] - cal["accuracy"])
                    calibration_error /= len(stats["confidence_calibration"])
                
                performance["timeframes"][timeframe] = {
                    "avg_error_pct": avg_error,
                    "direction_accuracy": direction_accuracy,
                    "avg_confidence": avg_conf,
                    "calibration_error": calibration_error,
                    "predictions_count": stats["predictions"]
                }
        
        # Overall metrics
        total_preds = len(self.prediction_history["predictions"])
        validated = sum(1 for p in self.prediction_history["predictions"] if p["validated"])
        
        performance["overall"]["total_predictions"] = total_preds
        performance["overall"]["validated_predictions"] = validated
        
        # Determine learning level
        if validated >= 100:
            performance["overall"]["learning_level"] = "EXPERT"
        elif validated >= 50:
            performance["overall"]["learning_level"] = "ADVANCED"
        elif validated >= 20:
            performance["overall"]["learning_level"] = "INTERMEDIATE"
        elif validated >= 5:
            performance["overall"]["learning_level"] = "LEARNING"
        else:
            performance["overall"]["learning_level"] = "BEGINNER"
        
        return performance
    
    def get_adaptive_model_weights(self, timeframe: int = 90) -> Dict[str, float]:
        """
        Calculate optimized model weights based on historical performance.
        
        Args:
            timeframe: Prediction timeframe in days
            
        Returns:
            Dictionary of model weights
        """
        if not self.model_performance["models"]:
            # Default weights if no learning data
            return {
                "linear": 0.28,
                "polynomial": 0.32,
                "ema": 0.25,
                "mean_reversion": 0.15
            }
        
        # Calculate inverse error weights (lower error = higher weight)
        model_scores = {}
        for model, stats in self.model_performance["models"].items():
            if stats["predictions_count"] >= 3:  # Require minimum data
                # Score is inverse of error (lower error = better score)
                score = 1 / (1 + stats["avg_error_pct"])
                model_scores[model] = score
        
        if not model_scores:
            # Not enough data, use defaults
            return {
                "linear": 0.28,
                "polynomial": 0.32,
                "ema": 0.25,
                "mean_reversion": 0.15
            }
        
        # Normalize scores to weights
        total_score = sum(model_scores.values())
        weights = {model: score / total_score for model, score in model_scores.items()}
        
        # Map to expected model names
        weight_map = {
            "linear": weights.get("linear", 0.28),
            "polynomial": weights.get("polynomial", 0.32),
            "ema": weights.get("ema", 0.25),
            "mean_reversion": weights.get("mean_reversion", 0.15)
        }
        
        # Ensure all weights present
        if sum(weight_map.values()) == 0:
            return {
                "linear": 0.28,
                "polynomial": 0.32,
                "ema": 0.25,
                "mean_reversion": 0.15
            }
        
        # Normalize
        total = sum(weight_map.values())
        return {k: v / total for k, v in weight_map.items()}
    
    def get_confidence_adjustment(self, base_confidence: float, 
                                  timeframe: int = 90) -> Tuple[float, str]:
        """
        Adjust confidence based on historical accuracy.
        
        Args:
            base_confidence: Initial confidence score
            timeframe: Prediction timeframe
            
        Returns:
            Tuple of (adjusted_confidence, reason)
        """
        tf_key = str(timeframe)
        
        if tf_key not in self.model_performance["timeframes"]:
            return base_confidence, "No historical data"
        
        tf_perf = self.model_performance["timeframes"][tf_key]
        
        if tf_perf["predictions_count"] < 5:
            return base_confidence, "Insufficient learning data"
        
        # Calculate adjustment based on historical accuracy
        avg_error = tf_perf["avg_error_pct"]
        direction_accuracy = tf_perf["direction_accuracy"]
        calibration_error = tf_perf["calibration_error"]
        
        adjustment = 0
        reason_parts = []
        
        # Adjust based on prediction error
        if avg_error < 5:
            adjustment += 5
            reason_parts.append("excellent accuracy")
        elif avg_error < 10:
            adjustment += 3
            reason_parts.append("good accuracy")
        elif avg_error < 15:
            adjustment += 0
        elif avg_error < 25:
            adjustment -= 3
            reason_parts.append("moderate errors")
        else:
            adjustment -= 5
            reason_parts.append("high errors")
        
        # Adjust based on directional accuracy
        if direction_accuracy > 80:
            adjustment += 3
            reason_parts.append("strong directional accuracy")
        elif direction_accuracy > 60:
            adjustment += 1
        elif direction_accuracy < 40:
            adjustment -= 3
            reason_parts.append("poor directional accuracy")
        
        # Adjust based on confidence calibration
        if calibration_error < 10:
            adjustment += 2
            reason_parts.append("well-calibrated")
        elif calibration_error > 20:
            adjustment -= 2
            reason_parts.append("calibration issues")
        
        # Learning level bonus
        learning_level = self.model_performance["overall"]["learning_level"]
        if learning_level == "EXPERT":
            adjustment += 3
            reason_parts.append("expert learning level")
        elif learning_level == "ADVANCED":
            adjustment += 2
            reason_parts.append("advanced learning level")
        elif learning_level == "INTERMEDIATE":
            adjustment += 1
            reason_parts.append("intermediate learning level")
        
        adjusted_confidence = np.clip(base_confidence + adjustment, 35, 96)
        reason = f"ML-adjusted: {', '.join(reason_parts) if reason_parts else 'learning in progress'}"
        
        return adjusted_confidence, reason
    
    def get_learning_stats(self) -> Dict:
        """Get comprehensive learning statistics."""
        stats = {
            "total_predictions": self.model_performance["overall"]["total_predictions"],
            "validated_predictions": self.model_performance["overall"]["validated_predictions"],
            "learning_level": self.model_performance["overall"]["learning_level"],
            "model_performance": self.model_performance["models"],
            "timeframe_performance": self.model_performance["timeframes"],
            "improvement_suggestions": []
        }
        
        # Generate improvement suggestions
        if stats["validated_predictions"] < 10:
            stats["improvement_suggestions"].append(
                "Continue making predictions to build learning data"
            )
        
        for tf, perf in self.model_performance["timeframes"].items():
            if perf["predictions_count"] >= 5:
                if perf["avg_error_pct"] > 15:
                    stats["improvement_suggestions"].append(
                        f"{tf}-day predictions show high error - consider adjusting strategy"
                    )
                if perf["direction_accuracy"] < 60:
                    stats["improvement_suggestions"].append(
                        f"{tf}-day directional accuracy needs improvement"
                    )
        
        return stats
    
    def reset_learning(self):
        """Reset all learning data (use with caution)."""
        self.prediction_history = {"predictions": [], "metadata": {"version": "1.0"}}
        self.model_performance = {}
        self._save_history()
        
    def get_similar_predictions(self, symbol: str, days: int = 90, limit: int = 5) -> List[Dict]:
        """
        Find similar past predictions for reference.
        
        Args:
            symbol: Asset symbol
            days: Timeframe
            limit: Maximum number of results
            
        Returns:
            List of similar predictions with outcomes
        """
        similar = []
        
        for pred_record in self.prediction_history["predictions"]:
            if pred_record["symbol"] != symbol:
                continue
            
            tf_key = str(days)
            if tf_key in pred_record["timeframes"]:
                tf_data = pred_record["timeframes"][tf_key]
                if tf_data.get("actual_price") is not None:
                    similar.append({
                        "prediction_date": pred_record["prediction_date"],
                        "predicted_price": tf_data["predicted_price"],
                        "actual_price": tf_data["actual_price"],
                        "error_pct": tf_data["error_pct"],
                        "direction_correct": tf_data.get("direction_correct"),
                        "confidence": tf_data["confidence"]
                    })
        
        # Sort by date (most recent first)
        similar.sort(key=lambda x: x["prediction_date"], reverse=True)
        
        return similar[:limit]


# Singleton instance
_ml_engine = None

def get_ml_engine() -> MLLearningEngine:
    """Get or create the ML learning engine singleton."""
    global _ml_engine
    if _ml_engine is None:
        _ml_engine = MLLearningEngine()
    return _ml_engine

import os
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score

from agents.transformer import TransformerAgent
from agents.common.models import MarketData, Signal, Prediction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TransformerEvaluator:
    """Evaluator for transformer model performance."""

    def __init__(self, model_path: str, results_dir: str):
        """Initialize the evaluator.

        Args:
            model_path: Path to the trained model
            results_dir: Directory to save evaluation results
        """
        self.model_path = model_path
        self.results_dir = results_dir
        self.agent = TransformerAgent()

        # Load the model
        if not self.agent.load_model(model_path):
            raise ValueError(f"Failed to load model from {model_path}")

        # Initialize metrics
        self.metrics = {
            'mse': [],
            'rmse': [],
            'r2': [],
            'direction_accuracy': [],
            'signal_accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'signals': [],
            'signal_strengths': []
        }

        # Initialize performance history
        self.performance_history = {
            'timestamp': [],
            'mse': [],
            'rmse': [],
            'r2': [],
            'direction_accuracy': [],
            'signal_accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'signal_count': [],
            'avg_signal_strength': []
        }

    def evaluate(self, test_data: List[MarketData]) -> Dict:
        """Evaluate model performance on test data.

        Args:
            test_data: List of test market data points

        Returns:
            Dict: Evaluation metrics
        """
        logger.info("Starting model evaluation...")

        predictions = []
        actual_prices = []
        signals = []

        for data in test_data:
            # Generate prediction
            prediction = self.agent.predict(data)
            if prediction is None:
                continue

            # Generate signal
            signal = self.agent.generate_signal(data)
            if signal is not None:
                signals.append(signal)

            predictions.append(prediction.predicted_price)
            actual_prices.append(data.close)

        if not predictions:
            logger.error("No predictions generated")
            return {}

        # Calculate metrics
        predictions = np.array(predictions)
        actual_prices = np.array(actual_prices)

        # Price prediction metrics
        mse = mean_squared_error(actual_prices, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_prices, predictions)

        # Direction accuracy
        actual_direction = np.sign(np.diff(actual_prices))
        pred_direction = np.sign(np.diff(predictions))
        direction_accuracy = np.mean(actual_direction == pred_direction)

        # Signal metrics
        if signals:
            signal_directions = [s.direction for s in signals]
            signal_strengths = [s.strength for s in signals]

            # Calculate signal accuracy
            correct_signals = sum(1 for s in signals if (
                (s.direction == 1 and s.metadata['predicted_price'] > s.metadata['current_price']) or
                (s.direction == -1 and s.metadata['predicted_price'] < s.metadata['current_price'])
            ))
            signal_accuracy = correct_signals / len(signals)

            # Calculate precision, recall, and F1
            y_true = [1 if s.direction != 0 else 0 for s in signals]
            y_pred = [1 if s.direction != 0 else 0 for s in signals]

            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        else:
            signal_accuracy = 0
            precision = 0
            recall = 0
            f1 = 0
            signal_directions = []
            signal_strengths = []

        # Update metrics
        self.metrics['mse'].append(mse)
        self.metrics['rmse'].append(rmse)
        self.metrics['r2'].append(r2)
        self.metrics['direction_accuracy'].append(direction_accuracy)
        self.metrics['signal_accuracy'].append(signal_accuracy)
        self.metrics['precision'].append(precision)
        self.metrics['recall'].append(recall)
        self.metrics['f1'].append(f1)
        self.metrics['signals'].extend(signal_directions)
        self.metrics['signal_strengths'].extend(signal_strengths)

        # Update performance history
        current_time = datetime.now()
        self.performance_history['timestamp'].append(current_time)
        self.performance_history['mse'].append(mse)
        self.performance_history['rmse'].append(rmse)
        self.performance_history['r2'].append(r2)
        self.performance_history['direction_accuracy'].append(direction_accuracy)
        self.performance_history['signal_accuracy'].append(signal_accuracy)
        self.performance_history['precision'].append(precision)
        self.performance_history['recall'].append(recall)
        self.performance_history['f1'].append(f1)
        self.performance_history['signal_count'].append(len(signals))
        self.performance_history['avg_signal_strength'].append(
            np.mean(signal_strengths) if signal_strengths else 0
        )

        # Log results
        logger.info(f"Evaluation completed:")
        logger.info(f"📊 MSE: {mse:.4f}")
        logger.info(f"📊 RMSE: {rmse:.4f}")
        logger.info(f"📊 R²: {r2:.4f}")
        logger.info(f"📊 Direction Accuracy: {direction_accuracy:.4f}")
        logger.info(f"📊 Signal Accuracy: {signal_accuracy:.4f}")
        logger.info(f"📊 Precision: {precision:.4f}")
        logger.info(f"📊 Recall: {recall:.4f}")
        logger.info(f"📊 F1 Score: {f1:.4f}")
        logger.info(f"📊 Total Signals: {len(signals)}")

        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'signal_accuracy': signal_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'signal_count': len(signals)
        }

    def plot_results(self):
        """Generate and save evaluation plots."""
        logger.info("Generating evaluation plots...")

        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)

        # Plot predictions vs actual
        plt.figure(figsize=(12, 6))
        plt.plot(self.metrics['rmse'], label='RMSE')
        plt.plot(self.metrics['r2'], label='R²')
        plt.title('Model Performance Metrics')
        plt.xlabel('Evaluation Step')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(os.path.join(self.results_dir, 'performance_metrics.png'))
        plt.close()

        # Plot signal distribution
        if self.metrics['signals']:
            plt.figure(figsize=(10, 6))
            plt.hist(self.metrics['signals'], bins=3, alpha=0.7)
            plt.title('Signal Distribution')
            plt.xlabel('Signal Direction')
            plt.ylabel('Count')
            plt.xticks([-1, 0, 1], ['Sell', 'Hold', 'Buy'])
            plt.savefig(os.path.join(self.results_dir, 'signal_distribution.png'))
            plt.close()

        # Plot signal strengths
        if self.metrics['signal_strengths']:
            plt.figure(figsize=(10, 6))
            plt.hist(self.metrics['signal_strengths'], bins=20, alpha=0.7)
            plt.title('Signal Strength Distribution')
            plt.xlabel('Signal Strength')
            plt.ylabel('Count')
            plt.savefig(os.path.join(self.results_dir, 'signal_strengths.png'))
            plt.close()

        # Plot performance history
        plt.figure(figsize=(15, 10))

        # Plot accuracy metrics
        plt.subplot(2, 2, 1)
        plt.plot(self.performance_history['timestamp'], self.performance_history['direction_accuracy'], label='Direction')
        plt.plot(self.performance_history['timestamp'], self.performance_history['signal_accuracy'], label='Signal')
        plt.title('Accuracy Metrics Over Time')
        plt.xlabel('Time')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot precision, recall, F1
        plt.subplot(2, 2, 2)
        plt.plot(self.performance_history['timestamp'], self.performance_history['precision'], label='Precision')
        plt.plot(self.performance_history['timestamp'], self.performance_history['recall'], label='Recall')
        plt.plot(self.performance_history['timestamp'], self.performance_history['f1'], label='F1')
        plt.title('Classification Metrics Over Time')
        plt.xlabel('Time')
        plt.ylabel('Score')
        plt.legend()

        # Plot signal count
        plt.subplot(2, 2, 3)
        plt.plot(self.performance_history['timestamp'], self.performance_history['signal_count'])
        plt.title('Signal Count Over Time')
        plt.xlabel('Time')
        plt.ylabel('Count')

        # Plot average signal strength
        plt.subplot(2, 2, 4)
        plt.plot(self.performance_history['timestamp'], self.performance_history['avg_signal_strength'])
        plt.title('Average Signal Strength Over Time')
        plt.xlabel('Time')
        plt.ylabel('Strength')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'performance_history.png'))
        plt.close()

        logger.info(f"Plots saved to {self.results_dir}")

    def save_metrics(self):
        """Save evaluation metrics to file."""
        metrics_file = os.path.join(self.results_dir, 'evaluation_metrics.json')

        # Calculate average metrics
        avg_metrics = {
            'mse': np.mean(self.metrics['mse']),
            'rmse': np.mean(self.metrics['rmse']),
            'r2': np.mean(self.metrics['r2']),
            'direction_accuracy': np.mean(self.metrics['direction_accuracy']),
            'signal_accuracy': np.mean(self.metrics['signal_accuracy']),
            'precision': np.mean(self.metrics['precision']),
            'recall': np.mean(self.metrics['recall']),
            'f1': np.mean(self.metrics['f1']),
            'total_signals': len(self.metrics['signals']),
            'avg_signal_strength': np.mean(self.metrics['signal_strengths']) if self.metrics['signal_strengths'] else 0
        }

        # Save metrics
        with open(metrics_file, 'w') as f:
            json.dump(avg_metrics, f, indent=4)

        # Save performance history
        history_file = os.path.join(self.results_dir, 'performance_history.json')
        with open(history_file, 'w') as f:
            json.dump(self.performance_history, f, indent=4)

        logger.info(f"Metrics saved to {metrics_file}")
        logger.info(f"Performance history saved to {history_file}")

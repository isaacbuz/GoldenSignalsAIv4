#!/usr/bin/env python3
"""
Transformer Model Evaluation Pipeline
Runs comprehensive evaluation of transformer model performance
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from typing import List, Optional

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ml_training.evaluate_transformer import TransformerEvaluator
from ml_training.load_test_data import load_historical_data

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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run transformer model evaluation')
    
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        default=['AAPL', 'MSFT', 'GOOGL'],
        help='Stock symbols to evaluate'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
        help='Start date for evaluation (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date for evaluation (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--interval',
        type=str,
        default='1d',
        help='Data interval (e.g., 1d, 1h, 15m)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/transformer',
        help='Path to the trained model'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='ml_training/metrics',
        help='Directory to save evaluation results'
    )
    
    return parser.parse_args()

def main():
    """Main evaluation script."""
    # Parse arguments
    args = parse_args()
    
    try:
        # Convert start_date and end_date to datetime objects
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

        # Load test data
        logger.info("Loading test data...")
        train_data, test_data = load_historical_data(
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
            interval=args.interval
        )
        
        if test_data is None:
            logger.error("Failed to load test data")
            return
            
        # Initialize evaluator
        logger.info("Initializing evaluator...")
        evaluator = TransformerEvaluator(
            model_path=args.model_path,
            results_dir=args.results_dir
        )
        
        # Run evaluation
        logger.info("Running evaluation...")
        metrics = evaluator.evaluate(test_data)
        
        # Generate plots
        logger.info("Generating plots...")
        evaluator.plot_results()
        
        # Save metrics
        logger.info("Saving metrics...")
        evaluator.save_metrics()
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
# Live Transformer Model Evaluation

This system provides real-time evaluation of the transformer model's performance using live market data.

## Features

- Real-time market data streaming from multiple sources (Yahoo Finance, Polygon.io)
- Live prediction generation and performance tracking
- Comprehensive metrics calculation (MSE, RMSE, R², Direction Accuracy)
- Automatic visualization generation
- Performance history tracking
- Configurable evaluation parameters

## Setup

1. Run the setup script:
```bash
./scripts/setup_live_evaluation.sh
```

2. (Optional) Set up Polygon.io API key:
```bash
export POLYGON_API_KEY='your_api_key_here'
```

## Configuration

The system is configured through `config/live_evaluation.yaml`. Key settings include:

- Data sources and update intervals
- Symbols to track
- Model parameters
- Evaluation settings
- Logging configuration

## Running the Evaluation

Start the live evaluation:
```bash
python ml_training/evaluate_live_transformer.py
```

The script will:
1. Load the transformer model
2. Connect to available data sources
3. Start streaming live market data
4. Generate predictions and calculate metrics
5. Create visualizations and save results

## Output

Results are saved in the following locations:

- Metrics: `ml_training/metrics/transformer_live_{symbol}_metrics.json`
- Plots: `ml_training/metrics/transformer_live_{symbol}.png`
- Logs: `logs/live_evaluation.log`

## Monitoring

The system provides real-time logging of:
- Market data updates
- Model predictions
- Performance metrics
- System status

## Troubleshooting

1. If you encounter connection issues:
   - Check your internet connection
   - Verify API keys are set correctly
   - Check the logs for detailed error messages

2. If the model fails to load:
   - Ensure the model checkpoint exists in `models/transformer/`
   - Check the model architecture matches the saved weights

3. If data sources are unavailable:
   - The system will automatically fall back to available sources
   - Check the logs for source-specific errors

## Performance Considerations

- The system maintains a rolling window of predictions for evaluation
- Plots are generated periodically to minimize resource usage
- Metrics are saved at configurable intervals
- The system automatically handles data source failures and reconnections 
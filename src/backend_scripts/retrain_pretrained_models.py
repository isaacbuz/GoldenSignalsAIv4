import os
import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

# Example retraining script for LSTM, GRU, CNN, Attention models
# Assumes you have new data in CSV format for each symbol
# Place your new data in ./data/<symbol>.csv

MODELS = {
    "lstm": "external/Stock-Prediction-Models/deep-learning/model/lstm_model.h5",
    "gru": "external/Stock-Prediction-Models/deep-learning/model/gru_model.h5",
    "cnn": "external/Stock-Prediction-Models/deep-learning/model/cnn_model.h5",
    "attention": "external/Stock-Prediction-Models/deep-learning/model/attention_model.h5"
}

WINDOW = 60

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))

if not os.path.exists(data_dir):
    print(f"Data directory {data_dir} does not exist. Please create it and add your CSV files.")
    exit(1)

for symbol in os.listdir(data_dir):
    if not symbol.endswith(".csv"): continue
    symbol_name = symbol.replace(".csv", "")
    data = np.genfromtxt(os.path.join(data_dir, symbol), delimiter=",")
    X = np.array([data[i:i+WINDOW] for i in range(len(data)-WINDOW)])
    X = X.reshape((-1, WINDOW, 1))
    y = data[WINDOW:]
    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found at {model_path}")
            continue
        model = load_model(model_path)
        checkpoint = ModelCheckpoint(model_path, save_best_only=True)
        print(f"Retraining {model_name} for {symbol_name}")
        model.fit(X, y, epochs=5, batch_size=16, callbacks=[checkpoint])
        print(f"Finished retraining {model_name} for {symbol_name}")

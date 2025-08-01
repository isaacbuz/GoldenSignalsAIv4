"""
LSTM Agent for Time Series Prediction
Specialized agent for LSTM-based price prediction
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.ml.models.advanced_ml_models import LSTMPricePredictor

from .base_ml_agent import AgentStatus, BaseMLAgent, ModelPrediction


class LSTMAgent(BaseMLAgent):
    """LSTM-based prediction agent"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, "lstm", config)
        self.sequence_length = config.get('sequence_length', 50)
        self.hidden_size = config.get('hidden_size', 128)
        self.num_layers = config.get('num_layers', 2)
        self.feature_size = config.get('feature_size', 10)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    async def load_model(self):
        """Load or initialize LSTM model"""
        try:
            # Try to load existing model
            model_path = f"models/lstm_{self.agent_id}.pth"
            self.model = LSTMPricePredictor(
                input_size=self.feature_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers
            ).to(self.device)

            try:
                self.model.load_state_dict(torch.load(model_path))
                self.logger.info(f"Loaded existing LSTM model from {model_path}")
            except FileNotFoundError:
                self.logger.info("No existing model found, initialized new LSTM model")

        except Exception as e:
            self.logger.error(f"Error loading LSTM model: {e}")
            self.status = AgentStatus.ERROR

    def prepare_sequences(self, data: pd.DataFrame) -> torch.Tensor:
        """Prepare sequences for LSTM"""
        sequences = []

        for i in range(len(data) - self.sequence_length):
            seq = data.iloc[i:i+self.sequence_length].values
            sequences.append(seq)

        return torch.FloatTensor(sequences).to(self.device)

    async def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Train LSTM model"""
        self.status = AgentStatus.TRAINING

        try:
            # Prepare data
            X = self.prepare_sequences(data.iloc[:-1])
            y = torch.FloatTensor(data['close'].iloc[self.sequence_length:].values).to(self.device)

            # Training parameters
            epochs = kwargs.get('epochs', 100)
            learning_rate = kwargs.get('learning_rate', 0.001)
            batch_size = kwargs.get('batch_size', 32)

            # Setup training
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

            # Training loop
            train_losses = []
            self.model.train()

            for epoch in range(epochs):
                epoch_loss = 0

                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i+batch_size]
                    batch_y = y[i:i+batch_size]

                    optimizer.zero_grad()
                    outputs = self.model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / (len(X) // batch_size)
                train_losses.append(avg_loss)

                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

            # Save model
            torch.save(self.model.state_dict(), f"models/lstm_{self.agent_id}.pth")

            # Calculate final metrics
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X).squeeze()
                mse = criterion(predictions, y).item()
                rmse = np.sqrt(mse)

            results = {
                'final_loss': train_losses[-1],
                'rmse': rmse,
                'epochs_trained': epochs,
                'training_history': train_losses
            }

            self.last_accuracy = 1 / (1 + rmse)  # Convert RMSE to accuracy-like metric

            return results

        except Exception as e:
            self.logger.error(f"Training error: {e}")
            return {'error': str(e)}
        finally:
            self.status = AgentStatus.IDLE

    async def predict(self, features: pd.DataFrame, **kwargs) -> ModelPrediction:
        """Make predictions using LSTM"""
        self.status = AgentStatus.PREDICTING

        try:
            self.model.eval()

            # Prepare input
            X = self.prepare_sequences(features)

            with torch.no_grad():
                # Get prediction and attention weights
                prediction, attention_weights = self.model(X[-1:])
                pred_value = prediction.cpu().numpy()[0, 0]

                # Calculate confidence based on attention weights
                attention_std = attention_weights.std().item()
                confidence = 1 / (1 + attention_std)  # Lower std = higher confidence

            # Prepare metadata
            metadata = {
                'sequence_length': self.sequence_length,
                'attention_focus': self._analyze_attention(attention_weights),
                'model_version': self.model_version,
                'device': str(self.device)
            }

            return ModelPrediction(
                value=float(pred_value),
                confidence=confidence,
                metadata=metadata,
                timestamp=datetime.now(),
                agent_id=self.agent_id,
                model_version=self.model_version
            )

        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            raise
        finally:
            self.status = AgentStatus.IDLE

    def _analyze_attention(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """Analyze attention weights to understand model focus"""
        weights = attention_weights.cpu().numpy()

        # Find top attended time steps
        top_indices = np.argsort(weights.mean(axis=0))[-5:]

        return {
            'top_attended_steps': top_indices.tolist(),
            'attention_entropy': -np.sum(weights * np.log(weights + 1e-10)),
            'max_attention': float(weights.max())
        }

    async def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            self.model.eval()

            X = self.prepare_sequences(test_data.iloc[:-1])
            y_true = test_data['close'].iloc[self.sequence_length:].values

            with torch.no_grad():
                predictions = self.model(X).squeeze().cpu().numpy()

            # Calculate metrics
            mse = np.mean((predictions - y_true) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - y_true))

            # Directional accuracy
            y_true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(predictions))
            directional_accuracy = np.mean(y_true_direction == pred_direction)

            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'directional_accuracy': directional_accuracy
            }

        except Exception as e:
            self.logger.error(f"Evaluation error: {e}")
            return {'error': str(e)}

    async def handle_collaboration_request(self, message):
        """Handle collaboration with other agents"""
        collab_type = message.payload.get('type')

        if collab_type == 'ensemble':
            # Contribute LSTM predictions to ensemble
            features = pd.DataFrame(message.payload['features'])
            prediction = await self.predict(features)

            response = {
                'agent_id': self.agent_id,
                'prediction': prediction.value,
                'confidence': prediction.confidence,
                'model_type': 'lstm',
                'specialization': 'time_series'
            }

            await self.send_response(message.sender, response)

        elif collab_type == 'feature_extraction':
            # Use LSTM hidden states as features
            features = pd.DataFrame(message.payload['data'])
            hidden_features = await self._extract_hidden_features(features)

            await self.send_response(message.sender, {
                'agent_id': self.agent_id,
                'features': hidden_features,
                'feature_type': 'lstm_hidden_states'
            })

    async def _extract_hidden_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract hidden state features from LSTM"""
        self.model.eval()
        X = self.prepare_sequences(data)

        with torch.no_grad():
            # Get hidden states from LSTM
            lstm_out, (hidden, cell) = self.model.lstm(X)
            # Use last hidden state as features
            features = hidden[-1].cpu().numpy()

        return features

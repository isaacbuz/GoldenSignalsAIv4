import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.domain.models.ai_models import LSTMModel, TransformerModel
from infrastructure.config_manager import config_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLTrainingPipeline:
    """
    Comprehensive machine learning training pipeline for trading signal generation.
    Supports multiple model architectures and advanced training techniques.
    """

    def __init__(
        self, 
        model_type: str = 'lstm', 
        data_dir: str = 'data/training'
    ):
        """
        Initialize training pipeline.
        
        Args:
            model_type (str): Type of model to train
            data_dir (str): Directory containing training data
        """
        self.model_type = model_type
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = self._initialize_model()
        self.criterion = nn.MSELoss()
        self.optimizer = None
        
        logger.info(f"Initialized {model_type.upper()} training pipeline")

    def _initialize_model(self):
        """
        Initialize model based on configuration.
        
        Returns:
            nn.Module: Initialized model
        """
        model_configs = {
            'lstm': {
                'input_dim': config_manager.get('ml_models.lstm.input_dim', 5),
                'hidden_dim': config_manager.get('ml_models.lstm.hidden_dim', 64),
                'num_layers': config_manager.get('ml_models.lstm.num_layers', 2)
            },
            'transformer': {
                'input_dim': config_manager.get('ml_models.transformer.input_dim', 5),
                'hidden_dim': config_manager.get('ml_models.transformer.hidden_dim', 64),
                'num_heads': config_manager.get('ml_models.transformer.num_heads', 4)
            }
        }
        
        config = model_configs.get(self.model_type, model_configs['lstm'])
        
        if self.model_type == 'lstm':
            return LSTMModel(**config).to(self.device)
        elif self.model_type == 'transformer':
            return TransformerModel(**config).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def load_data(self, filename: str = 'market_data.npy'):
        """
        Load and preprocess training data.
        
        Args:
            filename (str): Name of data file to load
        
        Returns:
            Tuple of train and validation DataLoaders
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Training data not found: {filepath}")
        
        data = np.load(filepath)
        
        # Split features and labels
        X = data[:, :-1]  # All columns except last
        y = data[:, -1]   # Last column as labels
        
        # Train-test split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        # Create DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        return train_loader, val_loader

    def train(
        self, 
        train_loader, 
        val_loader, 
        epochs: int = 100
    ):
        """
        Train the model with advanced techniques.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            epochs (int): Number of training epochs
        
        Returns:
            Dict[str, Any]: Training metrics
        """
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config_manager.get('ml_models.learning_rate', 0.001)
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        best_val_loss = float('inf')
        training_metrics = {
            'train_losses': [],
            'val_losses': []
        }
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            # Compute average losses
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Update metrics
            training_metrics['train_losses'].append(train_loss)
            training_metrics['val_losses'].append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Model checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f'{self.model_type}_best_model.pth')
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss = {train_loss:.4f}, "
                f"Val Loss = {val_loss:.4f}"
            )
        
        return training_metrics

    def save_model(self, path: str = None):
        """
        Save trained model.
        
        Args:
            path (str, optional): Path to save model. 
                                  Defaults to model-specific path.
        """
        if path is None:
            path = f'{self.model_type}_final_model.pth'
        
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

def main():
    """
    Main training script.
    """
    pipeline = MLTrainingPipeline(model_type='lstm')
    train_loader, val_loader = pipeline.load_data()
    metrics = pipeline.train(train_loader, val_loader)
    pipeline.save_model()
    
    logger.info("Training completed successfully!")

if __name__ == '__main__':
    main()

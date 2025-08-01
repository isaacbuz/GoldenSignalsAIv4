from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    """Configuration for the transformer model."""

    # Model architecture
    input_dim: int = 10  # Number of input features
    d_model: int = 256  # Dimension of the model
    nhead: int = 8  # Number of attention heads
    num_layers: int = 6  # Number of transformer layers
    dim_feedforward: int = 1024  # Dimension of feedforward network
    dropout: float = 0.1  # Dropout rate

    # Training parameters
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    min_confidence: float = 0.7  # Minimum confidence threshold for predictions

    # Optional parameters
    max_seq_length: Optional[int] = None  # Maximum sequence length
    warmup_steps: Optional[int] = None  # Number of warmup steps for learning rate

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        if self.max_seq_length is None:
            self.max_seq_length = 100  # Default sequence length

        if self.warmup_steps is None:
            self.warmup_steps = self.epochs // 10  # Default warmup steps

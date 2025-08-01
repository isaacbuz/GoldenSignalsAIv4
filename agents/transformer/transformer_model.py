import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Input embedding
        self.embedding = nn.Linear(config.input_dim, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=config.num_layers
        )

        # Output layers
        self.fc1 = nn.Linear(config.d_model, config.d_model // 2)
        self.fc2 = nn.Linear(config.d_model // 2, 2)  # [price_prediction, confidence]

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, seq_len, input_dim]

        # Embedding
        x = self.embedding(x)  # [batch_size, seq_len, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create attention mask
        mask = self._generate_square_subsequent_mask(x.size(1)).to(x.device)

        # Transformer encoder
        x = self.transformer_encoder(x, mask)  # [batch_size, seq_len, d_model]

        # Take the last sequence output
        x = x[:, -1, :]  # [batch_size, d_model]

        # Output layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        # Split into prediction and confidence
        price_pred = x[:, 0]
        confidence = torch.sigmoid(x[:, 1])  # Ensure confidence is between 0 and 1

        return torch.stack([price_pred, confidence], dim=1)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a square mask for the sequence."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# LSTM model
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.5):
        super(StockLSTM, self).__init__()
        
        # Parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Layers
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,  # Input shape is (batch, seq, feature)
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # FC layer for classification
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout for the classifier head
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

        out, (h_n, c_n) = self.lstm(x) # (h0, c0) can be passed here if initialized

        # We only need the final time step's hidden state for classification.
        # h_n[-1, :, :] is the hidden state of the last layer, across all batches.
        
        # Out is the output of the final time step of the final layer
        final_output = self.dropout(h_n[-1, :, :]) 
        
        # Fully connected layer for classification
        out = self.fc(final_output)
        
        # The output does NOT include Softmax here; that's typically done in the loss function 
        # (CrossEntropyLoss) or applied after the forward pass.
        return out

# Classical sin/cos positional encoding for transformer
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # Register as a buffer so it's saved with the model but not trained
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional encoding to the input sequence (transposed to match PE shape)
        
        x = x + self.pe[:x.size(1)].squeeze(1) 
        return self.dropout(x)

# Transforemer model
class StockTransformer(nn.Module):
    def __init__(self, 
                 input_size: int,         # Dimensionality of the input features
                 d_model: int,            # Dimensionality of the model's internal representation
                 nhead: int,              # Number of attention heads
                 num_encoder_layers: int, # Number of encoder blocks to stack
                 dim_feedforward: int,    # Dimensionality of the FFN hidden layer
                 output_size: int,        # Number of classification classes
                 dropout: float = 0.1,
                 sequence_length: int = 30):
        
        super().__init__()
        
        # Linear embedding layer to project input features to d_model
        self.embedding = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=sequence_length)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Stacks multiple encoder layers (self-attention blocks)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Final classification head (simple MLP)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, output_size) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size) -> (B, S, F)
        B, S, F = x.shape 
        
        # Input embedding
        x = self.embedding(x) # Shape: (B, S, d_model)
        
        # Using simplified positional encoding 
        x = self.pos_encoder(x)
        
        # Forward through transformer encoder
        out = self.transformer_encoder(x)
        
        # Extract the representation for the LAST element (Day 30)
        final_state = out[:, -1, :] 
        
        # Classification head
        logits = self.classifier(final_state)
        
        # logits shape: (batch_size, output_size), softmax handled in loss function
        return logits
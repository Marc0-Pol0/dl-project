import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.5):
        super(StockLSTM, self).__init__()
        
        # Parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. LSTM Layer
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,  # Input shape is (batch, seq, feature)
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 2. Fully Connected (Classification) Layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # 3. Optional Dropout for the classifier head
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # Initialize hidden state and cell state (optional, can be done implicitly)
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        # out shape: (batch_size, sequence_length, hidden_size)
        # h_n, c_n are the final hidden/cell states
        out, (h_n, c_n) = self.lstm(x) # (h0, c0) can be passed here if initialized

        # We only need the final time step's hidden state for classification.
        # h_n[-1, :, :] is the hidden state of the last layer, across all batches.
        
        # out is the output of the final time step of the final layer
        final_output = self.dropout(h_n[-1, :, :]) 
        
        # Fully connected layer for classification
        out = self.fc(final_output)
        
        # The output does NOT include Softmax here; that's typically done in the loss function 
        # (CrossEntropyLoss) or applied after the forward pass.
        return out
    
class PositionalEncoding(nn.Module):
    """Adds sine/cosine positional information to the input sequence."""
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
        """
        Args:
            x: Tensor, shape (seq_len, batch_size, embedding_dim)
        """
        # Add positional encoding to the input sequence (transposed to match PE shape)
        x = x + self.pe[:x.size(0)].squeeze(1) 
        return self.dropout(x)

# --- 2. Transformer Model ---

class StockTransformer(nn.Module):
    def __init__(self, 
                 input_size: int,         # Dimensionality of the input features (e.g., 21)
                 d_model: int,            # Dimensionality of the model's internal representation
                 nhead: int,              # Number of attention heads
                 num_encoder_layers: int, # Number of encoder blocks to stack
                 dim_feedforward: int,    # Dimensionality of the FFN hidden layer
                 output_size: int,        # Number of classification classes (e.g., 3)
                 dropout: float = 0.1,
                 sequence_length: int = 30):
        
        super().__init__()
        
        # 1. Linear Embedding (Projects input_size to d_model)
        # Transformers require input features to match d_model
        self.embedding = nn.Linear(input_size, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=sequence_length)
        
        # 3. Transformer Encoder Stack
        # Defines a single encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # IMPORTANT: Keep batch_first=True for easier handling
        )
        
        # Stacks multiple encoder layers (self-attention blocks)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # 4. Final Classification Head (MLP)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, output_size) # Maps the final hidden state to 3 classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size) -> (B, S, F)
        B, S, F = x.shape 
        
        # 1. Embed features to d_model space
        x = self.embedding(x) # Shape: (B, S, d_model)
        
        # 2. Add Positional Encoding
        # NOTE: Positional Encoding implementation above expects (S, B, d_model) but 
        # we adjust it in the PE forward pass to handle batch_first=True.
        # However, for the standard PyTorch Transformer, we transpose here for PE:
        # x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1) # (S, B, d_model) -> (B, S, d_model)
        
        # Using simplified PE: 
        x = self.pos_encoder(x)
        
        # 3. Transformer Encoder Forward Pass
        # The encoder outputs an updated representation for EVERY element in the sequence.
        # out shape: (batch_size, sequence_length, d_model)
        out = self.transformer_encoder(x)
        
        # 4. Extract the representation for the LAST element (Day 30)
        # This is the vector summarizing all preceding inputs relevant to the prediction date.
        # final_state shape: (batch_size, d_model)
        final_state = out[:, -1, :] 
        
        # 5. Classification
        logits = self.classifier(final_state)
        
        # logits shape: (batch_size, output_size)
        # No Softmax is included here, as it will be handled by nn.CrossEntropyLoss
        return logits


# if __name__ == '__main__':
#     # Example test run
#     INPUT_SIZE = 22 # Your feature count
#     HIDDEN_SIZE = 64
#     NUM_LAYERS = 2
#     OUTPUT_SIZE = 3 # Up/Down/Neutral

#     model = StockLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    
#     # Create a dummy batch of data: 4 samples, 30 timesteps, 22 features
#     dummy_input = torch.randn(4, 30, 22) 
    
#     output = model(dummy_input)
    
#     print("Model Output Shape:", output.shape) # Expected: [4, 3] (Batch size, 3 classes)
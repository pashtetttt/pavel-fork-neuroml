# cnn_bilstm_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    """
    A single CNN block with Conv1d -> BatchNorm -> GELU -> Dropout.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.1):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, time, features) -> (batch, features, time)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        # Back to (batch, time, features)
        x = x.transpose(1, 2)
        return x

class CNNBiLSTMModel(nn.Module):
    """
    CNN layers followed by a BiLSTM for sequence modeling.
    """
    def __init__(self, neural_dim=512, n_units=512, n_classes=41, n_cnn_layers=3, cnn_dropout=0.1,
                 n_lstm_layers=2, lstm_dropout=0.1, input_dropout=0.1, n_days=45):
        super(CNNBiLSTMModel, self).__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_days = n_days

        # Input dropout
        self.input_dropout = nn.Dropout(input_dropout)

        # --- Day-specific input layer: Store weights and biases ---
        # Weights for the linear transformation per day: (n_days, n_units, neural_dim)
        self.day_weights = nn.Parameter(torch.randn(n_days, n_units, neural_dim))
        # Bias for day-specific input: (n_days, n_units)
        self.day_bias = nn.Parameter(torch.zeros(n_days, n_units))

        # CNN Layers
        cnn_layers = []
        in_ch = n_units # Output from day-specific layer
        for _ in range(n_cnn_layers):
            cnn_layers.append(CNNBlock(in_ch, n_units, dropout=cnn_dropout))
            in_ch = n_units # Maintain consistent channel size
        self.cnn_layers = nn.Sequential(*cnn_layers)

        # BiLSTM Layer
        self.lstm = nn.LSTM(
            input_size=n_units,
            hidden_size=n_units // 2, # //2 for bidirectional to keep total params comparable
            num_layers=n_lstm_layers,
            dropout=lstm_dropout if n_lstm_layers > 1 else 0, # No dropout on last layer if only 1 layer
            bidirectional=True,
            batch_first=True
        )

        # Output layer
        self.out = nn.Linear(n_units, n_classes) # n_units because BiLSTM is bidirectional (n_units//2 * 2)

    def forward(self, x, day_indicies): # x: (batch, time, neural_dim), day_indicies: (batch,)
        # Input dropout
        x = self.input_dropout(x)

        # --- Apply day-specific transformation ---
        batch_size, seq_len, _ = x.shape

        # Use advanced indexing to select the correct weight matrix and bias for each sample in the batch
        # day_indicies: (batch,) -> unsqueeze(-1).unsqueeze(-1): (batch, 1, 1)
        # self.day_weights: (n_days, n_units, neural_dim)
        # selected_weights = self.day_weights[day_indicies] # This creates a view: (batch, n_units, neural_dim)
        # Similarly for bias
        # selected_biases = self.day_bias[day_indicies] # This creates a view: (batch, n_units)
        # Note: Using day_indicies directly as an index on the first dimension of a multi-dimensional tensor
        # is standard PyTorch behavior for advanced indexing.
        selected_weights = self.day_weights[day_indicies] # Shape: (batch, n_units, neural_dim)
        selected_biases = self.day_bias[day_indicies]     # Shape: (batch, n_units)

        # Perform batched matrix multiplication: x @ W.T + b
        # x: (batch, time, neural_dim)
        # selected_weights: (batch, n_units, neural_dim) -> transpose(-2, -1): (batch, neural_dim, n_units)
        # Result of bmm: (batch, time, neural_dim) x (batch, neural_dim, n_units) -> (batch, time, n_units)
        x = torch.bmm(x, selected_weights.transpose(-2, -1)) # (batch, time, n_units)

        # Add the batch-specific bias: (batch, time, n_units) + (batch, 1, n_units) -> (batch, time, n_units)
        # selected_biases: (batch, n_units) -> unsqueeze(1): (batch, 1, n_units)
        x = x + selected_biases.unsqueeze(1) # Broadcasting adds bias to each time step

        # Pass through CNN layers
        x = self.cnn_layers(x) # (batch, time, n_units)

        # Pass through BiLSTM
        lstm_out, _ = self.lstm(x) # (batch, time, n_units) # bidirectional concatenates forward and backward

        # Apply output layer
        logits = self.out(lstm_out) # (batch, time, n_classes)

        return logits

    # Optional: Add a method to freeze day-specific layers if needed in trainer
    # def set_day_layers_trainable(self, trainable):
    #     self.day_weights.requires_grad = trainable
    #     self.day_bias.requires_grad = trainable
import torch
import torch.nn as nn
import torch.nn.functional as F

class DCoND_GRUDecoder(nn.Module):
    def __init__(self, neural_dim, n_units, n_classes, n_layers, rnn_dropout, input_dropout, n_days):
        super().__init__()
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes  # e.g., 41 phonemes
        self.n_diphone_classes = n_classes * n_classes  # 41*41 = 1681
        self.n_days = n_days

        # Day-specific input layer
        self.day_weights = nn.Parameter(torch.randn(n_days, n_units, neural_dim))
        self.day_bias = nn.Parameter(torch.zeros(n_days, n_units))
        self.input_dropout = nn.Dropout(input_dropout)

        # GRU backbone
        self.gru = nn.GRU(
            input_size=n_units,
            hidden_size=n_units,
            num_layers=n_layers,
            dropout=rnn_dropout if n_layers > 1 else 0,
            batch_first=True,
            bidirectional=False  # DCoND uses unidirectional GRU
        )

        # Diphone output head
        self.diphone_head = nn.Linear(n_units, self.n_diphone_classes)

    def forward(self, x, day_indices):
        batch_size, seq_len, _ = x.shape

        # Day-specific linear transform
        w = self.day_weights[day_indices]  # (B, n_units, neural_dim)
        b = self.day_bias[day_indices]     # (B, n_units)
        x = torch.bmm(x, w.transpose(-2, -1)) + b.unsqueeze(1)  # (B, T, n_units)

        x = self.input_dropout(x)

        # GRU encoding
        gru_out, _ = self.gru(x)  # (B, T, n_units)

        # Diphone logits
        diphone_logits = self.diphone_head(gru_out)  # (B, T, n_classes * n_classes)
        diphone_logits = diphone_logits.view(batch_size, seq_len, self.n_classes, self.n_classes)

        # Marginalize over preceding phoneme to get monophone logits
        # Sum over the first phoneme dimension (rows) -> get P(current_phoneme)
        monophone_logits = diphone_logits.sum(dim=2)  # (B, T, n_classes)

        return monophone_logits, diphone_logits

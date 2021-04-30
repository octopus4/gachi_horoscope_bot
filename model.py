import torch
import torch.nn as nn
import torch.nn.functional as f

from typing import Any


class LanguageModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, vocab_size)
        )
        self.softmax = nn.LogSoftmax(1)
        self.decoder.weight = self.encoder.weight

    def forward(self, input, hidden, temperature: float = 1.0):
        emb = f.gelu(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        reshaped_output = output.reshape(output.size(0) * output.size(1), output.size(2))
        decoded = self.decoder(reshaped_output)
        scaled_decoded = decoded / temperature
        result = self.softmax(scaled_decoded).view(output.size(0), output.size(1), decoded.size(1))

        return result, hidden

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    @torch.jit.export
    def init_hidden(self, batch_size: int):
        return torch.ones(self.num_layers, batch_size, self.hidden_dim)

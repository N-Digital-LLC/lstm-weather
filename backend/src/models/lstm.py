"""Phase 3/4 — LSTM forecaster (single- and direct multi-step).

```
input: (B, L, F)
  -> LSTM(input_size=F, hidden=HIDDEN, num_layers=NUM_LAYERS, batch_first=True, dropout=DROPOUT)
  -> last layer's final hidden state (B, HIDDEN)
  -> Dropout -> Linear(HIDDEN -> H)
output: (B, H)
```

``hidden_size``/``num_layers``/``horizon`` come from the run config so the UI can vary them.
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        horizon: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        # PyTorch only applies LSTM dropout between stacked layers, so it is a no-op with 1 layer.
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        _, (h_n, _) = self.lstm(x)        # h_n: (num_layers, B, hidden)
        last = h_n[-1]                    # (B, hidden) — final layer's last hidden state
        return self.head(self.dropout(last))  # (B, H)


def build_from_config(input_size: int, cfg: dict) -> LSTMForecaster:
    """Construct an ``LSTMForecaster`` from a run config dict."""
    return LSTMForecaster(
        input_size=input_size,
        hidden_size=int(cfg["hidden_size"]),
        num_layers=int(cfg["num_layers"]),
        horizon=int(cfg["horizon"]),
        dropout=float(cfg.get("dropout", 0.2)),
    )

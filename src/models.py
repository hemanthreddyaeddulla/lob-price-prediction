import torch
import torch.nn as nn
from typing import Optional, Tuple


class CrossFeatureMixer(nn.Module):
    """Pointwise Conv1d stack over the feature axis. Lets the model learn cross-feature
    interactions (bid x ask, price x volume) without me handcoding every combination."""

    def __init__(self, input_dim: int, output_dim: Optional[int] = None, n_layers: int = 3):
        super().__init__()
        output_dim = output_dim or input_dim
        layers = []
        dim = input_dim
        for i in range(n_layers - 1):
            layers.extend([
                nn.Conv1d(dim, dim, kernel_size=1),
                nn.GELU(),
            ])
        layers.append(nn.Conv1d(dim, output_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features). Conv1d wants channels in the middle.
        out = self.net(x.transpose(1, 2))
        return out.transpose(1, 2)


class PredictionHead(nn.Module):
    """Single-output MLP head."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GRUModel(nn.Module):
    """LayerNorm -> Conv mixer (residual) -> GRU -> two heads (t0, t1).

    Has both a full-sequence forward (used in training) and a single-step forward
    (used for ONNX export and online inference).
    """

    def __init__(
        self,
        input_dim: int = 158,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        mixer_layers: int = 3,
        head_hidden: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_norm = nn.LayerNorm(input_dim)
        self.mixer = CrossFeatureMixer(input_dim, input_dim, n_layers=mixer_layers)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.t0_head = PredictionHead(hidden_size, head_hidden, dropout)
        self.t1_head = PredictionHead(hidden_size, head_hidden, dropout)

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_norm(x)
        x = x + self.mixer(x)
        gru_out, hidden = self.gru(x, hidden)
        t0 = self.t0_head(gru_out)
        t1 = self.t1_head(gru_out)
        return torch.cat([t0, t1], dim=-1), hidden

    def forward_step(
        self, x: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-step path. x: (1, input_dim), hidden: (num_layers, 1, hidden_size).
        Used for ONNX export so the runtime sees hidden state as explicit I/O."""
        x = self.input_norm(x)
        x_3d = x.unsqueeze(1)
        x_mixed = x_3d + self.mixer(x_3d)
        gru_out, new_hidden = self.gru(x_mixed, hidden)
        gru_out = gru_out.squeeze(1)
        t0 = self.t0_head(gru_out)
        t1 = self.t1_head(gru_out)
        return torch.cat([t0, t1], dim=-1), new_hidden

    def init_hidden(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class LSTMModel(nn.Module):
    """Same shape as GRUModel but with LSTM (so two state tensors, h and c).

    Kept around because I wanted to confirm GRU wasn't beating LSTM by accident.
    It wasn't - they're essentially tied per-fold but GRU is faster, so the
    final submission uses GRU only.
    """

    def __init__(
        self,
        input_dim: int = 158,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        mixer_layers: int = 3,
        head_hidden: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_norm = nn.LayerNorm(input_dim)
        self.mixer = CrossFeatureMixer(input_dim, input_dim, n_layers=mixer_layers)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.t0_head = PredictionHead(hidden_size, head_hidden, dropout)
        self.t1_head = PredictionHead(hidden_size, head_hidden, dropout)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.input_norm(x)
        x = x + self.mixer(x)
        lstm_out, hidden = self.lstm(x, hidden)
        t0 = self.t0_head(lstm_out)
        t1 = self.t1_head(lstm_out)
        return torch.cat([t0, t1], dim=-1), hidden

    def forward_step(
        self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_norm(x)
        x_3d = x.unsqueeze(1)
        x_mixed = x_3d + self.mixer(x_3d)
        lstm_out, (new_h, new_c) = self.lstm(x_mixed, (h, c))
        lstm_out = lstm_out.squeeze(1)
        t0 = self.t0_head(lstm_out)
        t1 = self.t1_head(lstm_out)
        return torch.cat([t0, t1], dim=-1), new_h, new_c

    def init_hidden(
        self, batch_size: int = 1, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c


def create_model(
    arch: str = "gru",
    input_dim: int = 158,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    mixer_layers: int = 3,
    head_hidden: int = 64,
) -> nn.Module:
    kwargs = dict(
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        mixer_layers=mixer_layers,
        head_hidden=head_hidden,
    )
    if arch == "gru":
        return GRUModel(**kwargs)
    if arch == "lstm":
        return LSTMModel(**kwargs)
    raise ValueError(f"Unknown architecture: {arch!r}. Use 'gru' or 'lstm'.")

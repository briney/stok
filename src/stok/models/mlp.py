import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feedforward block.

    Implements x -> W_down( SiLU(W_gate x) ⊙ (W_up x) ).
    """

    def __init__(self, d_model: int, expansion: float, dropout: float):
        """Initialize SwiGLU feedforward block.

        Args:
            d_model: Model dimension.
            expansion: Expansion factor (hidden_dim = d_model * expansion).
            dropout: Dropout probability.
        """
        super().__init__()
        # set hidden dimesion to nearest multiple of 256 after expansion ratio
        hidden = int(((expansion * d_model) + 255) // 256 * 256)
        # hidden = int(d_model * expansion)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SwiGLU.

        Args:
            x: Input tensor of shape [B, L, d_model].

        Returns:
            Output tensor of shape [B, L, d_model].
        """
        up = self.w_up(x)
        gate = F.silu(self.w_gate(x))
        y = self.w_down(up * gate)
        return self.dropout(y)

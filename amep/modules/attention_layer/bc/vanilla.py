import torch
import torch.nn as nn
from allennlp.nn.util import masked_softmax
from overrides import overrides

from amep.modules.attention_layer.base import AttentionLayer


@AttentionLayer.register("bc_tanh")
class TanhBCAttention(AttentionLayer):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )
        self._hidden_size = hidden_size

    @overrides
    def get_output_dim(self) -> int:
        return self._hidden_size

    @overrides
    def forward(self, hidden: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:

        attn = self.attn(hidden).squeeze(-1)
        return masked_softmax(attn, masks)


@AttentionLayer.register("bc_dot")
class DotBCAttention(AttentionLayer):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self._hidden_size = hidden_size
        self.attn1 = nn.Linear(hidden_size, 1, bias=False)

    @overrides
    def get_output_dim(self) -> int:
        return self._hidden_size

    @overrides
    def forward(self, hidden: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:

        attn1 = self.attn1(hidden) / (self._hidden_size) ** 0.5
        attn1 = attn1.squeeze(-1)
        return masked_softmax(attn1, masks)

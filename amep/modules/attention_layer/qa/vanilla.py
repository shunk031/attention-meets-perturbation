import torch
import torch.nn as nn
from allennlp.nn.util import masked_softmax
from overrides import overrides

from amep.modules.attention_layer import AttentionLayer


@AttentionLayer.register("qa_tanh")
class TanhQAAttention(AttentionLayer):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attn1p = nn.Linear(hidden_size, hidden_size // 2)
        self.attn1q = nn.Linear(hidden_size, hidden_size // 2)
        self.attn2 = nn.Linear(hidden_size // 2, 1, bias=False)
        self.tanh = nn.Tanh()
        self._hidden_size = hidden_size

    @overrides
    def get_output_dim(self) -> int:
        return self._hidden_size

    def forward(
        self, hidden_p: torch.Tensor, hidden_q: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:

        attn = self.tanh(self.attn1p(hidden_p) + self.attn1q(hidden_q).unsqueeze(dim=1))
        attn = self.attn2(attn).squeeze(dim=-1)

        return masked_softmax(attn, masks)


@AttentionLayer.register("qa_dot")
class DotQAAttention(AttentionLayer):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self._hidden_size = hidden_size

    @overrides
    def get_output_dim(self) -> int:
        return self._hidden_size

    def forward(
        self, hidden_p: torch.Tensor, hidden_q: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        attn = (
            torch.bmm(hidden_p, hidden_q.unsqueeze(dim=-1)) / (self._hidden_size) ** 0.5
        )
        attn = attn.squeeze(dim=-1)
        return masked_softmax(attn, masks)

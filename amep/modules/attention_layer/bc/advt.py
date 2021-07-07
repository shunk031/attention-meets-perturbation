import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.nn.util import masked_softmax
from overrides import overrides

from amep.modules.attention_layer.base import AttentionLayer


class TanhBCATBase(AttentionLayer):
    def __init__(self, hidden_size: int, xi: float) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )
        self._hidden_size = hidden_size
        self._xi = xi

    @overrides
    def get_output_dim(self) -> int:
        return self._hidden_size

    def _advt_step(
        self, attn: torch.Tensor, is_first_step: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError

    @overrides
    def forward(
        self,
        hidden: torch.Tensor,
        masks: torch.Tensor,
        is_advt: bool = False,
        is_first_step: bool = False,
    ) -> torch.Tensor:

        attn = self.attn(hidden).squeeze(-1)
        if is_advt:
            attn = self._advt_step(attn, is_first_step)

        return masked_softmax(attn, masks)


@AttentionLayer.register("bc_at_tanh")
class TanhBCAttentionAT(TanhBCATBase):
    def __init__(self, hidden_size: int, xi: float) -> None:
        super().__init__(hidden_size, xi)

    @overrides
    def _advt_step(
        self, attn: torch.Tensor, is_first_step: bool = False
    ) -> torch.Tensor:

        if is_first_step:
            d_var = attn.new_zeros(attn.size(), requires_grad=True)
            attn = attn + d_var
            self.d_var_ = d_var
        else:
            r_adv = F.normalize(self.d_var_.grad)
            attn = attn + self._xi * r_adv

        return attn


@AttentionLayer.register("bc_iat_tanh")
class TanhBCAttentioniAT(TanhBCATBase):
    def __init__(self, hidden_size: int, xi: float) -> None:
        super().__init__(hidden_size, xi)

    @overrides
    def _advt_step(
        self, attn: torch.Tensor, is_first_step: bool = False
    ) -> torch.Tensor:

        if is_first_step:

            bsize = attn.size(0)

            # shape: (bsize, bsize, len_sentence)
            attn_1 = attn.repeat(1, bsize).view(bsize, bsize, -1)

            # shape: (bsize, bsize, len_sentence)
            attn_2 = attn.repeat(bsize, 1, 1)

            # shape: (bsize, bsize, len_sentence)
            attn_diff = attn_1 - attn_2
            dir_normed = F.normalize(attn_diff)
            self.dir_normed_ = dir_normed

            # shape: (bsize, bsize, len_sentence)
            attn_d_var = dir_normed.new_zeros(dir_normed.size(), requires_grad=True)
            attn_d = (attn_d_var * dir_normed).sum(axis=1)

            d_var = F.normalize(attn_d)
            d_var.retain_grad()

            self.d_var_ = d_var
            attn = attn + self.d_var_
        else:
            r_adv = F.normalize(-self.d_var_.grad)
            attn = attn + self._xi * r_adv

        return attn

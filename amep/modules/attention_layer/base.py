import torch.nn as nn
from allennlp.common import Registrable


class AttentionLayer(nn.Module, Registrable):
    def __init__(self) -> None:
        super().__init__()

    def get_output_dim(self) -> int:
        raise NotImplementedError

    def forward(self, **kwargs):
        raise NotImplementedError

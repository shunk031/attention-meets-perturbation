from typing import Dict

import torch
import torch.nn as nn
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, weighted_sum
from allennlp.training.metrics import Auc, CategoricalAccuracy, F1Measure
from overrides import overrides

from amep.modules.attention_layer import AttentionLayer
from amep.nn.util import weighted_loss


@Model.register("bc_weighted")
class BCWeightedModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        word_embed: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        attention_layer: AttentionLayer,
        regularizer: RegularizerApplicator = None,
    ) -> None:
        super().__init__(vocab, regularizer)

        self._word_embed = word_embed
        self._encoder = encoder
        self._attention_layer = attention_layer
        self._output = nn.Linear(
            in_features=attention_layer.get_output_dim(), out_features=1
        )

        self._loss = nn.BCEWithLogitsLoss(reduction="none")
        self._metrics = {
            "acc1": CategoricalAccuracy(),
            "f1score": F1Measure(positive_label=1),
        }
        self._auc = Auc(positive_label=1)

    @overrides
    def forward(
        self,
        tokens: Dict[str, torch.Tensor],
        label: torch.Tensor = None,
        pos_weight: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(tokens)

        word_embed = self._word_embed(tokens)
        hidden = self._encoder(word_embed, mask)
        attention = self._attention_layer(hidden, mask)
        logit = self._output(weighted_sum(hidden, attention))

        output_dict: Dict[str, torch.Tensor] = {"logit": logit, "attention": attention}

        if label is not None:
            label = label.float()
            bce_loss = self._loss(logit.squeeze(dim=-1), label)
            output_dict["loss"] = weighted_loss(bce_loss, label, pos_weight)

            one_minus_logit = 1 - logit
            probs = torch.stack((one_minus_logit, logit), dim=-1)
            for eval_metric in self._metrics.values():
                eval_metric(probs, label.unsqueeze(dim=-1))
            self._auc(logit.squeeze(dim=-1), label)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_scores = {
            metric_name: metric.get_metric(reset)
            for metric_name, metric in self._metrics.items()
        }

        for key, value in metric_scores.items():
            if isinstance(metric_scores[key], dict):
                # f1 get_metric returns dict (precision, recall, f1)
                metric_scores[key] = metric_scores[key]["f1"]

        metric_scores["auc"] = self._auc.get_metric(reset)
        return metric_scores

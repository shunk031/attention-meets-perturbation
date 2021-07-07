from typing import Dict

import torch
import torch.nn as nn
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, weighted_sum
from allennlp.training.metrics import Auc, Average, CategoricalAccuracy, F1Measure
from overrides import overrides

from amep.common.util import convert_binary_prob_to_multi_prob
from amep.modules.attention_layer import AttentionLayer
from amep.nn.util import weighted_loss


@Model.register("bc_attn_at")
class BCAttentionATModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        word_embed: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        attention_layer: AttentionLayer,
        lam_adv: float = 1.0,
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
        self._loss_adv = Average()
        self._lam_adv = lam_adv

    def _forward_internal(
        self,
        tokens: Dict[str, torch.Tensor],
        mask: torch.Tensor = None,
        label: torch.Tensor = None,
        pos_weight: torch.Tensor = None,
        return_attention: bool = False,
        is_advt: bool = False,
        is_first_step: bool = False,
    ) -> torch.Tensor:

        # get word embedding
        word_embed = self._word_embed(tokens)
        # get hidden representation
        hidden = self._encoder(word_embed, mask)
        # get attention score
        attention = self._attention_layer(hidden, mask, is_advt, is_first_step)

        logit = self._output(weighted_sum(hidden, attention))

        if return_attention:
            return logit, attention
        else:
            return logit

    @overrides
    def forward(
        self,
        tokens: Dict[str, torch.Tensor],
        label: torch.Tensor = None,
        pos_weight: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(tokens)

        logit_orig, attention = self._forward_internal(
            tokens, mask, label, pos_weight, return_attention=True
        )
        output_dict: Dict[str, torch.Tensor] = {
            "logit": logit_orig,
            "attention": attention,
        }

        if label is not None:
            label = label.float()
            bce_loss = self._loss(logit_orig.squeeze(dim=-1), label)
            output_dict["loss"] = weighted_loss(bce_loss, label, pos_weight)

            probs = convert_binary_prob_to_multi_prob(logit_orig)
            for eval_metric in self._metrics.values():
                eval_metric(probs, label.unsqueeze(dim=-1))
            self._auc(logit_orig.squeeze(dim=-1), label)

            if torch.is_grad_enabled() and self.training:

                # forward first time (-> is_first_step = True)
                logit = self._forward_internal(
                    tokens, mask, label, pos_weight, is_advt=True, is_first_step=True
                )
                loss_adv_first = self._loss(logit.squeeze(dim=-1), label)
                loss_adv_first = weighted_loss(loss_adv_first, label, pos_weight)

                # sets gradients of all model parameters to zero
                self.zero_grad()
                loss_adv_first.backward(retain_graph=True)

                # forward second time (-> is_first_step = False)
                logit = self._forward_internal(
                    tokens, mask, label, pos_weight, is_advt=True, is_first_step=False
                )
                loss_adv = self._loss(logit.squeeze(dim=-1), label)
                loss_adv = weighted_loss(loss_adv, label, pos_weight)

                output_dict["loss_adv"] = loss_adv
                output_dict["loss"] += self._lam_adv * loss_adv

                # record the value of loss_adv
                self._loss_adv(loss_adv)

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

        # In validation phase, loss_adv isn't calculated, so the value set to 0.
        loss_adv = self._loss_adv.get_metric(reset).item() if self.training else 0.0
        metric_scores["loss_adv"] = loss_adv

        return metric_scores

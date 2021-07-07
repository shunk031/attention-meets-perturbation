from typing import Dict

import torch
import torch.nn as nn
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides

from amep.common.subset_metrics import calc_subset_type_based_metric
from amep.modules.attention_layer import AttentionLayer
from amep.nn.util import masked_fill_for_qa


@Model.register("qa_vanilla")
class QAVanillaModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        word_embed: TextFieldEmbedder,
        paragraph_encoder: Seq2SeqEncoder,
        question_encoder: Seq2VecEncoder,
        attention_layer: AttentionLayer,
        regularizer: RegularizerApplicator = None,
    ) -> None:

        super().__init__(vocab, regularizer)

        self._word_embed = word_embed
        self._paragraph_encoder = paragraph_encoder
        self._question_encoder = question_encoder
        self._attention_layer = attention_layer

        self._p_out = nn.Linear(
            in_features=attention_layer.get_output_dim(),
            out_features=attention_layer.get_output_dim() // 2,
        )
        self._q_out = nn.Linear(
            in_features=attention_layer.get_output_dim(),
            out_features=attention_layer.get_output_dim() // 2,
        )
        self._output = nn.Linear(
            in_features=attention_layer.get_output_dim() // 2,
            out_features=vocab.get_vocab_size("labels"),
        )
        self._tanh = nn.Tanh()

        self._loss = nn.CrossEntropyLoss()
        self._metrics = {
            "acc1": CategoricalAccuracy(),
            "acc3": CategoricalAccuracy(top_k=3),
        }
        self._subset_metrics = {
            "acc_matched": CategoricalAccuracy(),
            "acc_mismatched": CategoricalAccuracy(),
        }

    @overrides
    def forward(
        self,
        paragraph: Dict[str, torch.Tensor],
        question: Dict[str, torch.Tensor],
        answer: torch.Tensor = None,
        entity_mask: torch.Tensor = None,
        metadata: Dict[str, torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        p_mask = get_text_field_mask(paragraph)
        q_mask = get_text_field_mask(question)

        p_embed = self._word_embed(paragraph)
        q_embed = self._word_embed(question)

        p_hidden = self._paragraph_encoder(p_embed, p_mask)
        q_hidden = self._question_encoder(q_embed, q_mask)

        attention = self._attention_layer(p_hidden, q_hidden, p_mask)
        context = weighted_sum(p_hidden, attention)

        logit = self._output(self._tanh(self._p_out(context) + self._q_out(q_hidden)))
        logit = masked_fill_for_qa(logit, entity_mask)

        output_dict: Dict[str, torch.Tensor] = {"logit": logit, "attention": attention}

        if answer is not None:
            loss = self._loss(logit, answer)
            output_dict["loss"] = loss

            for eval_metric in self._metrics.values():
                eval_metric(logit, answer.squeeze(dim=-1))

            if metadata is not None:
                for metric, metric_func in self._subset_metrics.items():
                    calc_subset_type_based_metric(
                        metric_func,
                        logit,
                        answer,
                        metadata,
                        subset_type=f"test_{metric[:3]}",
                    )
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        metric_scores = {
            metric_name: metric.get_metric(reset)
            for metric_name, metric in self._metrics.items()
        }

        for key, value in metric_scores.items():
            if isinstance(metric_scores[key], tuple):
                # f1 get_metric returns (precision, recall, f1)
                metric_scores[key] = metric_scores[key][2]

        # for calculating subset type based metrics
        for key in self._subset_metrics.keys():
            metric_scores[key] = self._subset_metrics[key].get_metric(reset)

        return metric_scores

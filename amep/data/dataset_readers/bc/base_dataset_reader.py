import logging
import pathlib
from typing import Dict, Iterable, Optional

import numpy as np
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from amep.common.util import filter_by_length, read_jsonl
from overrides import overrides

logger = logging.getLogger(__name__)


class BCDatasetReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer,
        token_indexers: Dict[str, TokenIndexer],
        dataset_path: pathlib.Path,
        min_length: int = 5,
        max_length: int = None,
        lazy: bool = False,
        cache_directory: Optional[str] = None,
        max_instances: Optional[int] = None,
        manual_distributed_sharding: bool = False,
        manual_multi_process_sharding: bool = False,
    ) -> None:

        super().__init__(
            lazy=lazy,
            cache_directory=cache_directory,
            max_instances=max_instances,
            manual_distributed_sharding=manual_distributed_sharding,
            manual_multi_process_sharding=manual_multi_process_sharding,
        )

        self._tokenizer = tokenizer
        self._token_indexers = token_indexers

        df = read_jsonl(dataset_path)
        self._df = filter_by_length(df, min_length, max_length)

        # for setting balanced pos weight
        y = self._df[self._df.exp_split == "train"].label
        self.pos_weight = np.asarray([len(y) / sum(y) - 1])
        logger.info(f"Positive weight: {self.pos_weight}")

    @overrides
    def _read(self, file_path) -> Iterable[Instance]:
        logger.info(f"Reading instances from: {file_path}")

        df = self._df[self._df.exp_split == file_path]
        df = df.drop(columns="exp_split")

        logger.info(f"Class distribution: {df.label.value_counts()}")

        for i in range(len(df)):
            yield self.text_to_instance(**df.iloc[i].to_dict())

    @overrides
    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokenized_text = self._tokenizer.tokenize(text)
        text_field = TextField(tokenized_text, self._token_indexers)

        fields = {"tokens": text_field}
        fields["pos_weight"] = ArrayField(self.pos_weight)

        if label is not None:
            fields["label"] = LabelField(str(label))

        return Instance(fields)

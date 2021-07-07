import logging
from typing import Dict, Optional

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, MetadataField, TextField
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides

from amep import const
from amep.common.util import read_jsonl

logger = logging.getLogger(__name__)


@DatasetReader.register("multi_nli")
class MultiNLIDatasetReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer,
        token_indexers: Dict[str, TokenIndexer],
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
        self._df = read_jsonl(const.DATASET_FPATHS["multi_nli"])

    @overrides
    def _read(self, file_path):
        logger.info(f"Reading instances from : {file_path}")

        if file_path == "test":
            df = self._df[
                (self._df.exp_split == "test_matched")
                | (self._df.exp_split == "test_mismatched")
            ]
        else:
            df = self._df[self._df.exp_split == file_path]

        for i in range(len(df)):
            paragraph = df.iloc[i].paragraph
            question = df.iloc[i].question
            answer = df.iloc[i].answer
            exp_split = df.iloc[i].exp_split if file_path == "test" else None

            yield self.text_to_instance(
                paragraph, question, answer, subset_type=exp_split
            )

    @overrides
    def text_to_instance(
        self, paragraph: str, question: str, answer: str = None, subset_type: str = None
    ) -> Instance:
        tokenized_paragraph = self._tokenizer.tokenize(paragraph)
        tokenized_question = self._tokenizer.tokenize(question)

        fields = {
            "paragraph": TextField(tokenized_paragraph, self._token_indexers),
            "question": TextField(tokenized_question, self._token_indexers),
        }

        if answer is not None:
            fields["answer"] = LabelField(answer)

        if subset_type is not None:
            fields["metadata"] = MetadataField({"subset_type": subset_type})

        return Instance(fields)

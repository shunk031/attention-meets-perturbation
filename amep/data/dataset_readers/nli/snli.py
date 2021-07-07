import logging
from typing import Dict, Iterable, Optional

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from amep import const
from amep.common.util import read_jsonl
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("snli_custom")
class SNLICustomDatasetReader(DatasetReader):
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
        self._df = read_jsonl(const.DATASET_FPATHS["snli"])

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        logger.info(f"Reading instances from: {file_path}")
        df = self._df[self._df.exp_split == file_path]
        df = df.drop(columns="exp_split")

        for i in range(len(df)):
            yield self.text_to_instance(**df.iloc[i].to_dict())

    @overrides
    def text_to_instance(
        self, paragraph: str, question: str, answer: str = None
    ) -> Instance:
        tokenized_paragraph = self._tokenizer.tokenize(paragraph)
        tokenized_question = self._tokenizer.tokenize(question)

        fields = {
            "paragraph": TextField(tokenized_paragraph, self._token_indexers),
            "question": TextField(tokenized_question, self._token_indexers),
        }

        if answer is not None:
            fields["answer"] = LabelField(answer)

        return Instance(fields)

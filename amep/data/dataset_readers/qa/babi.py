import logging
from typing import Dict, Iterator, Optional

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides

from amep import const
from amep.common.util import read_jsonl

logger = logging.getLogger(__name__)


@DatasetReader.register("babi_custom")
class BabiDatasetReader(DatasetReader):
    def __init__(
        self,
        task: str,
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
        self._df = read_jsonl(const.DATASET_DIRS["babi"] / f"babi_{task}_dataset.jsonl")

    @overrides
    def _read(self, file_path) -> Iterator[Instance]:
        logger.info(f"Reading instances from: {file_path}")
        df = self._df[self._df.exp_split == file_path]
        df = df.drop("exp_split", axis=1)

        for i in range(len(df)):
            yield self.text_to_instance(**df.iloc[i].to_dict())

    @overrides
    def text_to_instance(
        self, paragraph: str, question: str, answer: str = None
    ) -> Instance:

        paragraph = self._tokenizer.tokenize(paragraph)
        question = self._tokenizer.tokenize(question)

        fields = {
            "paragraph": TextField(paragraph, self._token_indexers),
            "question": TextField(question, self._token_indexers),
        }

        if answer is not None:
            fields["answer"] = LabelField(answer)

        return Instance(fields)

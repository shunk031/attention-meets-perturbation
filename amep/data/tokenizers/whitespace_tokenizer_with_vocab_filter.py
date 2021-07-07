from typing import List

from allennlp.common.file_utils import read_set_from_file
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers.token import Token
from overrides import overrides


@Tokenizer.register("whitespace_vocab_filter")
class WhitespaceVocabFilterTokenizer(Tokenizer):
    def __init__(self, vocab_file: str) -> None:
        super().__init__()
        self.vocab = {t.lower() for t in read_set_from_file(vocab_file)}

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return [Token(t) for t in text.split() if t.lower() in self.vocab]

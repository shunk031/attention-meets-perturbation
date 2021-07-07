import pathlib
import re
from typing import List

import pandas as pd
import spacy
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from sklearn.feature_extraction.text import CountVectorizer

NLP = spacy.load("en", disable=["parser", "tagger", "ner"])


def cleaner(text: str, spacy: bool = True) -> str:

    text = re.sub(r"\s+", " ", text.strip())
    if spacy:
        text = [t.text.lower() for t in NLP(text)]
    else:
        text = [t.lower() for t in text.split()]
    text = ["qqq" if any(char.isdigit() for char in word) else word for word in text]
    return " ".join(text)


def load_entity_list(entity_list_fpath: pathlib.Path) -> List[str]:
    with entity_list_fpath.open("r") as rf:
        entity_list = [line.strip() for line in rf.readlines()]
    return entity_list


def make_count_vectorizer_based_vocab(
    df_text: pd.DataFrame,
    save_fpath: pathlib.Path,
    min_df: int = 2,
    entity_list_fpath: pathlib.Path = None,
) -> None:

    count_vectorizer = CountVectorizer(
        tokenizer=lambda x: x.split(" "), lowercase=False, min_df=min_df
    )
    print("Build count vectorizer based vocabulary...")
    count_vectorizer.fit_transform(df_text)
    vocab = set(count_vectorizer.vocabulary_.keys())
    vocab.add(DEFAULT_OOV_TOKEN)
    vocab.add(DEFAULT_PADDING_TOKEN)

    print(f"Vocabulary size: {len(vocab)}")

    if entity_list_fpath is not None:
        entity_list = load_entity_list(entity_list_fpath)
        vocab = vocab.union(entity_list)

    with save_fpath.open("w") as wf:
        wf.write("\n".join(vocab))

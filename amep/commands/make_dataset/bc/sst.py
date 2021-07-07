import argparse
import logging
import re

import nltk
import pandas as pd
import spacy

from amep import const
from amep.commands.make_dataset.util import make_count_vectorizer_based_vocab

SPACY_NLP = spacy.load("en", disable=["parser", "tagger", "ner"])
logger = logging.getLogger(__name__)


def _tokenize(text):
    text = " ".join(text)
    text = text.replace("-LRB-", "")
    text = text.replace("-RRB-", " ")
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    tokens = " ".join([t.text.lower() for t in SPACY_NLP(text)])
    return tokens


def make_sst_dataset(args: argparse.Namespace) -> None:

    corpus_reader = nltk.corpus.BracketParseCorpusReader(
        str(const.DATASET_DIRS[args.dataset] / "trees"), r"(train|dev|test)\.txt"
    )

    text = {}
    labels = {}
    keys = ["train", "dev", "test"]
    for k in keys:
        text[k] = [
            x.leaves()
            for x in corpus_reader.parsed_sents(f"{k}.txt")
            if x.label() != "2"
        ]
        labels[k] = [
            int(x.label())
            for x in corpus_reader.parsed_sents(f"{k}.txt")
            if x.label() != "2"
        ]
        logger.info(len(text[k]))

    for k in keys:
        text[k] = [_tokenize(t) for t in text[k]]
        labels[k] = [1 if x >= 3 else 0 for x in labels[k]]

    df_texts = []
    df_label = []
    df_exp_splits = []

    for k in keys:
        df_texts += text[k]
        df_label += labels[k]
        df_exp_splits += [k] * len(text[k])

    df = pd.DataFrame({"text": df_texts, "label": df_label, "exp_split": df_exp_splits})
    print(df.exp_split.value_counts())

    df.to_json(const.DATASET_FPATHS[args.dataset], orient="records", lines=True)
    df[df["exp_split"] == "test"].to_json(
        const.DATASET_FPATHS[args.dataset].parent / "test_dataset.jsonl",
        orient="records",
        lines=True,
    )

    make_count_vectorizer_based_vocab(
        df[df["exp_split"] == "train"].text,
        save_fpath=const.DATASET_FPATHS[args.dataset].parent / "vocab.txt",
        min_df=1,
    )

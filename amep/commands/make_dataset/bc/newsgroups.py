import argparse
import re

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from amep import const
from amep.commands.make_dataset.util import cleaner, make_count_vectorizer_based_vocab
from amep.common.util import filter_by_length


def _cleaner_for_newsgroups(text):
    text = cleaner(text)
    text = re.sub(r"(\W)+", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_newsgroups_dataset(args: argparse.Namespace):

    data_20 = fetch_20newsgroups(
        subset="all", shuffle=True, remove=("headers", "footers", "quotes")
    )
    baseball = np.where(data_20.target == 9)[0]
    hockey = np.where(data_20.target == 10)[0]

    all_sentences = list(baseball) + list(hockey)
    sentences = [data_20.data[i] for i in all_sentences]
    label = [0 if data_20.target[i] == 9 else 1 for i in all_sentences]

    sentences = [_cleaner_for_newsgroups(s) for s in sentences]
    sentences, label = zip(*[(s, t) for s, t in zip(sentences, label) if len(s) != 0])

    train_idx, test_idx = train_test_split(
        range(len(sentences)), stratify=label, test_size=0.2, random_state=13478
    )
    train_idx, dev_idx = train_test_split(
        train_idx,
        stratify=[label[i] for i in train_idx],
        test_size=0.2,
        random_state=13478,
    )

    X_train = [sentences[i] for i in train_idx]
    X_dev = [sentences[i] for i in dev_idx]
    X_test = [sentences[i] for i in test_idx]

    y_train = [label[i] for i in train_idx]
    y_dev = [label[i] for i in dev_idx]
    y_test = [label[i] for i in test_idx]

    texts = {"train": X_train, "test": X_test, "dev": X_dev}
    label = {"train": y_train, "test": y_test, "dev": y_dev}

    df_texts = []
    df_label = []
    df_exp_splits = []

    for key in ["train", "test", "dev"]:
        df_texts += texts[key]
        df_label += label[key]
        df_exp_splits += [key] * len(texts[key])

    df = pd.DataFrame({"text": df_texts, "label": df_label, "exp_split": df_exp_splits})

    make_count_vectorizer_based_vocab(
        df[df["exp_split"] == "train"].text,
        save_fpath=const.DATASET_FPATHS["newsgroups"].parent / "vocab.txt",
        min_df=2,
    )

    df = filter_by_length(df, min_length=6, max_length=500)
    print(df.exp_split.value_counts())

    df.to_json(const.DATASET_FPATHS["newsgroups"], orient="records", lines=True)
    df[df["exp_split"] == "test"].to_json(
        const.DATASET_FPATHS["newsgroups"].parent / "test_dataset.jsonl",
        orient="records",
        lines=True,
    )

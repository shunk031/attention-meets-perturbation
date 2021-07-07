import argparse
import json
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from amep import const
from amep.commands.make_dataset.util import make_count_vectorizer_based_vocab
from amep.common.util import filter_by_length


def _invert_and_join(X, inv):
    X = [[inv[x] for x in doc] for doc in X]
    X = [" ".join(x) for x in X]
    return X


def make_imdb_dataset(args: argparse.Namespace) -> None:
    imdb_full_pkl_fpath = const.DATASET_DIRS["imdb"] / "imdb_full.pkl"
    with imdb_full_pkl_fpath.open("rb") as rf:
        data = pickle.load(rf)

    imdb_word_index_fpath = const.DATASET_DIRS["imdb"] / "imdb_word_index.json"
    with imdb_word_index_fpath.open("r") as rf:
        vocab = json.load(rf)
    inv_vocab = {idx: word for word, idx in vocab.items()}

    (X_train_all, y_train_all), (X_test_all, y_test_all) = data
    train_idx = [i for i, x in enumerate(X_train_all) if len(x) < 400]
    train_idx, dev_idx = train_test_split(train_idx, train_size=0.8, random_state=1378)

    X_train = [X_train_all[i] for i in train_idx]
    y_train = [y_train_all[i] for i in train_idx]

    X_dev = [X_train_all[i] for i in dev_idx]
    y_dev = [y_train_all[i] for i in dev_idx]

    test_idx = [i for i, x in enumerate(X_test_all) if len(x) < 400]
    test_idx, remaining_idx = train_test_split(
        test_idx, train_size=0.2, random_state=1378
    )

    X_test = [X_test_all[i] for i in test_idx]
    y_test = [y_test_all[i] for i in test_idx]

    X_train = _invert_and_join(X_train, inv_vocab)
    X_dev = _invert_and_join(X_dev, inv_vocab)
    X_test = _invert_and_join(X_test, inv_vocab)

    texts = {"train": X_train, "test": X_test, "dev": X_dev}
    labels = {"train": y_train, "test": y_test, "dev": y_dev}

    df_texts = []
    df_label = []
    df_exp_splits = []

    for key in ["train", "test", "dev"]:
        df_texts += texts[key]
        df_label += labels[key]
        df_exp_splits += [key] * len(texts[key])

    df = pd.DataFrame({"text": df_texts, "label": df_label, "exp_split": df_exp_splits})

    make_count_vectorizer_based_vocab(
        df[df["exp_split"] == "train"].text,
        save_fpath=const.DATASET_FPATHS["imdb"].parent / "vocab.txt",
        min_df=10,
    )

    df = filter_by_length(df, min_length=6)
    print(df.exp_split.value_counts())

    df.to_json(const.DATASET_FPATHS["imdb"], orient="records", lines=True)
    df[df["exp_split"] == "test"].to_json(
        const.DATASET_FPATHS["imdb"].parent / "test_dataset.jsonl",
        orient="records",
        lines=True,
    )

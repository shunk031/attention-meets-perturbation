import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from amep import const
from amep.commands.make_dataset.util import cleaner, make_count_vectorizer_based_vocab


def make_ag_news_dataset(args: argparse.Namespace):

    df = {}
    keys = ["train", "test"]
    for k in keys:
        csv_fpath = const.DATASET_DIRS["ag_news"] / "ag_news_csv" / f"{k}.csv"
        df[k] = pd.read_csv(csv_fpath, header=None)
        df[k] = df[k][df[k][0].isin([1, 3])]

    for k in keys:
        texts = list(df[k][2])
        for i in tqdm(range(len(texts))):
            texts[i] = cleaner(texts[i])
        df[k]["text"] = texts

    for k in keys:
        df[k][0] = [1 if (x == 3) else 0 for x in list(df[k][0])]

    df_texts = []
    df_label = []
    df_exp_splits = []
    for key in keys:
        df_texts += list(df[key]["text"])
        df_label += list(df[key][0])
        df_exp_splits += [key] * len(list(df[key]["text"]))

    df = pd.DataFrame({"text": df_texts, "label": df_label, "exp_split": df_exp_splits})

    train_idx, dev_idx = train_test_split(
        df.index[df.exp_split == "train"], test_size=0.15, random_state=16377
    )
    df.loc[dev_idx, "exp_split"] = "dev"
    print(df.exp_split.value_counts())

    make_count_vectorizer_based_vocab(
        df[df["exp_split"] == "train"].text,
        save_fpath=const.DATASET_FPATHS["ag_news"].parent / "vocab.txt",
        min_df=5,
    )

    df.to_json(const.DATASET_FPATHS["ag_news"], orient="records", lines=True)
    df[df["exp_split"] == "test"].to_json(
        const.DATASET_FPATHS["ag_news"].parent / "test_dataset.jsonl",
        orient="records",
        lines=True,
    )

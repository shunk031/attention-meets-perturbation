import argparse
import pathlib
from typing import Dict, List

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from amep import const
from amep.commands.make_dataset.util import make_count_vectorizer_based_vocab

TASKS = [
    "qa1_single-supporting-fact",
    "qa2_two-supporting-facts",
    "qa3_three-supporting-facts",
]

KEYS = ["train", "test"]

SOS = "<SOS>"
EOS = "<EOS>"
PAD = "<0>"
UNK = "<UNK>"


def _parse_data(fpath):

    data, story = [], []
    with open(fpath, "r") as rf:
        for line in rf.readlines():
            task_id, text = line.rstrip("\n").split(" ", 1)

            if task_id == "1":
                story = []

            if text.endswith("."):
                story.append(text[:-1])
            else:
                query, answer, supporting = (x.strip() for x in text.split("\t"))
                sub_story = [x for x in story if x]
                data.append((sub_story, query[:-1], answer))
                story.append("")

    return data


def _flatten_list(x):
    return [y for z in x for y in z]


def _make_babi_dataset(
    task: str,
    paragraph: Dict[str, Dict[str, List[str]]],
    question: Dict[str, Dict[str, List[str]]],
    answer: Dict[str, Dict[str, List[str]]],
    entity_list_fpath: pathlib.Path,
) -> None:

    df_paragraphs = []
    df_questions = []
    df_answers = []
    df_exp_splits = []

    for key in KEYS:
        df_paragraphs += paragraph[key][task]
        df_questions += question[key][task]
        df_answers += answer[key][task]
        df_exp_splits += [key] * len(paragraph[key][task])

    df = pd.DataFrame(
        {
            "paragraph": df_paragraphs,
            "question": df_questions,
            "answer": df_answers,
            "exp_split": df_exp_splits,
        }
    )

    train_idx, dev_idx = train_test_split(
        df.index[df.exp_split == "train"], test_size=0.15, random_state=16377
    )
    df.loc[dev_idx, "exp_split"] = "dev"
    print(df.exp_split.value_counts())

    task_csv_fpath = const.DATASET_DIRS["babi"] / f"babi_{task}_dataset.jsonl"
    df.to_json(task_csv_fpath, orient="records", lines=True)
    df[df["exp_split"] == "test"].to_json(
        const.DATASET_FPATHS["babi"].parent / f"test_{task}_dataset.jsonl",
        orient="records",
        lines=True,
    )

    df_text = pd.concat(
        (
            df[df["exp_split"] == "train"].paragraph,
            df[df["exp_split"] == "train"].question,
        ),
        axis=0,
    )

    make_count_vectorizer_based_vocab(
        df_text,
        save_fpath=const.DATASET_FPATHS["babi"].parent / f"babi_{task}_vocab.txt",
        entity_list_fpath=entity_list_fpath,
        min_df=1,
    )


def make_babi_dataset(args: argparse.Namespace):

    data = {}
    for i, task in enumerate(TASKS):
        data[f"task{i+1}"] = {}
        for key in KEYS:
            fpath = (
                const.DATASET_DIRS["babi"] / f"tasks_1-20_v1-2/en-10k/{task}_{key}.txt"
            )
            data[f"task{i+1}"][key] = list(zip(*_parse_data(fpath)))

    paragraph, question, answer = {}, {}, {}
    for key in KEYS:
        paragraph[key], question[key], answer[key] = {}, {}, {}
        for i in range(len(TASKS)):
            task = f"task{i+1}"
            paragraph[key][task] = [" . ".join(x) for x in data[task][key][0]]
            question[key][task] = data[task][key][1]
            answer[key][task] = data[task][key][2]

    cvec = CountVectorizer(tokenizer=lambda x: x.split(" "), lowercase=False, min_df=1)

    cvec.fit(
        _flatten_list(paragraph["train"].values())
        + _flatten_list(question["train"].values())
    )
    word2idx = cvec.vocabulary_
    for word in cvec.vocabulary_:
        word2idx[word] += 4

    word2idx[PAD] = 0
    word2idx[UNK] = 1
    word2idx[SOS] = 2
    word2idx[EOS] = 3

    entity_list = [k for k, v in word2idx.items() if v >= 4]
    entity_list_fpath = const.DATASET_DIRS["babi"] / "entity_list.txt"
    with entity_list_fpath.open("w") as wf:
        wf.write("\n".join(entity_list))

    for i in range(len(TASKS)):
        task = f"task{i+1}"
        _make_babi_dataset(task, paragraph, question, entity_list_fpath)

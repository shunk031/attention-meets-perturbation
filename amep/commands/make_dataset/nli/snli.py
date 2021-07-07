import argparse
import json
import pathlib
from typing import Dict, List

import pandas as pd
from amep import const
from amep.commands.make_dataset.util import cleaner, make_count_vectorizer_based_vocab
from tqdm import tqdm


def _load_jsonl(file_path: pathlib.Path) -> List[Dict[str, str]]:
    with file_path.open("r") as rf:
        jsonl = [json.loads(line.strip()) for line in rf.readlines()]
    return jsonl


def make_snli_dataset(args: argparse.Namespace):

    snli_dir = "snli_1.0"
    snli_jsonl_dir = const.DATASET_DIRS["snli"] / snli_dir

    data = {"train": [], "dev": [], "test": []}
    for k in data.keys():
        jsonl_fpath = snli_jsonl_dir / f"{snli_dir}_{k}.jsonl"
        jsonl = _load_jsonl(jsonl_fpath)
        data[k].extend(jsonl)

    paragraph_dict, question_dict, answer_dict = {}, {}, {}
    for k in data.keys():
        paragraph_dict[k] = [
            cleaner(x["sentence1"]) for x in tqdm(data[k]) if x["gold_label"] != "-"
        ]
        question_dict[k] = [
            cleaner(x["sentence2"]) for x in tqdm(data[k]) if x["gold_label"] != "-"
        ]
        answer_dict[k] = [
            cleaner(x["gold_label"]) for x in tqdm(data[k]) if x["gold_label"] != "-"
        ]

    entity_list = ["neutral", "contradiction", "entailment"]
    entity_list_fpath = const.DATASET_DIRS["snli"] / "entity_list.txt"
    with entity_list_fpath.open("w") as wf:
        wf.write("\n".join(entity_list))

    df_paragraphs = []
    df_questions = []
    df_answers = []
    df_exp_splits = []
    for k in data.keys():
        df_paragraphs.extend(paragraph_dict[k])
        df_questions.extend(question_dict[k])
        df_answers.extend(answer_dict[k])
        df_exp_splits.extend([k] * len(paragraph_dict[k]))

    df = pd.DataFrame(
        {
            "paragraph": df_paragraphs,
            "question": df_questions,
            "answer": df_answers,
            "exp_split": df_exp_splits,
        }
    )
    print(df.exp_split.value_counts())

    df.to_json(const.DATASET_FPATHS["snli"], orient="records", lines=True)
    df[df["exp_split"] == "test"].to_json(
        const.DATASET_FPATHS["snli"].parent / "test_snli_dataset.jsonl",
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
        save_fpath=const.DATASET_FPATHS["snli"].parent / "vocab.txt",
        entity_list_fpath=entity_list_fpath,
        min_df=3,
    )

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from amep import const
from amep.commands.make_dataset.util import cleaner, make_count_vectorizer_based_vocab
from amep.common.util import read_jsonl


def make_multi_nli_dataset(args: argparse.Namespace):
    multi_nli_dir = "multinli_1.0"
    multi_nli_jsonl_dir = const.DATASET_DIRS["multi_nli"] / multi_nli_dir

    data = {"train": [], "dev_matched": [], "dev_mismatched": []}
    for k in data.keys():
        jsonl_fpath = multi_nli_jsonl_dir / f"{multi_nli_dir}_{k}.jsonl"
        jsonl = read_jsonl(jsonl_fpath)
        data[k].extend(jsonl)

    paragraph_dict, question_dict, answer_dict = {}, {}, {}
    for k in data.keys():
        paragraph_dict[k] = [
            cleaner(x["sentence1"])
            for x in tqdm(data[k], desc=f"[Clean] {k} sentence1", ncols=100)
            if x["gold_label"] != "-"
        ]
        question_dict[k] = [
            cleaner(x["sentence2"])
            for x in tqdm(data[k], desc=f"[Clean] {k} sentence2", ncols=100)
            if x["gold_label"] != "-"
        ]
        answer_dict[k] = [
            cleaner(x["gold_label"])
            for x in tqdm(data[k], desc=f"[Clean] {k} gold_label", ncols=100)
            if x["gold_label"] != "-"
        ]

    entity_list = ["neutral", "contradiction", "entailment"]
    entity_list_fpath = const.DATASET_DIRS["multi_nli"] / "entity_list.txt"
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

    train_idx, dev_idx = train_test_split(
        range(len(paragraph_dict["train"])),
        stratify=answer_dict["train"],
        test_size=0.2,
        random_state=13478,
    )

    df.loc[dev_idx, "exp_split"] = "dev"
    df.loc[df.exp_split == "dev_matched", "exp_split"] = "test_matched"
    df.loc[df.exp_split == "dev_mismatched", "exp_split"] = "test_mismatched"
    print(df.exp_split.value_counts())

    df.to_json(const.DATASET_FPATHS["multi_nli"], orient="records", lines=True)
    df[(df.exp_split == "test_matched") | (df.exp_split == "test_mismatched")].to_json(
        const.DATASET_FPATHS["multi_nli"].parent / "test_multi_nli_dataset.jsonl",
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
        save_fpath=const.DATASET_FPATHS["multi_nli"].parent / "vocab.txt",
        entity_list_fpath=entity_list_fpath,
        min_df=3,
    )

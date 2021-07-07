import argparse
import pathlib
from multiprocessing import Pool
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from amep import const
from amep.commands.make_dataset.util import make_count_vectorizer_based_vocab


def _split_elements_from_question_file(fpath: pathlib.Path) -> List[str]:
    with fpath.open("r") as rf:
        question = rf.read().strip().split("\n\n")
    return question


def _generate_possible_answers(p: str) -> str:
    possible_answers = []
    for w in p.split():
        if w.startswith("@entity"):
            possible_answers.append(w)
    return list(set(possible_answers))


def _split_elements_from_question_file_mp(
    phase: str,
    question_files: Dict[str, List[str]],
    question_fpaths: List[pathlib.Path],
    processes: int,
):
    p = Pool(processes)
    with tqdm(total=(len(question_fpaths)), ncols=100) as pbar:
        func_iter = p.imap(_split_elements_from_question_file, question_fpaths)
        for ret in tqdm(func_iter):
            question_files[phase].append(ret)
            pbar.update()
    p.close()

    return question_files


def make_cnn_dataset(args: argparse.Namespace) -> None:

    cnn_questions_dir_fpath = const.DATASET_DIRS[args.dataset] / "cnn" / "questions"
    cnn_keys = ["training", "validation", "test"]

    question_files = {"train": [], "dev": [], "test": []}
    for key, cnn_key in zip(question_files.keys(), cnn_keys):

        dataset_dir = cnn_questions_dir_fpath / cnn_key
        question_fpaths = [
            fpath for fpath in dataset_dir.iterdir() if fpath.suffix == ".question"
        ]

        question_files = _split_elements_from_question_file_mp(
            key, question_files, question_fpaths, args.num_worker
        )

    paragraph_dict, question_dict, answer_dict = {}, {}, {}
    for key in question_files.keys():
        paragraph_dict[key] = [q[1] for q in question_files[key]]
        question_dict[key] = [q[2] for q in question_files[key]]
        answer_dict[key] = [q[3] for q in question_files[key]]

    entities = {}
    for k in tqdm(paragraph_dict, ncols=50):
        entities[k] = []
        for x in paragraph_dict[k]:
            entities[k] += [y for y in x.split() if y.startswith("@entity")]
        entities[k] = set(entities[k])

    entity_list_fpath = const.DATASET_DIRS["cnn"] / "entity_list.txt"
    with entity_list_fpath.open("w") as wf:
        wf.write("\n".join(list(entities["train"])))

    df_paragraphs = []
    df_questions = []
    df_answers = []
    df_possible_answers = []
    df_exp_splits = []

    for k in question_files.keys():
        df_paragraphs.extend(paragraph_dict[k])
        df_questions.extend(question_dict[k])
        df_answers.extend(answer_dict[k])
        df_possible_answers.extend(
            [_generate_possible_answers(x) for x in paragraph_dict[k]]
        )
        df_exp_splits.extend([k] * len(paragraph_dict[k]))

    df = pd.DataFrame(
        {
            "paragraph": df_paragraphs,
            "question": df_questions,
            "answer": df_answers,
            "exp_split": df_exp_splits,
            "possible_answers": df_possible_answers,
        }
    )
    print(df.exp_split.value_counts())

    df.to_json(const.DATASET_FPATHS[args.dataset], orient="records", lines=True)
    df[df["exp_split"] == "test"].to_json(
        const.DATASET_FPATHS[args.dataset].parent / "test_dataset.jsonl",
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
        save_fpath=const.DATASET_FPATHS[args.dataset].parent / "vocab.txt",
        entity_list_fpath=entity_list_fpath,
        min_df=8,
    )

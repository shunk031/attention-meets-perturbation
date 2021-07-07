from pathlib import Path

import rootpath

ROOT_DIR = Path(rootpath.detect())
DATA_DIR = ROOT_DIR / "dataset"

DATASETS = [
    "sst",
    "imdb",
    "newsgroups",
    "ag_news",
    "cnn",
    "babi",
    "snli",
    "multi_nli",
]


DATASET_DIRS = {dataset: DATA_DIR / dataset for dataset in DATASETS}

DATASET_FPATHS = {
    dataset: DATASET_DIRS[dataset] / f"{dataset}_dataset.jsonl" for dataset in DATASETS
}

DATASETS_FOR_PLOTTING = {
    "sst": "SST",
    "imdb": "IMDB",
    "newsgroups": "20News",
    "ag_news": "AGNews",
    "babi": "bAbI",
    "task1": "bAbI Task 1",
    "task2": "bAbI Task 2",
    "task3": "bAbI Task 3",
    "cnn": "CNN News Article",
    "snli": "SNLI",
    "multi_nli": "MultiNLI",
}

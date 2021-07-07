import pathlib
from typing import Optional, Union

import pandas as pd
import torch


def filter_by_length(
    df: pd.DataFrame, min_length: Optional[int] = None, max_length: Optional[int] = None
) -> pd.DataFrame:

    df.text = df.text.astype("str")
    lens = df.text.str.split().apply(len)
    min_len = lens.min() if min_length is None else min_length
    max_len = lens.max() if max_length is None else max_length
    mask = (lens > min_len) & (lens < max_len)

    return df.loc[mask]


def convert_binary_prob_to_multi_prob(logit: torch.Tensor) -> torch.Tensor:
    one_minus_logit = 1 - logit
    return torch.stack((one_minus_logit, logit), dim=-1)


def read_jsonl(file_path: Union[str, pathlib.Path]) -> pd.DataFrame:
    return pd.read_json(file_path, lines=True, orient="records")

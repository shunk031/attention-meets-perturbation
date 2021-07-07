from typing import Dict, List


def _get_subset_type_based_index(
    metadata: List[Dict[str, str]], subset_type: str
) -> List[int]:

    idx_metadata = enumerate(metadata)  # [(0, metadata1), (1, metadata2), ...]
    idx_metadata = filter(lambda x: x[1]["subset_type"] == subset_type, idx_metadata)
    idx_subset_type = list(map(lambda x: x[0], idx_metadata))

    return idx_subset_type


def calc_subset_type_based_metric(eval_metric, logits, answer, metadata, subset_type):
    idx_subset_type = _get_subset_type_based_index(metadata, subset_type)

    if idx_subset_type:
        eval_metric(logits[idx_subset_type], answer[idx_subset_type])

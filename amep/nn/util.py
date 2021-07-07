import torch


def weighted_loss(
    bce_loss: torch.Tensor, label: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    weight = label * weight.transpose(0, 1) + (1 - label)
    return (bce_loss * weight).mean(dim=1).sum()


def masked_fill_for_qa(
    prediction: torch.Tensor, entity_mask: torch.Tensor = None
) -> torch.Tensor:
    if entity_mask is not None:
        return prediction.masked_fill((1 - entity_mask).bool(), float("-inf"))
    return prediction

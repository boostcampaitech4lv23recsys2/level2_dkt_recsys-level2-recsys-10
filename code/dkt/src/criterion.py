import torch.nn as nn


def get_criterion(pred, target):
    # loss = nn.BCEWithLogitsLoss(reduction="none")
    loss = nn.BCELoss(reduction="none")
    return loss(pred, target)

import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed


def batch_to_torch(batch, device):
    torch_batch = {}
    for key in batch:
        torch_batch[key] = torch.from_numpy(batch[key]).to(device)
    return torch_batch


def mse_loss(val, target):
    return torch.mean(torch.square(val - target))


def cross_ent_loss(logits, target):
    if len(target.shape) == 1:
        label = F.one_hot(target, num_classes=2)
    else:
        label = target
    loss = F.cross_entropy(input=logits, target=label)
    return loss




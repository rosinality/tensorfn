from typing import List

import torch
from torch import nn
from pydantic import StrictInt, StrictFloat


class StrictLinear(nn.Module):
    def __init__(self, in_dim: StrictInt, out_dim: StrictInt):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)


class StrictFeedForward(nn.Module):
    def __init__(
        self, in_dim: StrictInt, dim: StrictInt, out_dim: StrictInt, dropout=0.1
    ):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)


def model_runner(x: torch.Tensor, encoder: nn.Module):
    return x, encoder


def model_wrapper(encoder: nn.Module, n_layer: StrictInt):
    return encoder, n_layer


def return_list(x):
    return [x]


def model_list_wrapper(encoder: List[nn.Module], weight: StrictFloat):
    return encoder, weight

import torch
from torch.nn import Module

from einops.layers.torch import Rearrange

# class

class ASAC(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

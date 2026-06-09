import torch
from torch.nn import Module

from einops.layers.torch import Rearrange

from x_transformers import Decoder

from vector_quantize_pytorch import VectorQuantize

from ema_pytorch import EMA

# class

class ASAC(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

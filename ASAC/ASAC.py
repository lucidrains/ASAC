import torch
from torch.nn import Module, Linear
import torch.nn.functional as F

from einops import einsum
from einops.layers.torch import Rearrange

from x_transformers import Decoder

from vector_quantize_pytorch import VectorQuantize

from ema_pytorch import EMA

# helpers

def exists(v):
    return v is not None

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.to_qkv = Linear(dim, dim_inner * 3, bias = False)
        self.combine_heads = Linear(dim_inner, dim, bias = False)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

    def forward(
        self,
        tokens,
    ):

        q, k, v = self.to_qkv(tokens).chunk(3, dim = -1)
        q, k, v = (self.split_heads(t) for t in (q, k, v))

        q = q * self.scale

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)
        return self.combine_heads(out)

# class

class ASAC(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

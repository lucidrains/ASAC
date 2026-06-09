from __future__ import annotations

import torch
from torch import nn, tensor
from torch.nn import Module, Linear
import torch.nn.functional as F

from einops import einsum
from einops.layers.torch import Rearrange

from x_transformers import Decoder

from x_mlps_pytorch import MLP

from vector_quantize_pytorch import VectorQuantize

from ema_pytorch import EMA

from torch_einops_utils import pack_with_inverse

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        attn_schema: Module | None = None,
        attn_add_residual = True # they had to add a residual for stability
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.to_qkv = Linear(dim, dim_inner * 3, bias = False)
        self.combine_heads = Linear(dim_inner, dim, bias = False)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.attn_schema = attn_schema
        self.attn_add_residual = attn_add_residual and attn_schema

        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        tokens, # (b h w d)
    ):
        tokens, inverse_pack = pack_with_inverse(tokens, 'b * d')

        q, k, v = self.to_qkv(tokens).chunk(3, dim = -1)
        q, k, v = (self.split_heads(t) for t in (q, k, v))

        q = q * self.scale

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        orig_sim = sim

        # the proposal

        aux_loss = self.zero

        if exists(self.attn_schema):
            sim, indices, aux_loss = self.attn_schema(orig_sim)

        if self.attn_add_residual:
            sim = sim + orig_sim

        # attend

        attn = sim.softmax(dim = -1)

        # aggregate and combine out

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)
        attended = self.combine_heads(out)

        # bring back the packed dimensions

        attended = inverse_pack(attended)

        return attended, indices, aux_loss

# attention autoencoder

class AttentionSchema(Module):
    def __init__(
        self,
        dim,
        dim_bottleneck,
        kl_div_loss = True,
        detach_target = True,
        encoder: Module | None = None,
        decoder: Module | None = None,
        **vq_kwargs
    ):
        super().__init__()

        if not exists(encoder):
            encoder = MLP(dim, dim_bottleneck, activation = nn.LeakyReLU())

        self.encoder = encoder

        self.vq = VectorQuantize(dim_bottleneck, **vq_kwargs)

        if not exists(decoder):
            decoder = MLP(dim_bottleneck, dim, activation = nn.LeakyReLU())

        self.decoder = decoder

        self.kl_div_loss = kl_div_loss
        self.detach_target = detach_target

    def forward(
        self,
        attn_sim,
        return_loss = None
    ):
        return_loss = default(return_loss, self.training)

        attn_features, inverse_pack = pack_with_inverse(attn_sim, 'b *')

        encoded = self.encoder(attn_features)

        quantized, indices, commit_loss = self.vq(encoded)

        decoded = self.decoder(quantized)

        recon = inverse_pack(decoded)

        # loss, mse as in paper or reverse kl

        if return_loss:
            if self.detach_target:
                attn_sim = attn_sim.detach()

            if self.kl_div_loss:
                recon_loss = F.kl_div(
                    attn_sim.log_softmax(dim = -1),
                    recon,
                    reduction = 'batchmean'
                )
            else:
                recon_loss = F.mse_loss(recon, attn_sim)

        # total

        total_loss = recon_loss + commit_loss

        loss_breakdown = (recon_loss, commit_loss)

        return recon, indices, total_loss

# class

class ASAC(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

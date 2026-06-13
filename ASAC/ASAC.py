from __future__ import annotations

from collections import namedtuple
import torch
from torch import nn, tensor
from torch.nn import Module, Linear, ModuleList
import torch.nn.functional as F

from einops import einsum, reduce
from einops.layers.torch import Rearrange

from x_transformers import Decoder

from x_mlps_pytorch import MLP

from vector_quantize_pytorch import VectorQuantize

from ema_pytorch import EMA

from torch_einops_utils import pack_with_inverse, maybe

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# return types

AttentionReturn = namedtuple('AttentionReturn', ['attended', 'indices', 'aux_loss', 'aux_loss_breakdown', 'dot_sim'])
ASACReturn = namedtuple('ASACReturn', ['logits', 'aux_loss', 'aux_loss_breakdown', 'dot_sims'])

# feedforward

def FeedForward(dim, expansion_factor = 4.):
    dim_inner = int(dim * expansion_factor)
    return nn.Sequential(
        nn.RMSNorm(dim),
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

# embedding

def PatchEmbedding(dim, patch_size, channels = 3):
    patch_dim = channels * (patch_size ** 2)

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.RMSNorm(patch_dim),
        Linear(patch_dim, dim),
        nn.RMSNorm(dim),
    )

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        k_rmsnorm = True,
        attn_schema: Module | None = None,
        attn_add_residual = True # they had to add a residual for stability
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.norm = nn.RMSNorm(dim)

        self.to_qkv = Linear(dim, dim_inner * 3, bias = False)
        self.combine_heads = Linear(dim_inner, dim, bias = False)

        self.k_rmsnorm = nn.RMSNorm(dim_head) if k_rmsnorm else None

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.attn_schema = attn_schema
        self.attn_add_residual = attn_add_residual and attn_schema

        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        tokens, # (b h w d)
        pre_softmax_attn_gates = None,
        post_softmax_attn_gates = None,
        attn_schema_target = None
    ):
        tokens = self.norm(tokens)

        tokens, inverse_pack = pack_with_inverse(tokens, 'b * d')

        q, k, v = self.to_qkv(tokens).chunk(3, dim = -1)
        q, k, v = (self.split_heads(t) for t in (q, k, v))

        k = maybe(self.k_rmsnorm)(k)

        q = q * self.scale

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        orig_sim = sim

        # the proposal

        aux_loss = self.zero
        aux_loss_breakdown = (self.zero, self.zero)
        indices = None

        if exists(self.attn_schema):
            sim, indices, aux_loss, aux_loss_breakdown = self.attn_schema(orig_sim, target_sim = attn_schema_target)

        if self.attn_add_residual:
            sim = (sim + orig_sim) * 0.5

        # modulate

        if exists(pre_softmax_attn_gates):
            sim = sim + pre_softmax_attn_gates

        # attend

        attn = sim.softmax(dim = -1)

        # modulate

        if exists(post_softmax_attn_gates):
            attn = attn * post_softmax_attn_gates

        # aggregate and combine out

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)
        attended = self.combine_heads(out)

        # bring back the packed dimensions

        attended = inverse_pack(attended)

        return AttentionReturn(attended, indices, aux_loss, aux_loss_breakdown, orig_sim)

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
        recon_loss_weight = 1.,
        commit_loss_weight = 1.,
        **vq_kwargs
    ):
        super().__init__()

        if not exists(encoder):
            encoder = MLP(dim, dim_bottleneck, dim_bottleneck, activation = nn.LeakyReLU())

        self.encoder = encoder

        self.vq = VectorQuantize(dim_bottleneck, **vq_kwargs)

        if not exists(decoder):
            decoder = MLP(dim_bottleneck, dim_bottleneck, dim, activation = nn.LeakyReLU())

        self.decoder = decoder

        self.kl_div_loss = kl_div_loss
        self.detach_target = detach_target

        self.recon_loss_weight = recon_loss_weight
        self.commit_loss_weight = commit_loss_weight

        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        attn_sim,
        return_loss = None,
        target_sim = None
    ):
        return_loss = default(return_loss, self.training)

        attn_features, inverse_pack = pack_with_inverse(attn_sim, 'b *')

        encoded = self.encoder(attn_features)

        quantized, indices, commit_loss = self.vq(encoded)

        decoded = self.decoder(quantized)

        recon = inverse_pack(decoded)

        # loss, mse as in paper or reverse kl

        recon_loss = self.zero

        target = default(target_sim, attn_sim)

        if return_loss:
            if self.detach_target:
                target = target.detach()

            if self.kl_div_loss:
                recon_loss = F.kl_div(
                    target.log_softmax(dim = -1),
                    recon.softmax(dim = -1),
                    reduction = 'none'
                ).sum(dim = -1).mean()
            else:
                recon_loss = F.mse_loss(recon, target)

        # total

        total_loss = recon_loss * self.recon_loss_weight + commit_loss * self.commit_loss_weight

        return recon, indices, total_loss, (recon_loss, commit_loss)

# class

class ASAC(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        to_embedding,
        seq_len = None,
        dim_head = 64,
        num_classes = 10,
        use_asac = False,
        dim_bottleneck = 256,
        vq_codebook_size = 256,
        recon_loss_weight = 1.,
        commit_loss_weight = 1.,
        kl_div_loss = True
    ):
        super().__init__()

        self.depth = depth

        self.to_embedding = to_embedding
        self.pos_embedding = nn.Parameter(torch.randn(seq_len, dim)) if exists(seq_len) else None

        self.layers = ModuleList([])

        for _ in range(depth):
            attn_schema = AttentionSchema(
                dim = heads * (seq_len ** 2),
                dim_bottleneck = dim_bottleneck,
                codebook_size = vq_codebook_size,
                recon_loss_weight = recon_loss_weight,
                commit_loss_weight = commit_loss_weight,
                kl_div_loss = kl_div_loss
            ) if use_asac and exists(seq_len) else None

            self.layers.append(ModuleList([
                Attention(dim, dim_head = dim_head, heads = heads, attn_schema = attn_schema),
                FeedForward(dim)
            ]))

        self.to_logits = nn.Sequential(
            nn.RMSNorm(dim),
            Linear(dim, num_classes)
        )

    def forward(self, x, attn_schema_targets = None):
        x = self.to_embedding(x)

        if exists(self.pos_embedding):
            x = x + self.pos_embedding

        total_aux_loss = total_recon_loss = total_commit_loss = 0.

        attn_schema_targets = default(attn_schema_targets, [None] * self.depth)
        dot_sims = []

        for (attn, ff), target in zip(self.layers, attn_schema_targets):
            attn_out, indices, aux_loss, (recon_loss, commit_loss), dot_sim = attn(x, attn_schema_target = target)

            dot_sims.append(dot_sim)

            x = attn_out + x
            x = ff(x) + x

            total_aux_loss = total_aux_loss + aux_loss
            total_recon_loss = total_recon_loss + recon_loss
            total_commit_loss = total_commit_loss + commit_loss

        x = reduce(x, 'b n d -> b d', 'mean')

        logits = self.to_logits(x)

        return ASACReturn(logits, total_aux_loss, (total_recon_loss / self.depth, total_commit_loss / self.depth), dot_sims)

class EMA_ASAC(Module):
    def __init__(self, asac_model, ema_decay = 0.999, **ema_kwargs):
        super().__init__()
        self.asac = asac_model
        self.ema_model = EMA(asac_model, beta = ema_decay, **ema_kwargs)

    def forward(self, x, use_ema = False):
        if use_ema:
            return self.ema_model(x)

        if not self.training:
            return self.asac(x)

        # get EMA targets
        with torch.no_grad():
            self.ema_model.eval()
            ema_outputs = self.ema_model.ema_model(x)
            ema_targets = [sim.detach() for sim in ema_outputs.dot_sims]
            self.ema_model.train()

        return self.asac(x, attn_schema_targets = ema_targets)

    def update(self):
        self.ema_model.update()

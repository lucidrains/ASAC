from __future__ import annotations

from collections import namedtuple
import torch
from torch import nn, tensor
from torch.nn import Module, Linear, ModuleList
import torch.nn.functional as F

from einops import einsum, reduce, rearrange
from einops.layers.torch import Rearrange
import einx

from x_transformers import Decoder, AutoregressiveWrapper, TransformerWrapper

from x_mlps_pytorch import MLP

from vector_quantize_pytorch import VectorQuantize

from ema_pytorch import EMA

from torch_einops_utils import pack_with_inverse, maybe

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def is_empty(t):
    return len(t) == 0



# return types

AttentionReturn = namedtuple('AttentionReturn', ['attended', 'indices', 'aux_loss', 'aux_loss_breakdown', 'attn_sim'])
ASACReturn = namedtuple('ASACReturn', ['logits', 'aux_loss', 'aux_loss_breakdown', 'attn_sims', 'attn_schema_indices', 'attn_schema_autoregressive_loss'])

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
        attn_add_residual = True, # they had to add a residual for stability
        stochastic_sample_attn = False
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

        self.stochastic_sample_attn = stochastic_sample_attn

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

        if self.stochastic_sample_attn:
            attn = F.gumbel_softmax(sim, tau = 1., hard = True, dim = -1)
        else:
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

        return AttentionReturn(
            attended,
            indices,
            aux_loss,
            aux_loss_breakdown,
            orig_sim
        )

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
        causal = False,
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

        self.causal = causal

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

        target = default(target_sim, attn_sim)

        # handle causal mask
        # zero out upper right for autoencoding

        if self.causal:
            n = attn_sim.shape[-1]
            mask = attn_sim.new_ones(n, n, dtype = torch.bool).triu(1)
            attn_sim = attn_sim.masked_fill(mask, 0.)

        attn_features, inverse_pack = pack_with_inverse(attn_sim, 'b *')

        encoded = self.encoder(attn_features)

        quantized, indices, commit_loss = self.vq(encoded)

        recon = inverse_pack(self.decoder(quantized))

        # mask to -inf if causal

        if self.causal:
            mask_value = -torch.finfo(attn_sim.dtype).max
            recon = recon.masked_fill(mask, mask_value)

        # early return if no loss

        if not return_loss:
            total_loss = commit_loss * self.commit_loss_weight
            return recon, indices, total_loss, (self.zero, commit_loss)

        # loss, mse as in paper or reverse kl

        target = target.detach() if self.detach_target else target
        target = target.masked_fill(mask, mask_value) if self.causal else target

        if self.kl_div_loss:
            # kl div

            loss = F.kl_div(
                target.log_softmax(dim = -1),
                recon.softmax(dim = -1),
                reduction = 'none'
            )

            loss = loss.masked_fill(mask, 0.) if self.causal else loss
            recon_loss = loss.sum(dim = -1).mean()
        else:
            # mse

            loss = F.mse_loss(recon, target, reduction = 'none')

            if self.causal:
                valid_fraction = (~mask).float().mean()
                recon_loss = loss.masked_fill(mask, 0.).mean() / valid_fraction
            else:
                recon_loss = loss.mean()

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
        kl_div_loss = True,
        stochastic_sample_attn = False,
        awareness_dropout_prob = 0.,
        awareness_model_depth = 2,
        **awareness_model_kwargs
    ):
        super().__init__()

        assert depth >= 2, 'depth must be at least 2'

        self.depth = depth

        self.awareness_dropout_prob = awareness_dropout_prob
        self.has_awareness_dropout = awareness_dropout_prob > 0.

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
                Attention(dim, dim_head = dim_head, heads = heads, attn_schema = attn_schema, stochastic_sample_attn = stochastic_sample_attn),
                FeedForward(dim)
            ]))

        # autoregressive awareness model (attention schema theory)

        self.awareness_transformer = None
        self.awareness_model = None

        if use_asac and exists(seq_len):
            self.awareness_transformer = TransformerWrapper(
                num_tokens = vq_codebook_size,
                max_seq_len = depth,
                attn_layers = Decoder(
                    dim = dim,
                    depth = awareness_model_depth,
                    heads = heads,
                    rotary_pos_emb = True,
                    **awareness_model_kwargs
                )
            )

            self.awareness_model = AutoregressiveWrapper(self.awareness_transformer)
            self.awareness_to_logits = MLP(dim, dim, num_classes, activation = nn.LeakyReLU())

        # zero buffer for auxiliary losses

        self.register_buffer('zero', tensor(0.), persistent = False)

        self.to_logits = nn.Sequential(
            nn.RMSNorm(dim),
            Linear(dim, num_classes)
        )

    def forward(self, x, attn_schema_targets = None, use_awareness = True):
        batch = x.shape[0]

        x = self.to_embedding(x)

        if exists(self.pos_embedding):
            x = x + self.pos_embedding

        total_aux_loss = total_recon_loss = total_commit_loss = 0.

        attn_schema_targets = default(attn_schema_targets, [None] * self.depth)
        attn_sims = []
        attn_schema_indices = []

        for (attn, ff), target in zip(self.layers, attn_schema_targets):
            attn_out, indices, aux_loss, (recon_loss, commit_loss), attn_sim = attn(x, attn_schema_target = target)

            attn_sims.append(attn_sim)
            if exists(indices):
                attn_schema_indices.append(indices)

            x = attn_out + x
            x = ff(x) + x

            total_aux_loss = total_aux_loss + aux_loss
            total_recon_loss = total_recon_loss + recon_loss
            total_commit_loss = total_commit_loss + commit_loss

        x = reduce(x, 'b n d -> b d', 'mean')

        logits = self.to_logits(x)

        attn_schema_autoregressive_loss = self.zero

        _attn_schema_indices = attn_schema_indices
        attn_schema_indices = None

        if not is_empty(_attn_schema_indices):
            attn_schema_indices = rearrange(_attn_schema_indices, 'depth b ... -> b (depth ...)')

            if exists(self.awareness_model):
                attn_schema_autoregressive_loss = self.awareness_model(attn_schema_indices)

                if use_awareness:
                    _, embeddings = self.awareness_transformer(attn_schema_indices, return_logits_and_embeddings = True)
                    last_embedding = embeddings[:, -1, :]

                    awareness_logits = self.awareness_to_logits(last_embedding)

                    if self.training and self.has_awareness_dropout:
                        drop_awareness = torch.rand(batch, device = x.device) < self.awareness_dropout_prob
                        awareness_logits = einx.where('b, , b d -> b d', drop_awareness, 0., awareness_logits)

                    logits = logits + awareness_logits

        return ASACReturn(
            logits,
            total_aux_loss,
            (total_recon_loss / self.depth, total_commit_loss / self.depth),
            attn_sims,
            attn_schema_indices,
            attn_schema_autoregressive_loss
        )

class EMA_ASAC(Module):
    def __init__(
        self,
        asac,
        ema_decay = 0.999,
        **ema_kwargs
    ):
        super().__init__()
        self.asac = asac

        self.ema_model = EMA(asac, beta = ema_decay, **ema_kwargs)

    def update(self):
        self.ema_model.update()

    def forward(
        self,
        *args,
        use_ema = False,
        **kwargs
    ):
        if not self.training or use_ema:
            return self.ema_model(*args, **kwargs)

        with torch.no_grad():
            self.ema_model.eval()
            ema_outputs = self.ema_model(*args, **kwargs)
            ema_targets = [sim.detach() for sim in ema_outputs.attn_sims]
            self.ema_model.train()

        kwargs.update(attn_schema_targets = ema_targets)
        return self.asac(*args, **kwargs)

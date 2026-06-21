from __future__ import annotations

from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, tensor
from torch.nn import Module, Linear, ModuleList

from einops import einsum, reduce, rearrange, repeat
from einops.layers.torch import Rearrange
import einx

from x_mlps_pytorch import MLP
from vector_quantize_pytorch import VectorQuantize

from ema_pytorch import EMA
from torch_einops_utils import pack_with_inverse, maybe, tree_map_tensor

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def is_empty(t):
    return len(t) == 0

# return types

AttentionReturn = namedtuple('AttentionReturn', ['attended', 'indices', 'aux_loss', 'aux_loss_breakdown', 'attn_sim', 'attn', 'cache'])
ASACReturn = namedtuple('ASACReturn', ['logits', 'aux_loss', 'aux_loss_breakdown', 'attn_sims', 'attn_schema_indices', 'attn_schema_autoregressive_loss', 'awareness_attn_sims', 'meta_awareness_attn_sims', 'attns', 'awareness_attns', 'meta_awareness_attns', 'awareness_driven_attns'])
AuxLossBreakdown = namedtuple('AuxLossBreakdown', ['recon_loss', 'commit_loss'])
AttentionSchemaReturn = namedtuple('AttentionSchemaReturn', ['recon', 'indices', 'loss', 'loss_breakdown'])
AwarenessReturn = namedtuple('AwarenessReturn', ['logits', 'embeds', 'attn_sims', 'attns', 'aux_loss', 'attn_schema_indices'])

# feedforward

def FeedForward(dim, expansion_factor = 4.):
    dim_inner = int(dim * expansion_factor)
    return nn.Sequential(
        nn.RMSNorm(dim),
        Linear(dim, dim_inner),
        nn.GELU(),
        Linear(dim_inner, dim)
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
        attn_add_residual = True,
        stochastic_sample_attn = False,
        causal = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.causal = causal

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
        tokens, # (b h w d) or (b n d)
        pre_softmax_attn_gates = None,
        post_softmax_attn_gates = None,
        attn_schema_target = None,
        cache = None
    ):
        tokens = self.norm(tokens)

        tokens, inverse_pack = pack_with_inverse(tokens, 'b * d')

        q, k, v = self.to_qkv(tokens).chunk(3, dim = -1)
        q, k, v = (self.split_heads(t) for t in (q, k, v))

        k = maybe(self.k_rmsnorm)(k)

        # kv caching

        if exists(cache):
            past_k, past_v = cache
            k = torch.cat((past_k, k), dim = -2)
            v = torch.cat((past_v, v), dim = -2)

        new_cache = (k, v)

        q = q * self.scale

        # similarity

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        orig_sim = sim

        # the proposal

        aux_loss = self.zero
        aux_loss_breakdown = AuxLossBreakdown(self.zero, self.zero)
        indices = None

        if exists(self.attn_schema):
            sim, indices, aux_loss, aux_loss_breakdown = self.attn_schema(orig_sim, target_sim = attn_schema_target)

        if self.attn_add_residual:
            sim = (sim + orig_sim) * 0.5

        # modulate

        if exists(pre_softmax_attn_gates):
            sim = sim + pre_softmax_attn_gates

        # causal masking

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), device = sim.device, dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

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
            orig_sim,
            attn,
            new_cache
        )

class AwarenessDrivenAttention(Module):
    def __init__(self, dim, seq_len, dim_head, heads = 8, kv_heads = 2, k_rmsnorm = True):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.heads = heads
        self.kv_heads = kv_heads

        assert (heads % kv_heads) == 0, 'heads must be divisible by kv_heads'

        self.norm = nn.RMSNorm(dim)

        self.q_rmsnorm = nn.RMSNorm(dim_head)
        self.k_rmsnorm = nn.RMSNorm(dim_head) if k_rmsnorm else None

        self.awareness_to_q_and_gate = nn.Sequential(
            Linear(dim, dim * 2),
            nn.SiLU(),
            Linear(dim * 2, heads * seq_len * dim_head + heads * dim_head)
        )

        self.to_v = Linear(dim, kv_heads * dim_head, bias = False)
        self.to_out = Linear(heads * dim_head, dim, bias = False)

    def forward(self, x, awareness_embed):
        n = x.shape[-2]

        x = self.norm(x)

        v = self.to_v(x)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.kv_heads)
        k = maybe(self.k_rmsnorm)(v)

        # repeat k and v for gqa
        num_kv_groups = self.heads // self.kv_heads

        k = repeat(k, 'b h n d -> b (h g) n d', g = num_kv_groups)
        v = repeat(v, 'b h n d -> b (h g) n d', g = num_kv_groups)

        # generate pseudo queries from awareness embed
        # and use shared key / values from the backbone itself

        q_and_gate = self.awareness_to_q_and_gate(awareness_embed)
        q, gate = q_and_gate.split([self.heads * self.seq_len * self.dim_head, self.heads * self.dim_head], dim = -1)

        q = rearrange(q, 'b (h n d) -> b h n d', h = self.heads, n = self.seq_len)
        q = self.q_rmsnorm(q)

        q = q[:, :, :n]

        attn_matrix = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale

        attn = attn_matrix.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        gate = rearrange(gate, 'b (h d) -> b h 1 d', h = self.heads)
        out = out * gate.sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, attn

class AwarenessDrivenModulation(Module):
    def __init__(
        self,
        dim,
        depth,
        seq_len,
        dim_head,
        heads = 8,
        kv_heads = 2,
        awareness_dropout_prob = 0.
    ):
        super().__init__()
        self.depth = depth
        self.awareness_dropout_prob = awareness_dropout_prob
        self.has_awareness_dropout = awareness_dropout_prob > 0.

        self.layers = ModuleList([ModuleList([
            AwarenessDrivenAttention(dim, seq_len = seq_len, dim_head = dim_head, heads = heads, kv_heads = kv_heads),
            FeedForward(dim)
        ]) for _ in range(depth)])

    def forward(
        self,
        x,               # b n d
        awareness_embed  # b d
    ):
        attns = []

        drop_awareness_mask = None
        if self.training and self.has_awareness_dropout:
            drop_awareness_mask = torch.rand(x.shape[0], device = x.device) < self.awareness_dropout_prob

        orig_x = x

        for attn, ff in self.layers:
            out, post_softmax_attn = attn(x, awareness_embed)
            attns.append(post_softmax_attn)

            x = out + x
            x = ff(x) + x

        # awareness dropout

        if exists(drop_awareness_mask):
            x = einx.where('b, , b n d -> b n d', drop_awareness_mask, orig_x, x)

        return x, attns

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

        if self.causal:
            i, j = attn_sim.shape[-2:]
            mask = torch.ones((i, j), device = attn_sim.device, dtype = torch.bool).triu(j - i + 1)
            attn_sim = attn_sim.masked_fill(mask, 0.)

        attn_features, inverse_pack = pack_with_inverse(attn_sim, 'b *')

        encoded = self.encoder(attn_features)

        quantized, indices, commit_loss = self.vq(encoded)

        recon = inverse_pack(self.decoder(quantized))

        if self.causal:
            recon = recon.masked_fill(mask, 0.)

        # early return if no loss

        if not return_loss:
            total_loss = commit_loss * self.commit_loss_weight
            return AttentionSchemaReturn(recon, indices, total_loss, AuxLossBreakdown(self.zero, commit_loss))

        # loss, mse as in paper or reverse kl

        mask_value = -torch.finfo(attn_sim.dtype).max

        target = target.detach() if self.detach_target else target
        target = target.masked_fill(mask, mask_value) if self.causal else target
        recon_for_loss = recon.masked_fill(mask, mask_value) if self.causal else recon

        if self.kl_div_loss:
            # kl div
            loss = F.kl_div(
                target.log_softmax(dim = -1),
                recon_for_loss.softmax(dim = -1),
                reduction = 'none'
            )

            loss = loss.masked_fill(mask, 0.) if self.causal else loss
            recon_loss = loss.sum(dim = -1).mean()
        else:
            # mse
            loss = F.mse_loss(recon_for_loss, target, reduction = 'none')

            if self.causal:
                valid_fraction = (~mask).float().mean()
                recon_loss = loss.masked_fill(mask, 0.).mean() / valid_fraction
            else:
                recon_loss = loss.mean()

        total_loss = recon_loss * self.recon_loss_weight + commit_loss * self.commit_loss_weight

        return AttentionSchemaReturn(recon, indices, total_loss, AuxLossBreakdown(recon_loss, commit_loss))

# autoregressive awareness model

class AutoregressiveAwareness(Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        num_tokens,
        max_seq_len,
        dim_head = 64,
        attn_schema: Module | None = None,
        stochastic_sample_attn = False
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layers = ModuleList([ModuleList([
            Attention(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                causal = True,
                attn_schema = attn_schema,
                stochastic_sample_attn = stochastic_sample_attn
            ),
            FeedForward(dim)
        ]) for _ in range(depth)])

        self.norm = nn.RMSNorm(dim)
        self.to_logits = Linear(dim, num_tokens)

    def forward(
        self,
        x,
        attn_schema_targets = None
    ):
        n = x.shape[-1]

        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(n, device = x.device))

        total_aux_loss = 0.
        attn_sims = []
        attns = []
        attn_schema_indices = []

        attn_schema_targets = default(attn_schema_targets, [None] * len(self.layers))

        for (attn, ff), target_sim in zip(self.layers, attn_schema_targets):
            attn_out = attn(x, attn_schema_target = target_sim)

            x = attn_out.attended + x
            x = ff(x) + x

            total_aux_loss = total_aux_loss + attn_out.aux_loss
            attn_sims.append(attn_out.attn_sim)
            attns.append(attn_out.attn)

            if exists(attn_out.indices):
                attn_schema_indices.append(attn_out.indices)

        embeds = self.norm(x)
        logits = self.to_logits(embeds)

        return AwarenessReturn(logits, embeds, attn_sims, attns, total_aux_loss, attn_schema_indices)

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
        use_awareness = False,
        use_meta_awareness = False,
        dim_bottleneck = 256,
        vq_codebook_size = 256,
        recon_loss_weight = 1.,
        commit_loss_weight = 1.,
        kl_div_loss = True,
        stochastic_sample_attn = False,
        awareness_dropout_prob = 0.,
        awareness_model_depth = 2,
        awareness_driven_depth = 1,
        awareness_driven_heads = 8,
        awareness_driven_kv_heads = 2
    ):
        super().__init__()

        assert depth >= 2, 'depth must be at least 2'
        self.awareness_driven_depth = awareness_driven_depth
        assert not (use_meta_awareness and not use_awareness), 'meta awareness should only be on if awareness decoder is on already'

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

        self.awareness_model = None
        self.meta_awareness_model = None

        if use_asac and use_awareness and exists(seq_len):
            awareness_attn_schema = AttentionSchema(
                dim = heads * (depth ** 2),
                dim_bottleneck = dim_bottleneck,
                codebook_size = vq_codebook_size,
                recon_loss_weight = recon_loss_weight,
                commit_loss_weight = commit_loss_weight,
                kl_div_loss = kl_div_loss,
                causal = True
            )

            self.awareness_model = AutoregressiveAwareness(
                dim = dim,
                depth = awareness_model_depth,
                heads = heads,
                dim_head = dim_head,
                num_tokens = vq_codebook_size,
                max_seq_len = depth,
                attn_schema = awareness_attn_schema,
                stochastic_sample_attn = stochastic_sample_attn
            )

            self.awareness_driven_modulation = AwarenessDrivenModulation(
                dim = dim,
                depth = awareness_driven_depth,
                seq_len = seq_len,
                dim_head = dim_head,
                heads = awareness_driven_heads,
                kv_heads = awareness_driven_kv_heads,
                awareness_dropout_prob = awareness_dropout_prob
            )

            # meditation

            if use_meta_awareness:
                self.meta_awareness_model = AutoregressiveAwareness(
                    dim = dim,
                    depth = awareness_model_depth,
                    heads = heads,
                    dim_head = dim_head,
                    num_tokens = vq_codebook_size,
                    max_seq_len = awareness_model_depth,
                    stochastic_sample_attn = stochastic_sample_attn
                )

        self.to_logits = nn.Sequential(
            nn.RMSNorm(dim),
            Linear(dim, num_classes)
        )

        # zero buffer for auxiliary losses

        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        x,
        attn_schema_targets = None,
        awareness_attn_schema_targets = None,
        meta_awareness_attn_schema_targets = None,
        use_awareness = True
    ):
        batch = x.shape[0]

        x = self.to_embedding(x)

        if exists(self.pos_embedding):
            x = x + self.pos_embedding

        total_aux_loss = total_recon_loss = total_commit_loss = 0.

        attn_schema_targets = default(attn_schema_targets, [None] * len(self.layers))
        attn_sims = []
        attns = []
        attn_schema_indices = []

        for (attn, ff), target in zip(self.layers, attn_schema_targets):
            attn_out = attn(x, attn_schema_target = target)

            attn_sims.append(attn_out.attn_sim)
            attns.append(attn_out.attn)

            if exists(attn_out.indices):
                attn_schema_indices.append(attn_out.indices)

            x = attn_out.attended + x
            x = ff(x) + x

            total_aux_loss = total_aux_loss + attn_out.aux_loss
            total_recon_loss = total_recon_loss + attn_out.aux_loss_breakdown.recon_loss
            total_commit_loss = total_commit_loss + attn_out.aux_loss_breakdown.commit_loss

        attn_schema_autoregressive_loss = self.zero
        awareness_attn_sims = None
        meta_awareness_attn_sims = None
        awareness_attns = None
        meta_awareness_attns = None
        awareness_driven_attns = None

        # handle schema indices

        attn_schema_indices = rearrange(attn_schema_indices, 'depth b ... -> b (depth ...)') if not is_empty(attn_schema_indices) else None

        # awareness model

        if exists(self.awareness_model) and exists(attn_schema_indices):
            awareness_logits, awareness_embeddings, awareness_attn_sims, awareness_attns, awareness_aux_loss, awareness_attn_schema_indices = self.awareness_model(
                attn_schema_indices,
                attn_schema_targets = awareness_attn_schema_targets
            )

            # autoregressive loss on schema indices

            attn_schema_autoregressive_loss = F.cross_entropy(awareness_logits[:, :-1].flatten(0, 1), attn_schema_indices[:, 1:].flatten()) + awareness_aux_loss

            # handle schema indices for meta awareness

            awareness_attn_schema_indices = rearrange(awareness_attn_schema_indices, 'depth b ... -> b (depth ...)') if not is_empty(awareness_attn_schema_indices) else None

            if exists(self.meta_awareness_model) and exists(awareness_attn_schema_indices):
                meta_awareness_logits, meta_awareness_embeddings, meta_awareness_attn_sims, meta_awareness_attns, meta_awareness_aux_loss, _ = self.meta_awareness_model(
                    awareness_attn_schema_indices,
                    attn_schema_targets = meta_awareness_attn_schema_targets
                )

                meta_ce_loss = F.cross_entropy(meta_awareness_logits[:, :-1].flatten(0, 1), awareness_attn_schema_indices[:, 1:].flatten())
                attn_schema_autoregressive_loss = attn_schema_autoregressive_loss + meta_ce_loss + meta_awareness_aux_loss

            # awareness modulation - the evolutionary argument for attention schema theory
            # a simple attention block where the attention matrix is produced by the awareness
            # the awareness produces the next attention matrix via in-context learning of schema indices

            if use_awareness:
                x, awareness_driven_attns = self.awareness_driven_modulation(x, awareness_embeddings[:, -1])

        x = reduce(x, 'b n d -> b d', 'mean')

        logits = self.to_logits(x)

        return ASACReturn(
            logits,
            total_aux_loss,
            AuxLossBreakdown(total_recon_loss / self.depth, total_commit_loss / self.depth),
            attn_sims,
            attn_schema_indices,
            attn_schema_autoregressive_loss,
            awareness_attn_sims,
            meta_awareness_attn_sims,
            attns,
            awareness_attns,
            meta_awareness_attns,
            awareness_driven_attns
        )

# ema class

class EMA_ASAC(Module):
    DEFAULT_EMA_KWARGS = dict(
        beta = 0.99,
        update_after_step = 1000,
        update_every = 10
    )

    def __init__(
        self,
        asac,
        ema_kwargs: dict = None
    ):
        super().__init__()
        ema_kwargs = default(ema_kwargs, self.DEFAULT_EMA_KWARGS)

        self.asac = asac
        self.ema_model = EMA(asac, **ema_kwargs)

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

            ema_targets, ema_awareness_targets, ema_meta_awareness_targets = tree_map_tensor(lambda t: t.detach(), (ema_outputs.attn_sims, ema_outputs.awareness_attn_sims, ema_outputs.meta_awareness_attn_sims))

            self.ema_model.train()

        return self.asac(
            *args,
            attn_schema_targets = ema_targets,
            awareness_attn_schema_targets = ema_awareness_targets,
            meta_awareness_attn_schema_targets = ema_meta_awareness_targets,
            **kwargs
        )

import pytest
import torch

@pytest.mark.parametrize('use_asac', [False, True])
def test_asac(use_asac):
    from ASAC.ASAC import ASAC, Attention, AttentionSchema, PatchEmbedding

    from torch import nn
    from einops.layers.torch import Rearrange

    to_embedding = PatchEmbedding(dim = 512, patch_size = 32, channels = 3)

    asac = ASAC(
        dim = 512,
        depth = 2,
        heads = 8,
        seq_len = 64,
        to_embedding = to_embedding,
        use_asac = use_asac
    )

    data = torch.randn(1, 3, 256, 256)

    tokens = torch.randn(1, 4, 4, 512)
    logits, _, _ = asac(data)

    attn_schema = AttentionSchema(8 * 16 * 16, 64, codebook_size = 1024)

    attn = Attention(512, attn_schema = attn_schema)

    out = attn(tokens)
    out.aux_loss.backward()

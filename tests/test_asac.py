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
    logits = asac(data).logits

    attn_schema = AttentionSchema(8 * 16 * 16, 64, codebook_size = 1024)

    attn = Attention(512, attn_schema = attn_schema)

    out = attn(tokens)
    out.aux_loss.backward()

def test_ema_asac():
    from ASAC.ASAC import ASAC, PatchEmbedding, EMA_ASAC

    to_embedding = PatchEmbedding(dim = 512, patch_size = 32, channels = 3)

    asac = ASAC(
        dim = 512,
        depth = 2,
        heads = 8,
        seq_len = 64,
        to_embedding = to_embedding,
        use_asac = True
    )

    ema_asac = EMA_ASAC(asac)

    data = torch.randn(1, 3, 256, 256)

    ret = ema_asac(data)
    ret.aux_loss.backward()

    ema_asac.update()

    ema_ret = ema_asac(data, use_ema = True)
    assert ema_ret.logits.shape == ret.logits.shape

def test_autoregressive_schema_loss():
    from ASAC.ASAC import ASAC, PatchEmbedding
    
    to_embedding = PatchEmbedding(dim = 32, patch_size = 4, channels = 3)
    
    model = ASAC(
        dim = 32,
        depth = 64,
        heads = 2,
        dim_head = 16,
        seq_len = 16,
        dim_bottleneck = 16,
        vq_codebook_size = 32,
        to_embedding = to_embedding,
        use_asac = True,
        stochastic_sample_attn = True
    )

    data = torch.randn(2, 3, 16, 16)
    ret = model(data)
    
    assert ret.logits.shape == (2, 10)
    assert ret.attn_schema_indices.shape == (2, 64)
    assert ret.attn_schema_autoregressive_loss.ndim == 0
    
    import torch.nn.functional as F
    labels = torch.randint(0, 10, (2,))
    loss = F.cross_entropy(ret.logits, labels) + ret.attn_schema_autoregressive_loss
    loss.backward()

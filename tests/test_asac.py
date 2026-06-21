import pytest
import torch

@pytest.mark.parametrize('use_meta_awareness', [False, True])
@pytest.mark.parametrize('use_awareness', [False, True])
@pytest.mark.parametrize('use_asac', [False, True])
def test_asac(use_asac, use_awareness, use_meta_awareness):
    if use_meta_awareness and not use_awareness:
        pytest.skip('meta awareness requires awareness')

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
        use_asac = use_asac,
        use_awareness = use_awareness,
        use_meta_awareness = use_meta_awareness
    )

    data = torch.randn(1, 3, 256, 256)

    tokens = torch.randn(1, 4, 4, 512)
    logits = asac(data).logits

    attn_schema = AttentionSchema(8 * 16 * 16, 64, codebook_size = 1024)

    attn = Attention(512, attn_schema = attn_schema)

    out = attn(tokens)
    out.aux_loss.backward()

@pytest.mark.parametrize('use_meta_awareness', [False, True])
@pytest.mark.parametrize('use_awareness', [False, True])
def test_ema_asac(use_awareness, use_meta_awareness):
    if use_meta_awareness and not use_awareness:
        pytest.skip('meta awareness requires awareness')

    from ASAC.ASAC import ASAC, PatchEmbedding, EMA_ASAC

    to_embedding = PatchEmbedding(dim = 512, patch_size = 32, channels = 3)

    asac = ASAC(
        dim = 512,
        depth = 2,
        heads = 8,
        seq_len = 64,
        to_embedding = to_embedding,
        use_asac = True,
        use_awareness = use_awareness,
        use_meta_awareness = use_meta_awareness
    )

    ema_asac = EMA_ASAC(asac)

    data = torch.randn(1, 3, 256, 256)

    ret = ema_asac(data)
    ret.aux_loss.backward()

    ema_asac.update()

    ema_ret = ema_asac(data, use_ema = True)
    assert ema_ret.logits.shape == ret.logits.shape

@pytest.mark.parametrize('use_meta_awareness', [False, True])
@pytest.mark.parametrize('use_awareness', [True, False])
def test_autoregressive_schema_loss(use_awareness, use_meta_awareness):
    if use_meta_awareness and not use_awareness:
        pytest.skip('meta awareness requires awareness')

    from ASAC.ASAC import ASAC, PatchEmbedding, exists

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
        use_awareness = use_awareness,
        use_meta_awareness = use_meta_awareness,
        stochastic_sample_attn = True
    )

    data = torch.randn(2, 3, 16, 16)
    ret = model(data, use_awareness = use_awareness)

    assert ret.logits.shape == (2, 10)
    assert ret.attn_schema_indices.shape == (2, 64)
    assert ret.attn_schema_autoregressive_loss.ndim == 0

    assert exists(ret.attns)
    assert len(ret.attns) == 64

    if use_awareness:
        assert exists(ret.awareness_attns)
        assert len(ret.awareness_attns) == 2
        assert exists(ret.awareness_driven_attns)
        assert len(ret.awareness_driven_attns) == 1
    else:
        assert not exists(ret.awareness_attns)
        assert not exists(ret.awareness_driven_attns)

    if use_meta_awareness:
        assert exists(ret.meta_awareness_attns)
        assert len(ret.meta_awareness_attns) == 2
    else:
        assert not exists(ret.meta_awareness_attns)

    import torch.nn.functional as F
    labels = torch.randint(0, 10, (2,))
    loss = F.cross_entropy(ret.logits, labels) + ret.attn_schema_autoregressive_loss
    loss.backward()

@pytest.mark.parametrize('causal', [False, True])
def test_attention_schema(causal):
    from ASAC.ASAC import AttentionSchema, exists

    seq_len = 8
    heads = 4

    schema = AttentionSchema(
        dim = heads * (seq_len ** 2),
        dim_bottleneck = 64,
        codebook_size = 32,
        causal = causal
    )

    attn_sim = torch.randn(2, heads, seq_len, seq_len)
    recon, indices, loss, _ = schema(attn_sim)

    assert recon.shape == attn_sim.shape
    assert exists(indices)
    assert loss.ndim == 0

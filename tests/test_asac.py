import torch

def test_asac():
    from ASAC.ASAC import ASAC, Attention, AttentionSchema

    asac = ASAC()

    data = torch.randn(1, 3, 256, 256)

    tokens = torch.randn(1, 4, 4, 512)
    logits = asac(data)

    attn_schema = AttentionSchema(8 * 16 * 16, 64, codebook_size = 1024)

    attn = Attention(512, attn_schema = attn_schema)

    out, indices, loss = attn(tokens)
    loss.backward()

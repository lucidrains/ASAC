import torch

def test_asac():
    from ASAC.ASAC import ASAC

    asac = ASAC()

    data = torch.randn(1, 3, 256, 256)

    logits = asac(data)

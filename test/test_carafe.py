import torch
from torch.autograd import gradcheck

from tensorfn.nn import carafe


def test_carafe_gradcheck():
    feat = torch.randn(2, 64, 3, 3, requires_grad=True, device='cuda:0').double()
    mask = (
        torch.randn(2, 100, 6, 6, requires_grad=True, device='cuda:0')
        .sigmoid()
        .double()
    )

    assert gradcheck(
        lambda feat, mask: carafe(feat, mask, 5, 4, 2),
        (feat, mask),
        atol=1e-4,
        eps=1e-4,
    )


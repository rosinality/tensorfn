from torch import nn
from torch.nn import functional as F
from tensorfn import ensure_tuple


class DropBlock2d(nn.Module):
    def __init__(self, p, block_size, share_mask_across_batch=False):
        super().__init__()

        self.p = p
        self.block_size = ensure_tuple(block_size, 2)
        self.share_mask = share_mask_across_batch
        pad_h = self.block_size[0] - 1
        pad_w = self.block_size[1] - 1
        self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)

    def forward(self, input):
        if not self.training or self.p <= 0:
            return input

        batch, channel, height, width = input.shape

        prob = (
            self.p
            * (height * width)
            / (self.block_size[0] * self.block_size[1])
            / ((height - self.block_size[0] + 1) * (width - self.block_size[1] + 1))
        )

        mask_batch = 1 if self.share_mask else batch

        mask = input.new_empty(mask_batch, channel, height, width).bernoulli_(prob)

        if any(i != self.padding[0] for i in self.padding):
            mask = F.pad(mask, self.padding)
            mask = F.max_pool2d(mask, self.block_size, stride=1)

        else:
            mask = F.max_pool2d(
                mask,
                self.block_size,
                stride=1,
                padding=(self.padding[2], self.padding[0]),
            )

        mask = mask.mul_(-1).add_(1)
        weight = (
            mask.sum((2, 3), keepdim=True).add_(1e-8).reciprocal_().mul_(height * width)
        )

        mask = mask.mul_(weight)

        out = input * mask

        return out

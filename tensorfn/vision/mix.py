import numpy as np
import torch


def rand_bbox(size, lam):
    H = size[2]
    W = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bby1, bby2, bbx1, bbx2


def cutmix(input, target, alpha):
    lam = np.random.beta(alpha, alpha)
    rand_i = torch.randperm(input.shape[0], device=input.device)
    target_a = target
    target_b = target[rand_i]
    by1, by2, bx1, bx2 = rand_bbox(input.shape, lam)
    input[:, :, by1:by2, bx1:bx2] = input[rand_i, :, by1:by2, bx1:bx2]
    lam = 1 - ((bx2 - bx1) * (by2 - by1) / input.shape[-1] * input.shape[-2])

    return input, target_a, target_b, lam

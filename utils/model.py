import torch


def gradient_statistics(model):
    min_ = None
    max_ = None
    sum_ = 0.0
    sum_sq = 0.0
    abs_sum = 0.0
    n = 0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.flatten()
            if min_ is None:
                min_ = g.min()
                max_ = g.max()
            else:
                min_ = min(min_, g.min())
                max_ = max(max_, g.max())
            sum_ += g.sum()
            sum_sq += (g ** 2).sum()
            abs_sum += g.abs().sum()
            n += g.numel()
    return {
        "min": min_,
        "max": max_,
        "abs_mean": abs_sum / n,
    }
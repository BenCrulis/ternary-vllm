from timeit import timeit

import torch
from torch import nn

n_it = 100

mat = torch.randn(1024, 1024)
mat2 = torch.randn(1024, 1024)


t1 = timeit(lambda: torch.matmul(mat, mat2), number=n_it)
print(t1)

frac = 0.5
mat = torch.randn(1024, 1024)
mat2 = torch.randn(1024, 1024)
mat2[torch.rand(1024, 1024) < frac] = 0.0

t2 = timeit(lambda: torch.matmul(mat, mat2), number=n_it)
print(t2)

# mat2 = mat2.to_sparse()

# t3 = timeit(lambda: torch.matmul(mat, mat2), number=n_it)
# print(t3)


def ternary_mm(m, tm):
    res = torch.zeros(m.shape[0], tm.shape[1])
    mask_pos = tm > 0.0
    mask_neg = tm < 0.0
    for i in range(tm.shape[0]):
        sp = m[:, mask_pos[i]]
        sn = m[:, mask_neg[i]]
        res[:, i] = (sp.sum(dim=1) - sn.sum(dim=1)).squeeze()
    return res


t4 = timeit(lambda: ternary_mm(mat, mat2), number=n_it)
print(t4)
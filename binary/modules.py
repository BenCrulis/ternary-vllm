import torch
from torch import nn


def sign_ste(x):
    if not x.requires_grad:
        return ((x >= 0.0) * 2.0 - 1.0).to(dtype=t)
    x_c = x
    t = x.dtype
    return ((x >= 0.0) * 2.0 - 1.0).to(dtype=t) + x_c - x_c.detach()


def step_ste(x):
    if not x.requires_grad:
        return (x >= 0.0).to(dtype=t)
    x_c = x
    t = x.dtype
    return (x >= 0.0).to(dtype=t) + x_c - x_c.detach()


def tri_step_ste(x):
    if not x.requires_grad:
        return (x >= 0.5).to(dtype=t) - (x <= -0.5).to(dtype=t)
    x_c = x
    t = x.dtype
    return (x >= 0.5).to(dtype=t) - (x <= -0.5).to(dtype=t) + x_c - x_c.detach()


def sign_ste_sat(x):
    if not x.requires_grad:
        return ((x >= 0.0) * 2.0 - 1.0).to(dtype=t)
    x_c = x.clip(-1.0, 1.0)
    t = x.dtype
    return ((x >= 0.0) * 2.0 - 1.0).to(dtype=t) + x_c - x_c.detach()


def step_ste_sat(x):
    if not x.requires_grad:
        return (x >= 0.0).to(dtype=t)
    x_c = x.clip(-1.0, 1.0)
    t = x.dtype
    return (x >= 0.0).to(dtype=t) + x_c - x_c.detach()


class SignSTESat(nn.Module):
    def forward(self, x):
        if not x.requires_grad:
            return ((x >= 0.0) * 2.0 - 1.0).to(dtype=t)
        x_c = x.clip(-1.0, 1.0)
        t = x.dtype
        return ((x >= 0.0) * 2.0 - 1.0).to(dtype=t) + x_c - x_c.detach()
    


class ScaledBinaryLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, max_val=100.0):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        self.scale = nn.Parameter(torch.full((out_features,), 1.0))
        self.maxval = abs(max_val)
    
    def forward(self, x):
        w = sign_ste(self.weights).to(x.dtype)
        # x = x @ w.T
        x = nn.functional.linear(x, w)
        x *= self.scale
        if self.bias is not None:
            x += self.bias
        x = x.clip(-self.maxval, self.maxval)
        return x


class ScaledBinary01Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, max_val=100.0):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        self.scale = nn.Parameter(torch.full((out_features,), 1.0))
        self.maxval = abs(max_val)
    
    def forward(self, x):
        w = step_ste(self.weights).to(x.dtype)
        # x = x @ w.T
        x = nn.functional.linear(x, w)
        x *= self.scale
        if self.bias is not None:
            x += self.bias
        x = x.clip(-self.maxval, self.maxval)
        return x


class ScaledTernaryLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, max_val=100.0):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        self.scale = nn.Parameter(torch.full((out_features,), 1.0))
        self.maxval = abs(max_val)
    
    def forward(self, x):
        w = tri_step_ste(self.weights).to(x.dtype)
        # x = x @ w.T
        x = nn.functional.linear(x, w)
        x *= self.scale
        if self.bias is not None:
            x += self.bias
        x = x.clip(-self.maxval, self.maxval)
        return x


class SmoothlyBinarizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, max_val=100.0):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        self.maxval = abs(max_val)

    def count_converged_parameters(self, eps):
        return (1.0 - self.weights.abs() < eps).sum()
    
    def count_binary_parameters(self):
        return self.weights.numel()
    
    def compute_binarization_gradient(self):
        return -torch.sign(self.weights)
    
    def compute_and_set_biobjective_gradient(self):
        # deprecated
        g = self.weights.grad
        bg = self.compute_binarization_gradient()
        g_norm = torch.square(g).sum().sqrt()
        ng = g / g_norm if g_norm > 0.0 else g * 0.0
        bg_norm = torch.square(bg).sum().sqrt()
        nbg = bg / bg_norm if bg_norm > 0.0 else bg * 0.0
        final_g = (ng + nbg) * (g_norm + bg_norm) * 0.5
        self.weights.grad = final_g

    def forward(self, x):
        if self.training:
            w = self.weights.clip(-1.0, 1.0)
        else:
            w = (self.weights >= 0.0).to(x.dtype) * 2.0 - 1.0
        x = nn.functional.linear(x, w, bias=self.bias)
        x = x.clip(-self.maxval, self.maxval)
        return x


@torch.no_grad()
def compute_and_set_biobjective_gradient(model: nn.Module):
    norm_g_sq = 0.0
    norm_bg_sq = 0.0
    bgs = {}
    for mod in model.modules():
        if hasattr(mod, "compute_binarization_gradient"):
            bg = mod.compute_binarization_gradient()
            bgs[mod] = bg
            norm_bg_sq += torch.square(bg).sum()
        for p in mod.parameters(recurse=False):
            if p.grad is not None:
                norm_g_sq += torch.square(p.grad).sum()

    norm_g = norm_g_sq.sqrt()
    norm_bg = norm_bg_sq.sqrt()

    for mod in model.modules():
        if hasattr(mod, "compute_binarization_gradient"):
            bg = bgs[mod]
            g = mod.weights.grad
            ng = g / norm_g if norm_g > 0.0 else g * 0.0
            nbg = bg / norm_bg if norm_bg > 0.0 else bg * 0.0
            final_g = (ng + nbg) * (norm_g + norm_bg) * 0.5
            mod.weights.grad = final_g
        for p in mod.parameters(recurse=False):
            if p.grad is not None:
                p.grad = (p.grad / norm_g) * (norm_g + norm_bg) if norm_g > 0.0 else p.grad


def compute_smooth_binary_stats(model, eps=1e-3):
    n_smooth_binary_params = 0
    n_converged = 0
    for mod in model.modules():
        if hasattr(mod, "count_converged_parameters"):
            n_smooth_binary_params += mod.count_binary_parameters()
            n_converged += mod.count_converged_parameters(eps)
    return n_smooth_binary_params, n_converged


def linear_to_quantized(model: nn.Module, quantization="binary", scaling="none", neuron_scale="uniform", kmeans_iter=10):
    converted = False
    for name, mod in model.named_children():
        if isinstance(mod, nn.Linear):
            if quantization == "binary":
                sb = ScaledBinaryLinear(mod.in_features, mod.out_features, mod.bias is not None)
                sb.to(mod.weight.device)
                # std = mod.weight.data.std(axis=1)
                if neuron_scale == "uniform":
                    pos_mask = mod.weight > 0
                    neg_mask = mod.weight < 0
                    m_pos = (pos_mask * mod.weight).float().sum() # casting to .float() to avoid +inf
                    m_neg = (neg_mask * mod.weight).float().sum()

                    sb.scale.data[:] = (m_pos - m_neg) / (mod.weight.shape[1] * mod.weight.shape[0])
                elif neuron_scale == "independent":
                    # pos_mask = mod.weight > 0
                    # neg_mask = mod.weight < 0
                    # m_pos = (pos_mask * mod.weight).float().sum(dim=1) # casting to .float() to avoid +inf
                    # m_neg = (neg_mask * mod.weight).float().sum(dim=1)

                    # sb.scale.data[:] = (m_pos - m_neg) / mod.weight.shape[1]
                    sb.scale.data[:] = mod.weight.abs().mean(1)
                else:
                    raise ValueError(f"Unknown neuron scaling method: {neuron_scale}")

                if scaling is None or scaling == "none":
                    scale = 1.0
                elif isinstance(scaling, float):
                    scale = scaling
                elif scaling == "std":
                    full_std = mod.weight.data.std()
                    scale = 1.0/full_std
                elif scaling == "natural":
                    scale = 1.0 / sb.scale[:, None]
                else:
                    raise ValueError(f"Unknown scaling method: {scaling}")

                sb.weights.data[:] = (mod.weight * scale).clip(-1.0, 1.0).detach()

                if mod.bias is not None:
                    sb.bias.data = mod.bias.data
                model.add_module(name, sb)
                converted = True
            elif quantization == "binary01":
                sb = ScaledBinary01Linear(mod.in_features, mod.out_features, mod.bias is not None)
                sb.to(mod.weight.device)

                if neuron_scale == "uniform":
                    pos_mask = mod.weight > 0
                    m_pos = (pos_mask * mod.weight).float().sum() # casting to .float() to avoid +inf

                    sb.scale.data[:] = m_pos / (mod.weight.shape[1] * mod.weight.shape[0])
                elif neuron_scale == "independent":
                    pos_mask = mod.weight > 0
                    m_pos = (pos_mask * mod.weight).float().sum(dim=1) # casting to .float() to avoid +inf

                    sb.scale.data[:] = m_pos / mod.weight.shape[1]
                else:
                    raise ValueError(f"Unknown neuron scaling method: {neuron_scale}")

                if scaling is None or scaling == "none":
                    scale = 1.0
                elif isinstance(scaling, float):
                    scale = scaling
                elif scaling == "std":
                    full_std = mod.weight.data.std()
                    scale = 1.0/full_std
                elif scaling == "natural":
                    scale = 1.0 / sb.scale[:, None]
                else:
                    raise ValueError(f"Unknown scaling method: {scaling}")

                sb.weights.data[:] = (mod.weight * scale).clip(-1.0, 1.0).detach()

                if mod.bias is not None:
                    sb.bias.data = mod.bias.data
                model.add_module(name, sb)
                converted = True
            elif quantization == "smoothBinary":
                sb = SmoothlyBinarizedLinear(mod.in_features, mod.out_features, mod.bias is not None)
                sb.to(mod.weight.device)
                sb.weights.data[:] = mod.weight.detach()
                if mod.bias is not None:
                    sb.bias.data = mod.bias.data
                model.add_module(name, sb)
                converted = True
            elif quantization == "ternary":
                # use a modified k-means algorithm to compute the initial latent weight values
                st = ScaledTernaryLinear(mod.in_features, mod.out_features, mod.bias is not None)
                st.to(mod.weight.device)

                abs_w = mod.weight.abs().detach()
                if neuron_scale == "uniform":
                    abs_w = abs_w.flatten().unsqueeze(0)

                m = abs_w.detach().mean(dim=1)
                for i in range(kmeans_iter):
                    d = abs_w - m[:, None]
                    mask = d > - m[:, None] / 2.0
                    m = (abs_w * mask).float().sum(dim=1) / mask.sum(dim=1)

                scale = 1.0 / m

                st.weights.data[:] = (mod.weight * scale[:, None]).clip(-1.0, 1.0).detach()
                st.scale.data[:] = m.detach()

                if mod.bias is not None:
                    st.bias.data = mod.bias.data
                model.add_module(name, st)
                converted = True
    return converted

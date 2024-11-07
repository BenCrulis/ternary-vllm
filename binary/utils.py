from torch import nn

from binary.modules import (ScaledBinaryLinear, ScaledBinary01Linear, ScaledTernaryLinear, SmoothlyBinarizedLinear,
    linear_to_quantized)


def quantize_moondream(model,
                       start_skip=1,
                       last_skip=0,
                       quantization="binary",
                       scaling="none",
                       neuron_scale="uniform",
                       remove_blocks=None,
                       kmeans_iter=10):
    phimodel = model.text_model

    n_params_orig = sum(p.numel() for p in phimodel.parameters())

    n_blocks = len(model.text_model.transformer.h)

    layers_to_skip = [f"transformer.h.{i}." for i in range(start_skip)] +\
                    [f"transformer.h.{n_blocks-1-i}." for i in range(last_skip)]

    quantization_errors = []

    for name, mod in phimodel.named_modules():
        print(name)

        if "lm_head" not in name and not any([skipped in name for skipped in layers_to_skip]):
            converted, qerrs = linear_to_quantized(mod, quantization=quantization, scaling=scaling, neuron_scale=neuron_scale, kmeans_iter=kmeans_iter)
            quantization_errors.extend(qerrs)
            if converted:
                print(f"Converted {name} to {quantization}")
            # if isinstance(mod, nn.Linear):
            #     w = mod.weight
            #     w.data = w.clip(-1.0, 1.0)
            pass
    
    remove_blocks = remove_blocks or []

    phimodel.transformer.h = nn.ModuleList([mod for i, mod in enumerate(phimodel.transformer.h) if i not in remove_blocks])
    for i, mod in enumerate(phimodel.transformer.h):
        mod.mixer.layer_idx = i

    n_params = sum(p.numel() for p in phimodel.parameters())
    n_bin_params = 0
    n_binary_modules = 0
    n_ternary_modules = 0
    n_linear_modules = 0
    n_bits = 0
    for mod in phimodel.modules():
        if isinstance(mod, (ScaledBinaryLinear, ScaledBinary01Linear)):
            n_bin_params += mod.weights.numel()
            n_binary_modules += 1
            n_bits += mod.weights.numel() + 16 * 2 * mod.bias.numel()
        elif isinstance(mod, SmoothlyBinarizedLinear):
            n_bin_params += mod.weights.numel()
            n_binary_modules += 1
            n_bits += mod.weights.numel() + 16 * mod.bias.numel()
        elif isinstance(mod, ScaledTernaryLinear):
            n_ternary_modules += 1
            n_bits += mod.weights.numel()*2 + 16 * 2 * mod.bias.numel()
        elif isinstance(mod, nn.Linear):
            n_linear_modules += 1
            n_bits += mod.weight.numel() * 16 + mod.bias.numel() * 16
        elif isinstance(mod, (nn.Embedding, nn.LayerNorm)):
            n_bits += sum(p.numel() for p in mod.parameters()) * 16
        
    estimated_model_size_gb = n_params_orig * 2 / 1024 ** 3
    # estimated_binary_model_size_gb = (n_params - n_bin_params) * 2 / 1024 ** 3 + n_bin_params / 8 / 1024 ** 3
    estimated_quantized_model_size_gb = n_bits / 8 / 1024 ** 3

    avg_quantization_error = sum(quantization_errors) / len(quantization_errors)

    print(f"average quantization error: {avg_quantization_error:.6f}")
    print(f"number of parameters: {n_params}")
    print(f"number of binary parameters: {n_bin_params}")
    print(f"number of binary modules: {n_binary_modules}")
    print(f"number of ternary modules: {n_ternary_modules}")
    print(f"number of linear module left: {n_linear_modules}")

    print(f"estimated original size: {estimated_model_size_gb:.3f} GB")
    print(f"estimated quantized size: {estimated_quantized_model_size_gb:.3f} GB")
    return avg_quantization_error


def clip_binary_weights(model: nn.Module):
    for mod in model.modules():
        if isinstance(mod, (ScaledBinaryLinear, ScaledBinary01Linear, ScaledTernaryLinear, SmoothlyBinarizedLinear)):
            mod.weights.data = mod.weights.data.clip(-1.0, 1.0)

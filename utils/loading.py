import torch
from torch import nn
import safetensors

from transformers import AutoTokenizer, AutoModelForCausalLM


from binary.utils import quantize_moondream



def load_base_model(revision, device="cpu"):
    DTYPE=torch.float32
    DEVICE=device

    MD_REVISION = revision

    tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)
    moondream: nn.Module = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2", revision=MD_REVISION, trust_remote_code=True,
        attn_implementation=None,
        torch_dtype=DTYPE, device_map={"": DEVICE}
    )
    return moondream, tokenizer


def load_quantized_model(checkpoint_path, revision, start_skip=1, last_skip=1,
                        quantization="ternary", scaling="none", neuron_scale="none",
                        remove_blocks=None,
                        dont_load_weights=False,
                        kmeans_iter=10,
                        device="cpu"):
    moondream, tokenizer = load_base_model(revision, device=device)
    avg_qerr = quantize_moondream(moondream, start_skip, last_skip, quantization, scaling, neuron_scale, remove_blocks, kmeans_iter)
    
    if dont_load_weights:
        return moondream, tokenizer, avg_qerr
    
    with safetensors.safe_open(checkpoint_path, "pt") as md_weights:
        for name, p in moondream.named_parameters():
            v = md_weights.get_tensor(name)
            p.data[:] = v
    del md_weights

    return moondream, tokenizer, avg_qerr


def load_ternary_model(checkpoint_path, revision, dont_load_weights=False, kmeans_iter=10, device="cpu"):
    return load_quantized_model(checkpoint_path, revision,
                                start_skip=1, last_skip=1,
                                quantization="ternary",
                                scaling="none", neuron_scale="none", # not used for ternary models
                                remove_blocks=None,
                                dont_load_weights=dont_load_weights,
                                kmeans_iter=kmeans_iter,
                                device=device
                                )


def load_binary_model(checkpoint_path, revision, dont_load_weights=False, kmeans_iter=10, device="cpu"):
    return load_quantized_model(checkpoint_path, revision,
                                start_skip=1, last_skip=1,
                                quantization="binary",
                                scaling="std", neuron_scale="independent",
                                remove_blocks=None,
                                dont_load_weights=dont_load_weights,
                                kmeans_iter=kmeans_iter,
                                device=device
                                )
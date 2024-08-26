import torch
from torch import nn
import safetensors

from transformers import AutoTokenizer, AutoModelForCausalLM


from binary.utils import quantize_moondream



def load_base_model(revision):
    DTYPE=torch.float32
    DEVICE="cpu"

    MD_REVISION = revision

    tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)
    moondream: nn.Module = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2", revision=MD_REVISION, trust_remote_code=True,
        attn_implementation=None,
        torch_dtype=DTYPE, device_map={"": DEVICE}
    )
    return moondream, tokenizer


def load_quantized_model(checkpoint_path, revision, start_skip=1, last_skip=1,
                        quantization="ternary", scaling="none", neuron_scale="none", remove_blocks=None):
    moondream, tokenizer = load_base_model(revision)
    quantize_moondream(moondream, start_skip, last_skip, quantization, scaling, neuron_scale, remove_blocks)
    
    with safetensors.safe_open(checkpoint_path, "pt") as md_weights:
        for name, p in moondream.named_parameters():
            v = md_weights.get_tensor(name)
            p.data[:] = v
    del md_weights

    return moondream, tokenizer


def load_ternary_model(checkpoint_path, revision):
    return load_quantized_model(checkpoint_path, revision,
                                start_skip=1, last_skip=1,
                                quantization="ternary",
                                scaling="none", neuron_scale="none", # not used for ternary models
                                remove_blocks=None
                                )


def load_binary_model(checkpoint_path, revision):
    return load_quantized_model(checkpoint_path, revision,
                                start_skip=1, last_skip=1,
                                quantization="binary",
                                scaling="std", neuron_scale="independent",
                                remove_blocks=None
                                )
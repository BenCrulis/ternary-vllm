import re

import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization

from binary.modules import SignSTESat, linear_to_quantized, ScaledBinaryLinear

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

model_id = "vikhyatk/moondream2"
revision = "2024-05-08"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)


im_path = "/mnt/c/data/datasets/MessyTable/images/20190921-00001-01-01.jpg"


phimodel = model.text_model

n_params_orig = sum(p.numel() for p in phimodel.parameters())

for name, mod in phimodel.named_modules():
    print(name)

    if True or "lm_head" not in name: # and "transformer.h.0" not in name:
        converted = linear_to_quantized(mod)
        if converted:
            print(f"Converted {name} to ScaledBinaryLinear")
        # if isinstance(mod, nn.Linear):
        #     w = mod.weight
        #     w.data = w.clip(-1.0, 1.0)
        pass

n_params = sum(p.numel() for p in phimodel.parameters())
n_bin_params = 0
n_binary_modules = 0
n_linear_modules = 0
for mod in phimodel.modules():
    if isinstance(mod, ScaledBinaryLinear):
        n_bin_params += mod.weights.numel()
        n_binary_modules += 1
    elif isinstance(mod, nn.Linear):
        n_linear_modules += 1


estimated_model_size_gb = n_params_orig * 2 / 1024 ** 3
estimated_binary_model_size_gb = (n_params - n_bin_params) * 2 / 1024 ** 3 + n_bin_params / 8 / 1024 ** 3


print(f"number of parameters: {n_params}")
print(f"number of binary parameters: {n_bin_params}")
print(f"number of binary modules: {n_binary_modules}")
print(f"number of linear module left: {n_linear_modules}")

print(f"estimated original size: {estimated_model_size_gb:.3f} GB")
print(f"estimated binary size: {estimated_binary_model_size_gb:.3f} GB")

image = Image.open(im_path)
print("encoding image")
enc_image = model.encode_image(image)
print(enc_image.shape)
print("answering question")
print(model.answer_question(enc_image, "Describe this image.", tokenizer, do_sample=True))

pass
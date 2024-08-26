import argparse as ap
from pathlib import Path
import gc
import safetensors.torch
import numpy as np

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from tensorflow import keras

import larq
import larq_compute_engine as lce

import safetensors

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from binary.utils import quantize_moondream
from utils.conversion import torch_moondream_to_keras

from impl.tf.moondream.model import PhiModel

from training.datasets.llava import LLavaDS
from utils.scripting import get_var


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("--variant", type=str, default="matmul", help="one of {continuous, tf, matmul, unpack}")
    parser.add_argument("-c", "--checkpoint", type=Path, default=Path("checkpoints/moondream-q2-1-1-001"), help="path to checkpoint")
    parser.add_argument("--no-tf-quant", action="store_true", help="disable tflite quantization")
    parser.add_argument("--llavads", type=str, default=None)
    parser.add_argument("--coco", type=str, default=None)
    parser.add_argument("--model", type=str, default="vikhyatk/moondream2")
    parser.add_argument("--revision", type=str, default="2024-07-23")
    return parser.parse_args()

args = parse_args()

variant = args.variant
tf_quant = not args.no_tf_quant
if not tf_quant:
    print("disabling TFLite converter quantization")

print(f"starting conversion of the {variant} variant")

# MODEL = args.model
MODEL = args.checkpoint

MD_REVISION = args.revision

ternary = "continuous" not in variant.lower()

DTYPE=torch.float32
DEVICE="cpu"

ANSWER_EOS = "<|endoftext|>"

# Number of tokens used to represent each image.
IMG_TOKENS = 729

test_inference = False

tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)
moondream: nn.Module = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2", revision=MD_REVISION, trust_remote_code=True,
    attn_implementation=None,
    torch_dtype=DTYPE, device_map={"": DEVICE}
)

if ternary:
    print("setting ternary layers")
    quantize_moondream(moondream, start_skip=1, last_skip=1,
                        quantization="ternary", scaling="none", neuron_scale="none", remove_blocks=None)
    print(f"loading model from {MODEL}")
    with safetensors.safe_open(MODEL / "model.safetensors", "pt") as md_weights:
        for name, p in moondream.named_parameters():
            v = md_weights.get_tensor(name)
            p.data[:] = v
    del md_weights

LLavaDS_PATH = get_var("llavads", args)
COCO_PATH = get_var("coco", args)

train_file = "conversation_58k.json"
val_file = "detail_23k.json"

val_ds = LLavaDS(LLavaDS_PATH, COCO_PATH, file=val_file)

if test_inference:
    with torch.inference_mode():
        enc_image = moondream.encode_image(val_ds[0]["image"]) 
        print("finished encoding image")
        sample_predicted_text = moondream.answer_question(enc_image, "Describe this image.", tokenizer, do_sample=False)
        print("initial prediction:", sample_predicted_text)



# mixed_precision.set_global_policy('mixed_float16')
print("converting to keras")
tf_moondream, out_torch, out_tf = torch_moondream_to_keras(moondream.text_model, variant=variant)
print("deleting pytorch model")
del moondream
del out_tf
del out_torch
gc.collect()

inp_ids = tf.random.uniform((1, 6), maxval=50000, dtype=tf.int64)

empty_cache_fn = tf.function(tf_moondream.empty_cache)
compute_embeddings_fn = tf.function(tf_moondream.compute_embeddings, input_signature=[tf.TensorSpec(shape=(1, None), dtype=tf.int64)])
call_fn = tf.function(tf_moondream.call, input_signature=[tf.TensorSpec(shape=(1, None, 2048), dtype=tf.float32),
                                                          tf.TensorSpec(shape=(24, 2, 1, 32, None, 64), dtype=tf.float32)])

inp = tf.convert_to_tensor(compute_embeddings_fn(inp_ids.numpy()).numpy())
empty_cache = tf.convert_to_tensor(empty_cache_fn().numpy())
call_fn(inp, empty_cache)

# tf_moondream.compute_output_shape((None, None))
# inp = keras.Input((None,None))
# inp_cache = keras.Input(tf_moondream.empty_cache().shape)
# out = tf_moondream(inp, inp_cache)

# model = keras.Model(inputs=inp, outputs=out)

# tf_moondream(keras.Input((None, None)))


# out = model(inp)

# print(out)

# QuantDense(10, kernel_quantizer="ste_sign", kernel_constraint="weight_clip")(tf.keras.Input(shape=(10,))


converter = tf.lite.TFLiteConverter.from_concrete_functions([
    call_fn.get_concrete_function(inp, empty_cache),
    compute_embeddings_fn.get_concrete_function(inp_ids),
    empty_cache_fn.get_concrete_function(),
], tf_moondream)
# converter.experimental_new_converter = True
# converter.experimental_lower_to_saved_model
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS, # enable TensorFlow ops.
    # tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
]
# converter.target_spec.supported_types = {tf.dtypes.float32}
converter.allow_custom_ops = True
if tf_quant:
    print("activating TFLite default optimizations")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("converting model to TFLite")
tflite_model = converter.convert()
print("done.")

tf.random.set_seed(1234)
inp = tf.random.uniform((1, 10), maxval=50000, dtype=tf.int64)

if ternary:
    if tf_quant:
        filename = f"moondream-q2-{variant}.tflite"
    else:
        filename = f"moondream-q2-{variant}-no-tf-quant.tflite"
else:
    if tf_quant:
        filename = "moondream-tf.tflite"
    else:
        filename = f"moondream-tf-no-tf-quant.tflite"

with open(filename, "wb") as f:
    print(f"writing tflite model to disk ({filename})")
    f.write(tflite_model)
    f.flush()
print("model written to disk.")
pass
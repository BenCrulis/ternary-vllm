import argparse as ap
from pathlib import Path

import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import lite
from tensorflow.lite.python.interpreter import InterpreterWithCustomOps

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from larq_compute_engine.tflite.python.interpreter import Interpreter
from larq_compute_engine.tflite.python.interpreter_wrapper_lite import register_tflite_all_ops

from training.datasets.llava import LLavaDS
from utils.scripting import get_var


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("-f", "--file", type=Path, default="moondream-q2-matmul.tflite")
    parser.add_argument("--llavads", type=str, default=None)
    parser.add_argument("--coco", type=str, default=None)
    parser.add_argument("--model", type=str, default="vikhyatk/moondream2")
    parser.add_argument("--revision", type=str, default="2024-07-23")
    return parser.parse_args()

args = parse_args()

# MODEL = args.model
MODEL = Path("checkpoints/moondream-q2-1-1-001")

MD_REVISION = args.revision


DTYPE=torch.float32
DEVICE="cpu"

ANSWER_EOS = "<|endoftext|>"

# Number of tokens used to represent each image.
IMG_TOKENS = 729


print(f"loading model from {MODEL}")
tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)
moondream: nn.Module = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2", revision=MD_REVISION, trust_remote_code=True,
    attn_implementation=None,
    torch_dtype=DTYPE, device_map={"": DEVICE}
)

LLavaDS_PATH = get_var("llavads", args)
COCO_PATH = get_var("coco", args)

train_file = "conversation_58k.json"
val_file = "detail_23k.json"

val_ds = LLavaDS(LLavaDS_PATH, COCO_PATH, file=val_file)

# interpreter_object = Interpreter # LCE interpreter
# interpreter_object = lite.Interpreter # TF interpreter
interpreter_object = lambda model_path: InterpreterWithCustomOps([register_tflite_all_ops], model_path=model_path)

# tflite_model_path = "moondream-q2-tf.tflite"
# tflite_model_path = "moondream-q2-larq.tflite"
tflite_model_path = str(args.file)


if interpreter_object != Interpreter:
    interpreter = interpreter_object(tflite_model_path)
else:
    with open(tflite_model_path, mode="rb") as model_file:
        interpreter = interpreter_object(model_file.read())
# interpreter = lite.Interpreter("moondream-tf.tflite")
# predict_fn = interpreter.get_signature_runner()

with torch.inference_mode():
    # use original model to encode the image
    enc_image = moondream.encode_image(val_ds[0]["image"])
    print("finished encoding image")

    if False:
        # how we would do inference with the original model
        sample_predicted_text = moondream.answer_question(enc_image, "Describe this image.", tokenizer, do_sample=False)
        print("initial prediction:", sample_predicted_text)
    enc_image = enc_image.numpy()

del moondream.text_model

print(f"end of sequence is token {tokenizer.unk_token_id}")


chat_history = ""
question = "Describe this image."

prompt = f"<image>\n\n{chat_history}Question: {question}\n\nAnswer:"


def input_embeds(prompt, image_embeds, text_emb, tokenizer):
    def _tokenize(txt):
        return tokenizer(
            txt, return_tensors="np", add_special_tokens=False
        )["input_ids"].astype(np.int64)

    # Add BOS token
    embeds = []
    embeds.append(
        text_emb((np.array([[tokenizer.bos_token_id]], dtype=np.int64)))
    )

    if "<image>" not in prompt:
        embeds.append(text_emb(_tokenize(prompt)))
    else:
        assert prompt.count("<image>") == 1
        before, after = prompt.split("<image>")
        if len(before) > 0:
            embeds.append(text_emb(_tokenize(before)))
        embeds.append(image_embeds)
        if len(after) > 0:
            embeds.append(text_emb(_tokenize(after)))

    return np.concatenate(embeds, axis=1)

# call = interpreter.get_signature_runner("call")

init_cache_shape = interpreter.get_signature_runner("empty_cache").get_output_details()["output_0"]["shape"]

# cache = interpreter.get_signature_runner("empty_cache")()["output_0"] # throw an invalid tensor size error in newer versions of tensorflow
cache = np.zeros(init_cache_shape, dtype=np.float32)

# interpreter.resize_tensor_input(0, (1, 11, 2048))
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare your input data
token_ids = np.array(np.random.randint(0, 50000, size=(1, 11)), dtype=np.int64)

def embed(token_ids):
    compute_embeddings = interpreter.get_signature_runner("compute_embeddings")
    input_data = compute_embeddings(token_ids=token_ids)["output_0"]
    return input_data


def predict(token_ids, cache, embeddings=None):
    if embeddings is not None and token_ids is not None:
        raise ValueError("both token_ids and embeddings cannot be specified")
    elif embeddings is None and token_ids is None:
        raise ValueError("one of token_ids or embeddings has to be specified")

    if embeddings is None:
        compute_embeddings = interpreter.get_signature_runner("compute_embeddings")
        input_data = compute_embeddings(token_ids=token_ids)["output_0"]
        del compute_embeddings
    else:
        input_data = embeddings

    input_details = interpreter.get_input_details()

    inp_token_ids_idx = input_details[0]["index"]
    inp_cache_idx = input_details[1]["index"]

    # Set the input tensor
    interpreter.resize_tensor_input(inp_token_ids_idx, input_data.shape)
    interpreter.resize_tensor_input(inp_cache_idx, cache.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(0, input_data)
    interpreter.set_tensor(1, cache)

    time_bef = time.time()
    # Run inference
    interpreter.invoke()
    elapsed = time.time() - time_bef
    # print(f"done in {elapsed:.3f}s")

    output_details = interpreter.get_output_details()
    logits_idx = output_details[1]["index"]
    cache_idx = output_details[0]["index"]

    # Get the output tensor
    logits = interpreter.get_tensor(logits_idx)
    new_cache = interpreter.get_tensor(cache_idx)
    return logits, new_cache


def inference_loop(prompt, image_embed, cache, tokenizer):
    embs = input_embeds(prompt, image_embed, embed, tokenizer)
    print("prompt is encoded")
    print(prompt, end="", flush=True)
    pred_tokens = []
    SEQ_LEN = 6
    time_bef = time.time()
    for i in range(SEQ_LEN):
        if i == 1:
            elapsed = time.time() - time_bef
            print(f"\nimage decoding done in {elapsed:.2f}")
            print(prompt, end="", flush=True)
        if i <= 1:
            time_bef = time.time()
        logits, cache = predict(None, embeddings=embs, cache=cache)
        next_token = logits[0, -1, :].argmax()
        if next_token == tokenizer.unk_token_id:
            break
        print(tokenizer.decode(next_token), end="", flush=True)
        embs = embed(np.array([[next_token]], dtype=np.int64))
        pass
    print()
    elapsed = time.time() - time_bef
    t_per_s = (SEQ_LEN - 1) / elapsed
    print(f"cache shape {cache.shape}")
    print(f"done in {elapsed:.2f}s ({t_per_s:.2f}t/s)")
    pass

inference_loop(prompt, enc_image, cache, tokenizer)


logits, cache = predict(token_ids, cache=cache)

print(logits.shape)
print(cache.shape)

print("second call")
logits, cache = predict(token_ids, cache)

print(logits.shape)
print(cache.shape)


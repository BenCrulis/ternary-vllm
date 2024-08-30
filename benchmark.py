import argparse as ap
from pathlib import Path
import os
import gc
import time
import tracemalloc
import random
import csv

import resource
import psutil

from tqdm import tqdm

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import lite
from tensorflow.lite.python.interpreter import InterpreterWithCustomOps

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from larq_compute_engine.tflite.python.interpreter import Interpreter
from larq_compute_engine.tflite.python.interpreter_wrapper_lite import register_tflite_all_ops

from training.datasets.llava import LLavaDS, get_collate_fn
from utils.loading import load_base_model, load_ternary_model, load_binary_model
from utils.scripting import get_var


def get_mem_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    rss = memory_info.rss
    return rss

def get_peak():
    peak_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak_memory_kb


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("-f", "--file", type=Path, default="moondream-q2-matmul.tflite")
    parser.add_argument("-c", "--checkpoint", type=Path, default=None, help="pytorch checkpoint")
    parser.add_argument("--model", type=str, default="vikhyatk/moondream2")
    parser.add_argument("--revision", type=str, default="2024-07-23") # original revision was 2024-05-08
    parser.add_argument("-k", "--tokens", type=int, default=100)
    parser.add_argument("-n", "--samples", type=int, default=100)
    parser.add_argument("-t", "--threads", type=int, default=None, help="number of threads for the TFLite interpreter")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, default=Path("results_benchmarks/results.csv"))
    parser.add_argument("--dry", action="store_true", help="do not write anything to disk")
    return parser.parse_args()

args = parse_args()

print("TOKEN GENERATION BENCHMARKING SCRIPT")

print("args:", args)

dry = args.dry
out_path: Path = args.out

n_threads = args.threads
if n_threads is None:
    print("using automatic number of threads")
else:
    print(f"using {n_threads} thread(s)")
    print("setting limit in PyTorch")
    torch.set_num_threads(n_threads)

n_samples = args.samples

print(f"will evaluate on {n_samples} samples.")

checkpoint_path: Path = args.checkpoint

is_tflite_model = checkpoint_path is None
is_base_model = str(checkpoint_path) == "base"
is_binary = False if checkpoint_path is None else "moondream-11" in str(checkpoint_path) and "-q2" not in str(checkpoint_path)

if is_tflite_model:
    print(f"benchmarking TFLite model: {args.file}")
elif is_base_model:
    print("benchmarking base model")
else:
    print(f"benchmarking {'binary' if is_binary else 'ternary'} model: {checkpoint_path}")

VOCAB_SIZE = 51200

MD_REVISION = args.revision

DTYPE=torch.float32
DEVICE="cpu"

ANSWER_EOS = "<|endoftext|>"

# Number of tokens used to represent each image.
IMG_TOKENS = 729


gc.collect()
initial_mem = get_mem_usage()


if is_tflite_model:
    pass
elif not is_base_model:
    if is_binary:
        print("loading binary model")
        moondream, tokenizer = load_binary_model(checkpoint_path, MD_REVISION)
    else:
        print("loading ternary model")
        moondream, tokenizer = load_ternary_model(checkpoint_path, MD_REVISION)
    del tokenizer
else:
    print("loading base model")
    moondream, tokenizer = load_base_model(MD_REVISION)
    del tokenizer


gc.collect()


if is_tflite_model:
    # interpreter_object = Interpreter # LCE interpreter
    # interpreter_object = lite.Interpreter # TF interpreter
    interpreter_object = lambda model_path: InterpreterWithCustomOps([register_tflite_all_ops], model_path=model_path,
                                                                     num_threads=n_threads)


    # tflite_model_path = "moondream-q2-tf.tflite"
    # tflite_model_path = "moondream-q2-larq.tflite"
    tflite_model_path = str(args.file)

    print(f"loading {tflite_model_path}")

    # actually load the model
    if interpreter_object != Interpreter:
        interpreter = interpreter_object(tflite_model_path)
    else:
        with open(tflite_model_path, mode="rb") as model_file:
            interpreter = interpreter_object(model_file.read())


    psutil_mem_load = get_mem_usage()

    model_load_size, model_load_peak = tracemalloc.get_traced_memory()

    print("model load size", model_load_size)
    print("model load peak", model_load_peak)
    print("psutil load:", psutil_mem_load)

    gc.collect()
    psutil_mem_before_alloc = get_mem_usage()


    def tflite_embeds(token_ids):
        compute_embeddings = interpreter.get_signature_runner("compute_embeddings")
        input_data = compute_embeddings(token_ids=token_ids)["output_0"]
        return input_data


    init_cache_shape = interpreter.get_signature_runner("empty_cache").get_output_details()["output_0"]["shape"]
    # init_cache_shape = (0,)
    empty_cache = np.zeros(init_cache_shape, dtype=np.float32)


    def tflite_inference(tokens, cache):
        if cache is None:
            cache = empty_cache

        input_details = interpreter.get_input_details()

        inp_token_ids_idx = input_details[0]["index"]
        inp_cache_idx = input_details[1]["index"]

        token_embeds = tflite_embeds(tokens)

        # Set the input tensor
        interpreter.resize_tensor_input(inp_token_ids_idx, token_embeds.shape)
        interpreter.resize_tensor_input(inp_cache_idx, cache.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(0, token_embeds)
        interpreter.set_tensor(1, cache)

        tracemalloc.reset_peak()
        time_bef = time.time()
        # Run inference
        interpreter.invoke()
        elapsed = time.time() - time_bef

        cur_mem, peak_mem = tracemalloc.get_traced_memory()

        # print(f"done in {elapsed:.3f}s")

        output_details = interpreter.get_output_details()
        logits_idx = output_details[1]["index"]
        cache_idx = output_details[0]["index"]

        # Get the output tensor
        logits = interpreter.get_tensor(logits_idx)
        new_cache = interpreter.get_tensor(cache_idx)

        return logits, new_cache, elapsed, cur_mem, peak_mem
    
    model_embed_fn = tflite_embeds
    model_inf_fn = tflite_inference

else:
    def pytorch_embeds(token_ids):
        with torch.inference_mode():
            return moondream.text_model.transformer.embd(token_ids)

    def pytorch_inference(token_embeds):
        token_embeds = torch.tensor(token_embeds)

        with torch.inference_mode():
            tracemalloc.reset_peak()
            time_bef = time.time()
            # Run inference
            out = moondream.text_model(inputs_embeds=token_embeds)

            elapsed = time.time() - time_bef

            cur_mem, peak_mem = tracemalloc.get_traced_memory()
        
        logits = out.logits.detach().numpy()
        # print(f"done in {elapsed:.3f}s")

        return logits, elapsed, cur_mem, peak_mem
    
    model_embed_fn = pytorch_embeds
    model_inf_fn = pytorch_inference


if not dry:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csvfile = open(out_path, mode="w", newline='')

    fieldnames = ["index", "time start", "time end", "token/s", "elapsed", "initial mem", "mem", "peak mem", "memory before", "memory after"]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

    writer.writeheader()

else:
    print(f"WARNING: DRY RUN\nwould have written to {out_path}")


seed = args.seed
torch.random.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

gc.collect()
tracemalloc.start()

if is_tflite_model:
    interpreter.allocate_tensors()

pbar = tqdm(range(n_samples))

times = []
token_per_second_list = []

for i in pbar:

    x = np.zeros((1, 1), dtype=np.int64)
    
    psutil_mem_before_inference = get_mem_usage()

    tracemalloc.reset_peak()
    time_start = time.time()

    cache = None

    for j in range(args.tokens):
        logits, cache, elapsed, cur_mem, peak_mem = model_inf_fn(x, cache)
        x = logits.argmax(axis=-1)
        # print(x.item(), end=" ")
    # print()
    # print("cache shape:", cache.shape)

    time_end = time.time()

    # peak_mem = get_peak()

    elapsed = time_end - time_start
    t_per_s = args.tokens / elapsed

    times.append(elapsed)
    token_per_second_list.append(t_per_s)

    psutil_mem_after_inference = get_mem_usage()

    pbar.set_description(f"time: {elapsed:.0f}s, t/s: {t_per_s:.2f} peak memory: {peak_mem/(10**9):.2f}GB")
    
    if not dry:
        writer.writerow(
            {
                "index": i,
                "time start": time_start,
                "time end": time_end,
                "elapsed": elapsed,
                "token/s": t_per_s,
                "initial mem": initial_mem,
                "mem": cur_mem,
                "peak mem": peak_mem,
                "memory before": psutil_mem_before_inference,
                "memory after": psutil_mem_after_inference,
            }
        )
        csvfile.flush()

print(f"total time: {np.sum(times):.2f}s")
print("average time:", np.mean(times))
print(f"average tokens per second: {np.mean(token_per_second_list):.2f}")

if not dry:
    csvfile.close()

print("end of evaluation.")

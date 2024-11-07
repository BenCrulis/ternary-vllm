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
    parser.add_argument("-t", "--threads", type=int, default=None, help="number of threads for the TFLite interpreter")
    parser.add_argument("--model", type=str, default="vikhyatk/moondream2")
    parser.add_argument("--revision", type=str, default="2024-07-23")
    return parser.parse_args()

args = parse_args()

MD_REVISION = args.revision


ANSWER_EOS = "<|endoftext|>"

# Number of tokens used to represent each image.
IMG_TOKENS = 729


tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)


# txt = """<|endoftext|>Hello world!
# My name is John Doe.<|endoftext|>"""

txt = "\n\nQuestion: Describe this image.\n\nAnswer:"

# txt = """<|endoftext|>
# <|endoftext|>"""

tokens = tokenizer.encode(txt)

print(tokens)

pass



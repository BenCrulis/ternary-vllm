import argparse as ap
from pathlib import Path
import os
import random
import csv

from math import exp


from tqdm import tqdm

import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM

from training.datasets.llava import LLavaDS, get_collate_fn
from utils.loading import load_base_model, load_ternary_model, load_binary_model
from utils.scripting import get_var


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, default=None, help="pytorch checkpoint")
    parser.add_argument("--llavads", type=str, default=None)
    parser.add_argument("--subset", type=Path, default="detail_23k.json")
    parser.add_argument("--coco", type=str, default=None)
    parser.add_argument("--model", type=str, default="vikhyatk/moondream2")
    parser.add_argument("--revision", type=str, default="2024-07-23") # original revision was 2024-05-08
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("-n", "--samples", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("init_results.csv"))
    parser.add_argument("--dry", action="store_true", help="do not write anything to disk")
    return parser.parse_args()

args = parse_args()

print("args:", args)

dry = args.dry
out_path: Path = args.out

checkpoint_path: Path = args.checkpoint

is_binary = False if checkpoint_path is None else "moondream-11" in str(checkpoint_path) and "-q2" not in str(checkpoint_path)

BATCH_SIZE = 8

VOCAB_SIZE = 51200

MD_REVISION = args.revision

val_file = args.subset

DTYPE=torch.float32
DEVICE="cpu" if not torch.cuda.is_available() else "cuda"

print(f"using device {DEVICE}")

ANSWER_EOS = "<|endoftext|>"

# Number of tokens used to represent each image.
IMG_TOKENS = 729

val_size = args.samples

print(f"evaluating on {val_size} samples")

LLavaDS_PATH = get_var("llavads", args)
COCO_PATH = get_var("coco", args)

print("loading dataset")
val_ds = LLavaDS(LLavaDS_PATH, COCO_PATH, file=val_file)
val_ds = Subset(val_ds, range(0, len(val_ds), len(val_ds) // val_size))


if not dry:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csvfile = open(out_path, mode="w", newline='')

    fieldnames = ["variant", "k", "quantization error", "loss", "perplexity"]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

    writer.writeheader()

else:
    print(f"WARNING: DRY RUN\nwould have written to {out_path}")


for k in range(15): # number of k-means iterations
    print(f"running evaluation for number of k-means iterations k={k}")

    if is_binary:
        print("loading binary model")
        moondream, tokenizer, qerr = load_binary_model(checkpoint_path, MD_REVISION, dont_load_weights=True, kmeans_iter=k, device=DEVICE)
    else:
        print("loading ternary model")
        moondream, tokenizer, qerr = load_ternary_model(checkpoint_path, MD_REVISION, dont_load_weights=True, kmeans_iter=k, device=DEVICE)


    collate_fn = get_collate_fn(moondream.vision_encoder, tokenizer, IMG_TOKENS, ANSWER_EOS, DTYPE)


    def pytorch_embeds(token_ids):
        with torch.inference_mode():
            return moondream.text_model.transformer.embd(token_ids)

    def pytorch_inference(token_embeds):
        token_embeds = torch.tensor(token_embeds)

        with torch.inference_mode():
            # Run inference
            out = moondream.text_model(inputs_embeds=token_embeds)
        
        logits = out.logits.detach()
        return logits

    def compute_loss(batch):
        images, tokens, labels, attn_mask = batch

        tokens = tokens.to(DEVICE)
        labels = labels.to(DEVICE)
        attn_mask = attn_mask.to(DEVICE)

        with torch.no_grad():
            img_embs = moondream.vision_encoder(images)

        tok_embs = pytorch_embeds(tokens)
        inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)

        logits = pytorch_inference(inputs_embeds)

        vocab_size = logits.shape[-1]

        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.reshape(-1, vocab_size)
        shift_labels = shift_labels.reshape(-1)
        # Enable model parallelism
        shift_labels = shift_labels
        loss = loss_fct(shift_logits, shift_labels)

        return loss

    seed = args.seed
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=8,
    )

    pbar = tqdm(dl)

    l = 0

    with torch.inference_mode():
        for i, x in enumerate(pbar):        
            # l, num_tokens = compute_loss(model_embed_fn, model_inf_fn, batch)
            moondream.eval()
            l += compute_loss(x).item()
            pbar.set_description(f"k={k}, loss: {l/(i+1):.3f}")

        l /= len(dl)

        perplexity = exp(l)
        
        if not dry:
            writer.writerow(
                {
                    "k": k,
                    "quantization error": qerr,
                    "loss": l,
                    "perplexity": perplexity,
                    "variant": "binary" if is_binary else "ternary",
                }
            )
            csvfile.flush()

if not dry:
    csvfile.close()

print("end of evaluation.")

import os
import math
import argparse as ap

from utils.scripting import get_var

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import SGD
from datasets import load_dataset

from training.datasets.llava import LLavaDS, get_collate_fn
from binary.utils import quantize_moondream, clip_binary_weights
from binary.modules import compute_smooth_binary_stats, compute_and_set_biobjective_gradient


from tqdm import tqdm


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("--llavads", type=str, default=None)
    parser.add_argument("--coco", type=str, default=None)
    parser.add_argument("--model", type=str, default="vikhyatk/moondream2")
    parser.add_argument("--revision", type=str, default="2024-05-08")
    parser.add_argument("--deactivate-quantization", action="store_true")
    parser.add_argument("--quantization", type=str, default="binary")
    parser.add_argument("--start-skip", type=int, default=1)
    parser.add_argument("--last-skip", type=int, default=0)
    parser.add_argument("--neuron-scaling", type=str, default="uniform")
    parser.add_argument("--force-float32", action="store_true")
    parser.add_argument("--scaling", type=str, default="none")
    parser.add_argument("--remove", type=int, default=None, nargs="+")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--bfloat", action="store_true")
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--val-subset", type=int, default=200)
    parser.add_argument("--use-wandb", action="store_true")
    return parser.parse_args()

args = parse_args()

LLavaDS_PATH = get_var("llavads", args)
COCO_PATH = get_var("coco", args)

train_file = "conversation_58k.json"
val_file = "detail_23k.json"


val_size = args.val_subset
val_ds = LLavaDS(LLavaDS_PATH, COCO_PATH, file=val_file)
val_ds = Subset(val_ds, range(0, len(val_ds), len(val_ds) // val_size))

datasets = {
    "train": LLavaDS(LLavaDS_PATH, COCO_PATH, file=train_file),
    "val": val_ds,
}


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32 if DEVICE == "cpu" or args.force_float32 else torch.float16 # CPU doesn't support float16
if args.bfloat:
    DTYPE = torch.bfloat16
# DTYPE = torch.float16
MD_REVISION = "2024-05-08"

QUANTIZED = not args.deactivate_quantization
QUANTIZATION = args.quantization
START_SKIP = args.start_skip
LAST_SKIP = args.last_skip
REMOVE = args.remove


tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)
moondream = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2", revision=MD_REVISION, trust_remote_code=True,
    attn_implementation="flash_attention_2" if DEVICE == "cuda" else None,
    torch_dtype=DTYPE, device_map={"": DEVICE}
)


scaling = "none"
neuron_scaling = "none"
if QUANTIZED:
    print(f"using quantization: {QUANTIZATION}")
    scaling = args.scaling
    neuron_scaling = args.neuron_scaling
    quantize_moondream(moondream, start_skip=START_SKIP, last_skip=LAST_SKIP,
                       quantization=QUANTIZATION, scaling=scaling, neuron_scale=neuron_scaling, remove_blocks=REMOVE)
    # if QUANTIZATION == "smoothBinary":
    #     DTYPE = torch.float32
else:
    QUANTIZATION = "none"


EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

# Set this to 1 to disable gradient accumulation.
GRAD_ACCUM_STEPS = args.grad_accum_steps

# Learning rate for the Adam optimizer. Needs to be tuned on a case-by-case basis. As a general rule
# of thumb, increase it by 1.4 times each time you double the effective batch size.
#
# Source: https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
#
# Note that we linearly warm the learning rate up from 0.1 * LR to LR over the first 10% of the
# training run, and then decay it back to 0.1 * LR over the last 90% of the training run using a
# cosine schedule.
LR = args.lr
WD = args.wd
MOMENTUM = args.momentum

USE_WANDB = args.use_wandb

VAL_EVERY = args.val_every

arch_name = "moondream"
if QUANTIZATION == "binary01":
    arch_name = "moondream-01"
elif QUANTIZATION == "binary":
    arch_name = "moondream-11"
elif QUANTIZATION == "smoothBinary":
    arch_name = "moondream-s11"
elif QUANTIZATION == "ternary":
    arch_name = "moondream-q2"

remove_str = ("-r" + ",".join([str(x) for x in REMOVE])) if REMOVE is not None else ""

model_name = arch_name + f"-{START_SKIP}-{LAST_SKIP}" + remove_str
print(f"model name is: {model_name}")


ANSWER_EOS = "<|endoftext|>"

# Number of tokens used to represent each image.
IMG_TOKENS = 729

collate_fn = get_collate_fn(moondream.vision_encoder, tokenizer, IMG_TOKENS, ANSWER_EOS, DTYPE)


def compute_loss(batch):
    images, tokens, labels, attn_mask = batch

    images = images.to(DEVICE)
    tokens = tokens.to(DEVICE)
    labels = labels.to(DEVICE)
    attn_mask = attn_mask.to(DEVICE)

    with torch.no_grad():
        img_embs = moondream.vision_encoder.encoder(images)
        img_embs = moondream.vision_encoder.projection(img_embs)

    tok_embs = moondream.text_model.get_input_embeddings()(tokens)
    inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)

    outputs = moondream.text_model(
        inputs_embeds=inputs_embeds,
        labels=labels,
        attention_mask=attn_mask,
    )

    if len(outputs.logits.shape) != 3:
        raise ValueError(f"unexpected output shape: {outputs.logits.shape}")

    if outputs.logits.shape[1] <= 1:
        raise ValueError(f"sequence is too small, shape is: {outputs.logits.shape}")

    if not torch.all(torch.isfinite(outputs.logits)):
        print("output is not finite")
        print("loss")
        print(outputs.loss)
        print("outputs.shape", outputs.logits.shape)
        print("labels")
        print(labels)
        has_nan = torch.any(torch.isnan(outputs.logits))
        has_inf = torch.any(torch.isinf(outputs.logits))
        print("has nan:", has_nan)
        print("has inf:", has_inf)
        raise ValueError("loss is not finite")

    return outputs.loss

def lr_schedule(step, max_steps):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LR + 0.9 * LR * x / 0.1
    else:
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2

dataloaders = {
    "train": DataLoader(
        datasets["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    ),
    "val": DataLoader(
        datasets["val"],
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
    ),
}

n_frozen = 0
for p in moondream.text_model.parameters():
    if not p.requires_grad:
        n_frozen += 1
    p.requires_grad = True
print("number of previously frozen parameters:", n_frozen)

moondream.text_model.train()
moondream.text_model.transformer.gradient_checkpointing_enable()

total_steps = EPOCHS * len(dataloaders["train"]) // GRAD_ACCUM_STEPS
optimizer = SGD(
    [
        {"params": moondream.text_model.parameters()},
    ],
    lr=LR * 0.1,
    momentum=MOMENTUM,
    weight_decay=WD,
)

if USE_WANDB:
    import wandb
    wandb.init(
        project="moondream-ft",
        name=model_name,
        config={
            "EPOCHS": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
            "LR": LR,
            "WD": WD,
            "MOMENTUM": MOMENTUM,
            "QUANTIZED": QUANTIZED,
            "QUANTIZATION": QUANTIZATION,
            "SCALING": scaling,
            "NEURON SCALING": neuron_scaling,
            "START SKIP": START_SKIP,
            "LAST SKIP": LAST_SKIP,
        }
    )

moondream.to(dtype=DTYPE)
# moondream.text_model.half()
# moondream.vision_encoder.float()
moondream.text_model.eval()

if DEVICE != "cpu":
    with torch.inference_mode():
        enc_image = moondream.encode_image(val_ds[0]["image"])
        print("finished encoding image")
        sample_predicted_text = moondream.answer_question(enc_image, "Describe this image.", tokenizer, do_sample=True)
        print("initial prediction:", sample_predicted_text)
        # print(e)
        # print("got an error during sampling process, proceeding to training anyway")

i = 0
for epoch in range(EPOCHS):
    for batch in tqdm(dataloaders["train"], desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        i += 1

        moondream.text_model.train()

        loss = compute_loss(batch)
        if torch.isfinite(loss):
            loss.backward()
        else:
            print("loss is not finite")

        if i % GRAD_ACCUM_STEPS == 0:
            for p in moondream.text_model.parameters():
                if p.grad is not None:
                    p.grad.data = p.grad.clip(-1.0, 1.0)
                if torch.any(torch.isinf(p)):
                    print("inf in parameters")
                    print(p)
                    raise ValueError("inf in parameters")
                if torch.any(torch.isnan(p)):
                    print("nan in parameters")
                    print(p)
                    raise ValueError("nan in parameters")
            
            # for mod in moondream.modules():
            #     if hasattr(mod, "compute_and_set_biobjective_gradient"):
            #         mod.compute_and_set_biobjective_gradient()
            #         pass
            if QUANTIZATION == "smoothBinary":
                compute_and_set_biobjective_gradient(moondream)

            optimizer.step()
            optimizer.zero_grad()

            if QUANTIZED:
                clip_binary_weights(moondream)

            lr = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if VAL_EVERY > 1 and i % VAL_EVERY == 0 and USE_WANDB:
            # Calculate validation loss
            with torch.inference_mode():
                moondream.eval()
                val_loss = 0
                for val_batch in tqdm(dataloaders["val"], desc="Validation"):
                    with torch.no_grad():
                        val_loss += compute_loss(val_batch).item()
                val_loss /= len(dataloaders["val"])

                enc_image = moondream.encode_image(val_ds[0]["image"])
                sample_predicted_text = moondream.answer_question(enc_image, "Describe this image.", tokenizer, do_sample=True)
            pass

        if USE_WANDB:
            if QUANTIZATION == "smoothBinary":
                n_sb, n_converged = compute_smooth_binary_stats(moondream)
                fraction_converged = n_converged / n_sb

            wandb.log({
                "loss/train": loss.item(),
                "lr": optimizer.param_groups[0]['lr']
            } | ({"loss/val": val_loss, "sample text": sample_predicted_text} if i % VAL_EVERY == 0 else {})
            | ({"fraction_converged": fraction_converged} if QUANTIZATION == "smoothBinary" else {})
            )

model_path = f"checkpoints/{model_name}"
print(f"saving model to {model_path}")
moondream.save_pretrained(model_path)

if USE_WANDB:
    wandb.finish()

from pathlib import Path

import json

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize


class LLavaDS(Dataset):
    def __init__(self, root: Path, coco_root: Path, file="conversation_58k.json"):

        root = Path(root)
        coco_root = Path(coco_root)
        
        self.file = root / file

        images = {}
        if (coco_root / "images").exists():
            all_paths = coco_root.glob("images/*/*.jpg")
        else:
            all_paths = coco_root.glob("*/*.jpg")
        for im_path in all_paths:
            images[im_path.name] = im_path

        self.images = images

        with open(self.file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        im_name = sample["image"]
        im_path = self.images[im_name]
        im = Image.open(im_path)

        assert len(sample["conversations"]) % 2 == 0
        assert sample["conversations"][0]["from"] == "human"

        return {
            "image": im, # Should be a PIL image
            "qa": [
                {
                    "question": x["value"].replace("<image>", "").replace("\n", ""),
                    "answer": y["value"],
                }
                for x, y in zip(sample["conversations"], sample["conversations"][1:])
            ]
        }



def get_collate_fn(vision_encoder, tokenizer, IMG_TOKENS, ANSWER_EOS, DTYPE):
    def f(batch):
        images = [sample['image'] for sample in batch]
        images = [vision_encoder.preprocess(image) for image in images]

        labels_acc = []
        tokens_acc = []

        for sample in batch:
            toks = [tokenizer.bos_token_id]
            labs = [-100] * (IMG_TOKENS + 1)

            for qa in sample['qa']:
                q_t = tokenizer(
                    f"\n\nQuestion: {qa['question']}\n\nAnswer:",
                    add_special_tokens=False
                ).input_ids
                toks.extend(q_t)
                labs.extend([-100] * len(q_t))

                a_t = tokenizer(
                    f" {qa['answer']}{ANSWER_EOS}",
                    add_special_tokens=False
                ).input_ids
                toks.extend(a_t)
                labs.extend(a_t)

            tokens_acc.append(toks)
            labels_acc.append(labs)

        max_len = -1
        for labels in labels_acc:
            max_len = max(max_len, len(labels))

        attn_mask_acc = []

        for i in range(len(batch)):
            len_i = len(labels_acc[i])
            pad_i = max_len - len_i

            labels_acc[i].extend([-100] * pad_i)
            tokens_acc[i].extend([tokenizer.eos_token_id] * pad_i)
            attn_mask_acc.append([1] * len_i + [0] * pad_i)

        return (
            images,
            torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
            torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
            torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
        )
    return f


if __name__ == "__main__":

    ds = LLavaDS(Path("/mnt/c/data/datasets/LLaVA-Instruct-150K"), Path("/mnt/c/data/datasets/coco"))

    out = ds[0]

    print(out)
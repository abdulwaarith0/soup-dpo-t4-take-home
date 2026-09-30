"""The data the verdict asks for: the two changes, and nothing else.

1. Format: every row becomes conversational DPO (message lists) with the source's Zephyr markers
   (`<|user|>`, `<|assistant|>`, `</s>`) stripped, so TRL renders Qwen's own chat template and
   the model trains in the format it is served in.
2. Length: 250 pairs where the chosen answer is longer and 250 where it is shorter, so "longer
   wins" stops being a free signal (the original 500 were 69.6% chosen-longer).

Same source revision, same shuffle seed. The 100 held-out pairs (shuffled positions 500-599) are
excluded from selection and left untouched, so both runs are verified on identical pairs. Rows
are taken in the original shuffled order, so the original training rows are used first.

Usage: python data/prepare_fixed_data.py      -> data/train_fixed.jsonl
"""

import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

SOURCE = "0x7o/oasst-ru-dpo-v1"
REVISION = "56af31c2bb770371974508a47b3e8193a73dc3af"
SEED = 1337
HOLDOUT = range(500, 600)  # shuffled positions, as in prepare_data.py
PER_SIDE = 250

USER_RE = re.compile(r"^<\|user\|>\n(.*?)</s>\n?$", re.S)
ASSISTANT_RE = re.compile(r"^<\|assistant\|>\n(.*?)(?:</s>\n?)?$", re.S)
OUT = Path(__file__).resolve().parent / "train_fixed.jsonl"


def main() -> None:
    ds = load_dataset(SOURCE, split="train", revision=REVISION)
    order = list(range(len(ds)))
    random.Random(SEED).shuffle(order)
    held_out = {order[i] for i in HOLDOUT}

    longer, shorter, skipped = [], [], 0
    for idx in order:
        if idx in held_out:
            continue
        row = ds[idx]
        u = USER_RE.match(row["prompt"])
        c = ASSISTANT_RE.match(row["chosen"])
        r = ASSISTANT_RE.match(row["rejected"])
        if not (u and c and r):
            skipped += 1
            continue
        chosen, rejected = c.group(1).strip(), r.group(1).strip()
        if not chosen or not rejected or chosen == rejected or len(chosen) == len(rejected):
            skipped += 1
            continue
        pair = {
            "prompt": [{"role": "user", "content": u.group(1).strip()}],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
        }
        (longer if len(chosen) > len(rejected) else shorter).append(pair)

    rows = longer[:PER_SIDE] + shorter[:PER_SIDE]
    random.Random(SEED).shuffle(rows)
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]
    print(f"available outside the holdout: chosen longer {len(longer)}, chosen shorter {len(shorter)}, "
          f"skipped {skipped}")
    print(f"train_fixed.jsonl: {len(rows)} rows ({PER_SIDE} + {PER_SIDE})  sha256[:16]={digest}")


if __name__ == "__main__":
    main()

"""Build the 500-pair Russian preference set used for every run in this repo.

Source: 0x7o/oasst-ru-dpo-v1 on the Hugging Face Hub (1,322 rows, Russian
OpenAssistant conversations turned into prompt / chosen / rejected pairs).

The rows are sampled, NOT cleaned: the task asks what fails silently on messy
data, so the file keeps every quirk of the source (for example the chat-template
tokens baked into the text). What those quirks are, and what they do to a run,
is measured separately by audit_data.py.

Usage: python data/prepare_data.py            -> data/train.jsonl (+ data/holdout.jsonl)
"""

import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

SOURCE = "0x7o/oasst-ru-dpo-v1"
SEED = 1337
N_TRAIN = 500
N_HOLDOUT = 100  # never trained on; used by the Part 2 verification script
N_TASK_EVAL = 40  # first 40 holdout questions, for soup ship's task leg

OUT = Path(__file__).resolve().parent


def main() -> None:
    revision = HfApi().dataset_info(SOURCE).sha
    ds = load_dataset(SOURCE, split="train", revision=revision)
    rows = [
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in ds
    ]
    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)

    train = [rows[i] for i in order[:N_TRAIN]]
    holdout = [rows[i] for i in order[N_TRAIN:N_TRAIN + N_HOLDOUT]]

    for name, part in (("train.jsonl", train), ("holdout.jsonl", holdout)):
        path = OUT / name
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for row in part:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        print(f"{name}: {len(part)} rows  sha256[:16]={digest}")

    # soup ship leg 1: held-out user questions, template tokens stripped, scored by regex.
    # Pass = the answer contains Russian words and none of the source's template tokens.
    # It measures the one failure this dataset can plant (template text leaking into
    # answers); it is not a quality metric, and the report says so.
    no_junk_ru = r"^(?![\s\S]*(<\|(user|assistant|system)\|>|</s>))[\s\S]*[А-Яа-яЁё]{3,}"
    eval_path = OUT.parent / "eval" / "task_eval_ru.jsonl"
    eval_path.parent.mkdir(exist_ok=True)
    with eval_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in holdout[:N_TASK_EVAL]:
            question = row["prompt"].removeprefix("<|user|>\n").removesuffix("</s>\n").strip()
            fh.write(json.dumps({"prompt": question, "expected": no_junk_ru, "scoring": "regex",
                                 "category": "ru_no_template_leak"}, ensure_ascii=False) + "\n")
    print(f"task_eval_ru.jsonl: {N_TASK_EVAL} tasks")
    print(f"source: {SOURCE}@{revision}  rows: {len(rows)}  seed: {SEED}")


if __name__ == "__main__":
    main()

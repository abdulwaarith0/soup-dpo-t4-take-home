"""Part 3 (data side): measure the quirks of the preference file that a DPO run
would absorb without any error.

Each check prints a count, so the report can say what was found and how, and the
same file can be fed to Soup's own checks (soup data lint / validate / doctor) to
see which of these they catch.

Usage: python scripts/audit_data.py --data data/train.jsonl --max-length 1024
"""

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

TEMPLATE_TOKENS = ["<|user|>", "<|assistant|>", "<|system|>", "</s>", "<s>"]
CYR = re.compile(r"[А-Яа-яЁё]")
LAT = re.compile(r"[A-Za-z]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.jsonl", type=Path)
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-length", type=int, default=1024)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line]
    n = len(rows)
    found = collections.Counter()
    over, chosen_cut, rejected_cut, both_cut_same = 0, 0, 0, 0
    prompt_hashes = collections.Counter()
    longer = 0

    for r in rows:
        p, c, j = r["prompt"], r["chosen"], r["rejected"]
        for field, text in (("prompt", p), ("chosen", c), ("rejected", j)):
            for t in TEMPLATE_TOKENS:
                if t in text:
                    found[f"{field} contains {t!r}"] += 1
        if c.strip() == j.strip():
            found["chosen == rejected"] += 1
        if len(LAT.findall(c)) > len(CYR.findall(c)):
            found["chosen mostly Latin script (code / English)"] += 1
        longer += len(c) > len(j)
        prompt_hashes[hashlib.sha1(p.strip().lower().encode()).hexdigest()] += 1

        pl = len(tok(p, add_special_tokens=False)["input_ids"])
        cl = len(tok(c + tok.eos_token, add_special_tokens=False)["input_ids"])
        jl = len(tok(j + tok.eos_token, add_special_tokens=False)["input_ids"])
        if pl + max(cl, jl) > args.max_length:
            over += 1
        chosen_cut += pl + cl > args.max_length
        rejected_cut += pl + jl > args.max_length
        # both answers cut: the model never sees either ending, so the pair's signal is partial
        both_cut_same += (pl + cl > args.max_length) and (pl + jl > args.max_length)

    dup_prompts = sum(k - 1 for k in prompt_hashes.values() if k > 1)
    print(f"rows: {n}   file: {args.data}")
    print(f"tokenizer: {args.base}   max_length: {args.max_length}\n")
    for k, v in sorted(found.items()):
        print(f"  {k:48s} {v:4d}  ({100 * v / n:5.1f}%)")
    print(f"  {'chosen longer than rejected (characters)':48s} {longer:4d}  ({100 * longer / n:5.1f}%)")
    print(f"  {'duplicate prompts (beyond first)':48s} {dup_prompts:4d}")
    print(f"  {'pairs longer than max_length (truncated)':48s} {over:4d}  ({100 * over / n:5.1f}%)")
    print(f"  {'   chosen cut':48s} {chosen_cut:4d}")
    print(f"  {'   rejected cut':48s} {rejected_cut:4d}")
    print(f"  {'   both cut (neither ending seen)':48s} {both_cut_same:4d}")
    print("\nQwen2.5 special tokens for comparison: "
          f"{tok.convert_ids_to_tokens(tok('<|im_start|>')['input_ids'])} vs "
          f"'<|user|>' -> {tok.convert_ids_to_tokens(tok('<|user|>')['input_ids'])}")


if __name__ == "__main__":
    main()

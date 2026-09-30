"""Part 2: did the DPO run really change the model, in the direction the data asked for?

A falling training loss is measured inside the training process, on the pairs it
trained on. This script asks the questions that loss cannot answer, in a FRESH
process, from the saved adapter file, on pairs the run never saw:

  A  artifact   the adapter file holds the LoRA tensors the config implies, and
                lora_B (initialised to zero by PEFT) is no longer zero
  B  load       every tensor in the file reaches the model, and switching the
                adapter on changes the logits
  C  learning   on held-out pairs, the DPO implicit reward margin
                  beta * [(log pi(c) - log ref(c)) - (log pi(r) - log ref(r))]
                is positive more often than chance (ref = the same base, adapter off,
                which is exactly the reference Soup trains against under streaming)
  D  shortcuts  C again on the held-out pairs where the chosen answer is SHORTER
                (a length shortcut fails here), and C again with the prompt in
                Qwen's own chat template, the format soup ship and serving use

The same checks run on two controls built from the trained adapter itself:
  null    lora_B zeroed: a run that "completed" but changed nothing (for example an
          adapter that saved empty, or never loaded)
  random  lora_B replaced by Gaussian noise of the same norm: the weights DO change,
          so A and B pass, but nothing was learned, so C must fail

Usage (T4):  python scripts/verify_training.py --adapter runs/main --dtype float32
"""

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

USER_RE = re.compile(r"^<\|user\|>\n(.*?)</s>\n?$", re.S)
ASSISTANT_RE = re.compile(r"^<\|assistant\|>\n(.*?)(?:</s>\n?)?$", re.S)


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def load_pairs(path: Path, limit: int) -> list:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return rows[:limit]


def to_chat_format(row: dict, tok) -> dict:
    """The same pair, prompt rendered with the model's own chat template (how it is served)."""
    m = USER_RE.match(row["prompt"])
    c = ASSISTANT_RE.match(row["chosen"])
    r = ASSISTANT_RE.match(row["rejected"])
    if not (m and c and r):
        return None
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": m.group(1)}], tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt, "chosen": c.group(1), "rejected": r.group(1)}


@torch.no_grad()
def completion_logp(model, tok, prompt: str, completion: str, max_length: int, device) -> float:
    """Sum of log p(completion tokens | prompt), tokenised the way TRL's DPO does for text rows:
    prompt and completion tokenised separately, EOS appended to the completion, then
    concatenated and cut to max_length."""
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(completion + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + c_ids)[:max_length]
    n_prompt = min(len(p_ids), len(ids))
    if len(ids) - n_prompt == 0:
        return 0.0
    x = torch.tensor([ids], device=device)
    logits = model(input_ids=x).logits[0, :-1].float()
    logp = torch.log_softmax(logits, dim=-1)
    tgt = x[0, 1:]
    tok_lp = logp.gather(1, tgt[:, None])[:, 0]
    return tok_lp[n_prompt - 1:].sum().item()


def margins(model, tok, pairs: list, beta: float, max_length: int, device) -> list:
    out = []
    for row in pairs:
        pc = completion_logp(model, tok, row["prompt"], row["chosen"], max_length, device)
        pr = completion_logp(model, tok, row["prompt"], row["rejected"], max_length, device)
        with model.disable_adapter():
            rc = completion_logp(model, tok, row["prompt"], row["chosen"], max_length, device)
            rr = completion_logp(model, tok, row["prompt"], row["rejected"], max_length, device)
        out.append(beta * ((pc - rc) - (pr - rr)))
    return out


def summarise(ms: list) -> dict:
    n = len(ms)
    k = sum(m > 0 for m in ms)
    ties = sum(m == 0 for m in ms)
    return {
        "n": n,
        "accuracy": round(k / n, 3) if n else None,
        "accuracy_95ci": wilson(k, n),
        "exact_ties": ties,
        "mean_margin": round(sum(ms) / n, 5) if n else None,
    }


def check_artifact(adapter_dir: Path, n_layers: int) -> dict:
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    tensors = load_file(str(adapter_dir / "adapter_model.safetensors"))
    targets = cfg["target_modules"]
    targets = sorted(targets) if isinstance(targets, list) else [targets]
    expected = 2 * len(targets) * n_layers
    b_keys = [k for k in tensors if "lora_B" in k]
    nonzero_b = sum(tensors[k].abs().max().item() > 0 for k in b_keys)
    return {
        "tensors_in_file": len(tensors),
        "tensors_expected": expected,
        "inner_keys": sum(".inner." in k for k in tensors),
        "lora_B_nonzero": f"{nonzero_b}/{len(b_keys)}",
        "target_modules": targets,
        "r": cfg.get("r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "base_model_name_or_path": cfg.get("base_model_name_or_path"),
        "pass": len(tensors) == expected and nonzero_b == len(b_keys) and len(b_keys) > 0,
    }


def check_load(model, adapter_dir: Path, name: str) -> dict:
    """Every tensor in the file must equal a parameter of the loaded model."""
    tensors = load_file(str(adapter_dir / "adapter_model.safetensors"))
    params = dict(model.named_parameters())
    matched, mismatched, missing = 0, 0, 0
    for key, t in tensors.items():
        pk = key.replace(".weight", f".{name}.weight")
        p = params.get(pk)
        if p is None:
            missing += 1
        elif torch.equal(p.detach().cpu().to(t.dtype), t):
            matched += 1
        else:
            mismatched += 1
    return {"matched": matched, "mismatched": mismatched, "missing": missing,
            "pass": matched == len(tensors) and len(tensors) > 0}


def delta_norms(model, name: str) -> dict:
    """Relative size of the weight change the adapter applies: ||scale * B A|| / ||W||."""
    rels = []
    for mod_name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and name in getattr(mod, "lora_A", {}):
            a = mod.lora_A[name].weight.float()
            b = mod.lora_B[name].weight.float()
            scale = mod.scaling[name]
            w = mod.get_base_layer().weight.float()
            rels.append(((scale * (b @ a)).norm() / w.norm()).item())
    if not rels:
        return {"modules": 0}
    return {"modules": len(rels), "mean_rel_delta": f"{sum(rels) / len(rels):.2e}",
            "max_rel_delta": f"{max(rels):.2e}"}


@torch.no_grad()
def logit_shift(model, tok, text: str, device) -> float:
    x = tok(text, return_tensors="pt").input_ids.to(device)
    on = model(input_ids=x).logits.float()
    with model.disable_adapter():
        off = model(input_ids=x).logits.float()
    return (on - off).abs().max().item()


def make_controls(adapter_dir: Path, out_root: Path, seed: int) -> dict:
    tensors = load_file(str(adapter_dir / "adapter_model.safetensors"))
    g = torch.Generator().manual_seed(seed)
    arms = {}
    for arm in ("null", "random"):
        d = out_root / f"control_{arm}"
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(adapter_dir, d, ignore=shutil.ignore_patterns("checkpoint-*", "*.pt", "*.bin"))
        new = {}
        for k, t in tensors.items():
            if "lora_B" in k:
                if arm == "null":
                    new[k] = torch.zeros_like(t)
                else:
                    noise = torch.randn(t.shape, generator=g, dtype=torch.float32)
                    new[k] = (noise * (t.float().norm() / noise.norm())).to(t.dtype)
            else:
                new[k] = t.clone()
        save_file(new, str(d / "adapter_model.safetensors"))
        arms[arm] = d
    return arms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--holdout", default="data/holdout.jsonl", type=Path)
    ap.add_argument("--train", default="data/train.jsonl", type=Path)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=getattr(torch, args.dtype)).to(device)
    base.eval()
    n_layers = base.config.num_hidden_layers

    arms = {"trained": args.adapter}
    if not args.no_controls:
        arms.update(make_controls(args.adapter, args.adapter.parent / "verify_controls", args.seed))

    holdout = load_pairs(args.holdout, args.n)
    train = load_pairs(args.train, args.n)
    shorter = [r for r in holdout if len(r["chosen"]) < len(r["rejected"])]
    chat = [c for c in (to_chat_format(r, tok) for r in holdout) if c]
    probe = holdout[0]["prompt"] + holdout[0]["chosen"]

    model = None
    report = {"base": args.base, "dtype": args.dtype, "device": torch.cuda.get_device_name(0)
              if device == "cuda" else "cpu", "arms": {}}
    for name, path in arms.items():
        if model is None:
            model = PeftModel.from_pretrained(base, str(path), adapter_name=name)
        else:
            model.load_adapter(str(path), adapter_name=name)
        model.set_adapter(name)
        model.eval()
        res = {
            "A_artifact": check_artifact(path, n_layers),
            "B_load": check_load(model, path, name),
            "B_delta_norms": delta_norms(model, name),
            "B_max_logit_shift": round(logit_shift(model, tok, probe, device), 5),
            "C_holdout": summarise(margins(model, tok, holdout, args.beta, args.max_length, device)),
            "C_train_same_n": summarise(margins(model, tok, train, args.beta, args.max_length, device)),
            "D_holdout_chosen_shorter": summarise(
                margins(model, tok, shorter, args.beta, args.max_length, device)),
            "D_holdout_chat_template": summarise(
                margins(model, tok, chat, args.beta, args.max_length, device)),
        }
        c = res["C_holdout"]
        res["verdict"] = {
            "A_artifact": res["A_artifact"]["pass"],
            "B_load": res["B_load"]["pass"] and res["B_max_logit_shift"] > 0,
            "C_learned_on_unseen_pairs": c["accuracy_95ci"][0] > 0.5,
        }
        report["arms"][name] = res
        print(f"\n=== {name} ({path})")
        print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)

    t = report["arms"]["trained"]["verdict"]
    report["overall_pass"] = all(t.values())
    if not args.no_controls:
        report["controls_rejected"] = {
            arm: not all(report["arms"][arm]["verdict"].values()) for arm in ("null", "random")
        }
    out = args.out or args.adapter.parent / f"verify_{args.adapter.name}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nOVERALL trained-run verdict: {'PASS' if report['overall_pass'] else 'FAIL'}"
          f"  controls rejected: {report.get('controls_rejected')}  -> {out}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""Part 1: a VRAM budget for streamed DPO, computed BEFORE training.

Every term comes from the model's config.json and the training shape, plus two
facts read out of the code rather than assumed (both cited in the report):

  * TRL 0.29.1 DPOTrainer._compute_loss runs the policy forward, keeps
    outputs.logits AND a .contiguous() shifted copy alive for backward, then runs
    the reference forward (the frozen "ref" LoRA snapshot, no_grad) which makes
    the same two copies again, while the policy's are still held.
  * The logits reaching the loss are float32 (the local probe in
    logs/local-rtx3060/ printed dtype=torch.float32), so each copy is 4 bytes/elem.

Usage: python scripts/memory_budget.py --batch 2 --seq 1024
"""

import argparse
import json

from huggingface_hub import hf_hub_download

GB = 1e9
GIB = 1024**3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-targets", default="q_proj,v_proj")
    ap.add_argument("--stream-buffers", type=int, default=2)
    ap.add_argument("--weight-bytes", type=int, default=2, help="fp16 store on a T4")
    ap.add_argument("--logits-bytes", type=int, default=4, help="measured: float32")
    ap.add_argument("--cuda-context-gb", type=float, default=0.40,
                    help="driver context + cuBLAS workspace; visible only to nvidia-smi")
    args = ap.parse_args()

    cfg = json.load(open(hf_hub_download(args.base, "config.json")))
    h, i, v, n = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"], cfg["num_hidden_layers"]
    heads, kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = h // heads
    tied = cfg.get("tie_word_embeddings", False)
    rows = 2 * args.batch  # chosen and rejected go through the model as one batch
    s = args.seq
    wb, lb = args.weight_bytes, args.logits_bytes

    # one decoder layer's frozen weights (q,k,v with biases on Qwen2, o, gate/up/down, 2 norms)
    attn = h * h + h + 2 * (h * kv * hd + kv * hd) + h * h
    mlp = 3 * h * i
    layer_params = attn + mlp + 2 * h
    shapes = {
        "q_proj": (h, heads * hd), "k_proj": (h, kv * hd), "v_proj": (h, kv * hd),
        "o_proj": (heads * hd, h), "gate_proj": (h, i), "up_proj": (h, i), "down_proj": (i, h),
    }
    lora_params = n * sum(args.lora_r * (a + b) for t, (a, b) in shapes.items()
                          if t in args.lora_targets.split(","))

    logits_copy = rows * s * v * lb
    terms = [
        ("embedding / tied lm_head (resident, fp16)", v * h * wb * (1 if tied else 2)),
        (f"streamed layer buffers ({args.stream_buffers} x one layer)", args.stream_buffers * layer_params * wb),
        ("LoRA weights fp32 (cast for the fp16 GradScaler)", lora_params * 4),
        ("LoRA grads fp32", lora_params * 4),
        ("AdamW m + v fp32", lora_params * 8),
        ("TRL 'ref' adapter snapshot fp32", lora_params * 4),
        ("activation checkpoints (layer inputs, fp16)", n * rows * s * h * wb),
        ("one layer recompute working set (MLP dominates)", rows * s * (3 * i + 6 * h) * wb),
        ("policy logits + shifted copy (kept for backward)", 2 * logits_copy),
        ("reference logits + shifted copy (no_grad, transient)", 2 * logits_copy),
        ("logsumexp temp, one row (fp32 branch)", s * v * 4),
    ]
    total = sum(b for _, b in terms)

    print(f"model {args.base}: hidden {h}, layers {n}, vocab {v}, tied={tied}")
    print(f"shape: batch {args.batch} -> {rows} rows (chosen+rejected) x seq {s}; "
          f"LoRA r={args.lora_r} on {args.lora_targets} = {lora_params:,} params")
    print(f"one logits copy = {rows} x {s} x {v} x {lb} B = {logits_copy / GIB:.2f} GiB\n")
    for name, b in terms:
        print(f"  {name:52s} {b / GIB:7.3f} GiB  ({100 * b / total:4.1f}%)")
    print(f"  {'-' * 52} {'-' * 7}")
    print(f"  {'predicted torch max_memory_allocated':52s} {total / GIB:7.3f} GiB  = {total / GB:.2f} GB")
    ctx = args.cuda_context_gb * GB
    print(f"  {'+ CUDA context (nvidia-smi only, assumed)':52s} {ctx / GIB:7.3f} GiB")
    print(f"  {'predicted nvidia-smi floor (before allocator slack)':52s} {(total + ctx) / GIB:7.3f} GiB")
    print("\nWhat streaming saved: the resident alternative holds every layer, "
          f"{n * layer_params * wb / GIB:.2f} GiB of decoder weights, instead of "
          f"{args.stream_buffers * layer_params * wb / GIB:.2f} GiB of buffers.")


if __name__ == "__main__":
    main()

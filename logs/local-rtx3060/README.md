# Local development runs (RTX 3060 Laptop 6 GB, Windows 11)

Not the T4 evidence. These are the runs used to debug the pipeline before spending
Colab time, kept because two of them failed and the failures taught something.
Ampere card, so these ran in **bf16**, not the T4's fp16 path. Windows/WDDM, so memory
behaviour is not comparable to Linux. Same soup-cli 0.75.1, torch 2.11.0+cu128,
transformers / peft 0.21.1 / trl 0.29.1 from PyPI.

Runs 1-4 used `data/smoke.jsonl` (the first 16 rows of `data/train.jsonl`) and
`configs/smoke_local.yaml`, whose shape was lowered between runs; each log's
pre-flight panel prints the shape it ran with.

| Log | Shape | Outcome | What it showed |
|---|---|---|---|
| 1 | batch 2 x 1024 | refused by the pre-flight (9.99 GB predicted, 5.35 GB free) | the pre-flight works as a refusal gate |
| 2 | batch 1 x 512 | **CUDA OOM in the reference forward**, `selective_log_softmax`, after the pre-flight had predicted ~3.06 GB | allocated was already 3.06 GiB (3.29 GB) and the failed request was 298 MiB, i.e. one 2 x 511 x 151936 logits copy; the prediction was at least ~18% low |
| 3 | batch 1 x 256 | completed, 4 steps | predicted ~1.91 GB, measured `max_memory_allocated` 2.089 GiB = 2.24 GB (17% above the prediction) |
| 4 | batch 1 x 256 | completed, with `--probe` | the logits reaching the DPO loss are **float32**, and the reference pass adds exactly two logits-sized copies while the policy's two are still alive |
| 5 | batch 1 x 256, `data/smoke_fixed.jsonl` | completed | conversational rows (the fix-run format) train through streamed DPO before any T4 time was spent on them |
| 6 | `soup ship`, one suite | stopped by me | two bf16 models on a 6 GB card spilled into system RAM (Windows WDDM) and stalled the machine; ship was only run on the T4 |

`2_soup_crash_report.json` is the crash file Soup itself wrote for run 2 (GPU state at the failure).
`verify_smoke_bf16_n8.json` is `scripts/verify_training.py` run on run 3's adapter with 8 pairs in
bf16: a test that the script works, not a result (8 pairs cannot show learning either way).

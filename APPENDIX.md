# Appendix: the full numbers behind REPORT.md

Every figure here is read from a file in this repo. Log names are `logs/t4/<phase>.log` (UTC-timestamped,
exit code in `<phase>.exit`); GPU samples are `nvidia-smi/<phase>.csv` (every 500 ms) and
`nvidia-smi/<phase>_full.txt` (a full `nvidia-smi` screen every 60 s).

## A. Environment (T4, 2026-09-30)

| | |
|---|---|
| GPU | Tesla T4, 15,360 MiB, driver 580.82.07, CUDA 13.0 (`nvidia-smi/00_start.txt`) |
| Host | 2 vCPU Xeon @ 2.00 GHz, 12 GiB RAM, no swap (`00_host.txt`) |
| Colab Python | 3.13.15 (`01a_colab_python.txt`) |
| Run Python | 3.12.3 venv (uv); torch 2.14.0+cu130, transformers 5.17.0, peft 0.21.1, trl 0.29.1, accelerate 1.15.0, soup-cli 0.75.1 (`01_versions.txt`) |
| bf16 | `is_bf16_supported(including_emulation=False)` = False, bare call = True: the trap Soup's docs describe, reproduced |
| Base | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, copied to a plain directory (`10a_*`) |

## B. What broke, and what I did

| Phase | What happened | What I did |
|---|---|---|
| 01 install | `soup-cli[train]==0.75.1` not installable on Python 3.13 (0.73.0+ declare `<3.13`); unpinned resolves to 0.72.4 + transformers 4.57.6 + trl 0.24.0, exit 0 (`01a_unpinned_install_dry_run.log`) | 0.75.1 in a Python 3.12 venv. I did not run 0.72.4; Soup's v0.73.1 and v0.74.0 release notes say pre-Ampere cards failed on every task / could not stream before those releases |
| 10 train | `ValueError: cached snapshot file 'model.safetensors' points outside the Hugging Face blob store` before any GPU allocation. The snapshot links to `../../blobs/<sha>`, which is itself a link into a shared `hub/blobs/46/…` store; `spectrum_scan.py:690-696` only follows links that stay inside the repo's own `blobs/` | Same revision via `snapshot_download(local_dir=…)` (0 symlinks); `configs/colab_*.yaml` differ from the committed configs only in `base:` |
| local | Pre-flight admitted batch 1 × 512 on a 6 GB card; OOM in the reference forward (`logs/local-rtx3060/`) | Found the fp32 logits and the uncounted reference copies |

## C. Memory (main run, `07_memory_budget.log`, `10b_train_main.log`, `nvidia-smi/10b_train_main.csv`)

| Consumer (4 rows × 1024 × vocab 151,936) | Predicted GiB |
|---|---|
| Embedding / tied `lm_head`, fp16, resident | 0.435 |
| 2 streamed layer buffers | 0.174 |
| LoRA weights, grads, AdamW m+v, TRL `ref` snapshot (fp32) | 0.040 |
| Activation checkpoints (28 layer inputs, fp16) | 0.328 |
| One layer's recompute working set | 0.275 |
| Policy logits + shifted copy (fp32) | 4.637 |
| Reference logits + shifted copy (fp32) | 4.637 |
| Log-softmax temp, one row | 0.580 |
| **Total** | **11.106** |

Probe (`LOGITS_PROBE` lines): dtype `torch.float32`; allocated at the first reference call 4467 MiB vs 2628.5 MiB at
the policy call, i.e. the reference pass adds ~1.8 GiB at a 396-token batch, two 918 MiB copies.

| Phase | Soup pre-flight | `max_memory_allocated` | `max_memory_reserved` | nvidia-smi peak | mean GPU util |
|---|---|---|---|---|---|
| 10b main | ~9.99 GB | 11.406 GiB | 12.414 GiB | 12,883 MiB | 88% |
| 15 catalog-lr | ~9.99 GB | 11.406 GiB | 12.414 GiB | 12,883 MiB | 92% |
| 17 fix run | ~9.99 GB | (not probed) | | 13,917 MiB | 92% |
| 11 verify (fp32) | | | | 10,417 MiB | 96% |
| 12 ship (bf16) | | | | 6,499 MiB | 38% |

The fix run's sequences are longer (308,500 tokens against 289,200 for the same 500 pairs), most likely because
Qwen's template adds a default system prompt to every row; I did not measure its allocator peak.

**Throughput.** 289,200 tokens in 517.7 s = **559 tok/s** (main); the pre-flight forecast was 1,541-2,267 tok/s,
labelled "a compute-bound bound, not a promise". The forecast is a GEMM ceiling; it does not price the fp32
151,936-wide logits, the log-softmax, their copies, or DPO's third pass over the layers (policy, reference, recompute).

## D. Training runs (all 63 steps, exit 0)

| Run | Config | runtime | tokens | skipped (NaN) steps | mean loss | last loss |
|---|---|---|---|---|---|---|
| main | `dpo_stream_t4.yaml` | 517.7 s | 289,200 | step 11 | 0.6685 | 0.705 |
| catalog-lr (5e-6) | `control_recipe_lr.yaml` | 521.7 s | 289,200 | step 11 | 0.6908 | 0.6918 |
| fix run | `fixed_data.yaml` | 523.1 s | 308,500 | step 5 | 0.6902 | 0.7011 |

Step 1 of every run: loss 0.6931 = ln 2, margins 0 (policy == reference). Step 2 also has margins exactly 0 because
step 1 ran at learning rate 0 (warmup). TRL's saved `runs/main/ref/` adapter has all 56 `lora_B` at zero, so the
training reference was the base model.

## E. Verification (`runs/verify_*_v2.json`; first-rule outputs kept in `runs/verify_main.json`, `runs/verify_control_recipe_lr.json`)

| Adapter | ΔW rel. | logit shift | train acc | held-out acc [CI] | balanced, raw format | balanced, Qwen template [CI] | longer / shorter (Qwen) | first rule | verdict |
|---|---|---|---|---|---|---|---|---|---|
| main | 1.28e-3 | 2.223 | 0.81 | 0.58 [0.48, 0.67] | 0.486 | 0.49 [0.40, 0.58] | 0.723 / 0.257 | fail | FAIL |
| null | 0 | 0 | 0 | 0 (ties) | 0 | 0 | 0 / 0 | fail | rejected |
| random | 1.27e-3 | 0.152 | 0.49 | 0.43 [0.34, 0.53] | 0.449 | 0.55 [0.45, 0.65] | 0.446 / 0.657 | fail | rejected |
| catalog-lr | 1.32e-4 | 0.583 | 0.79 | 0.60 [0.502, 0.69] | 0.514 | 0.51 [0.42, 0.60] | 0.754 / 0.257 | **pass** | FAIL |
| fix run | 1.12e-3 | 0.585 | 0.83 | 0.54 [0.44, 0.63] | 0.495 | 0.48 [0.38, 0.57] | 0.692 / 0.257 | fail | FAIL |

Held-out set: 100 pairs, 65 with the longer answer chosen, 35 with the shorter. A pure "prefer longer" model scores
0.65 [0.55, 0.74] on plain accuracy (passes the first rule) and exactly 0.5 balanced (fails the current one).
The fix run's train accuracy is measured on its own training rows (conversational, rendered with Qwen's template).

## F. `soup ship` (`12_ship_main.log`, `runs/ship_main_verdict.json`)

Leg 1 (40 held-out Russian questions, pass = Russian and no leaked template text): base 0.95, tuned 0.95, no win.
Leg 2, threshold 5%: mini_arithmetic 0.9722/0.9722, mini_common_sense 0.9583/0.9583, mini_format_json 0.975/0.975,
mini_instruction 1.0/1.0, mini_mmlu 1.0/1.0, mini_over_refusal 0.925/0.925, mini_safety 1.0/1.0,
mini_tool_call 0.95/0.95. Verdict DON'T SHIP ("A strict win is required to ship"), exit 2, about 41 minutes on the
T4 in bf16. I did not run ship on the controls; the time went to the fix run instead.

## G. Data (`03_audit_data.log`, `04_soup_data_lint.log`)

All 500 rows carry `<|user|>`, `<|assistant|>` and `</s>`; Qwen tokenizes `<|user|>` as `<`, `|`, `user`, `|`, `>`.
Chosen longer than rejected in 348/500 (69.6%); `soup data lint` Cohen's d 0.459, verdict MINOR. 22 pairs exceed
1024 tokens (17 chosen, 8 rejected, 3 both cut). 22 chosen answers are mostly Latin script. No duplicate prompts.
The fixed set (`data/train_fixed.jsonl`): 250/250 length-balanced, conversational, lint length bias OK.

# Streamed DPO on a free Colab T4: ship or not?

Soup AI Engineer take-home. DPO with layer streaming (`training.stream_layers: true`) on a
Tesla T4, `soup-cli[train]` 0.75.1, Qwen2.5-1.5B-Instruct, 500 Russian preference pairs.

**Status: in progress.** The report and the T4 results land here when the runs are done.

| Path | What |
|---|---|
| `notebook/soup_dpo_t4.ipynb` | the Colab notebook that produced every T4 result, in order |
| `configs/dpo_stream_t4.yaml` | the run under review |
| `data/` | the 500-pair train set, the 100-pair holdout, and how both were made |
| `scripts/verify_training.py` | Part 2: the check that tells a real run from one that changed nothing |
| `scripts/memory_budget.py` | Part 1: the VRAM budget, computed before training |
| `scripts/audit_data.py` | Part 3: the data quirks a run absorbs silently |
| `scripts/train_with_mem.py` | runs `soup train` and records measured CUDA memory (+ a read-only logits probe) |
| `scripts/run_logged.sh`, `scripts/ts.py` | raw timestamped logs + raw nvidia-smi for every phase |
| `logs/t4/`, `nvidia-smi/` | raw T4 evidence |
| `logs/local-rtx3060/` | local debug runs, including the two that failed |

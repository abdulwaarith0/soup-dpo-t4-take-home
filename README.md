# Streamed DPO on a free Colab T4: ship or not?

Soup AI Engineer take-home. DPO with layer streaming (`training.stream_layers: true`) on a Tesla T4,
`soup-cli` 0.75.1, Qwen2.5-1.5B-Instruct, 500 Russian preference pairs.

**Verdict: DON'T SHIP.** The run is real and the memory held, but on unseen pairs the model is no better than a
"prefer the longer answer" heuristic, and it trained in a prompt format it will never be served in.
Read [`REPORT.md`](REPORT.md) (2 pages, also [`REPORT.pdf`](REPORT.pdf)); the full numbers are in
[`APPENDIX.md`](APPENDIX.md); how AI tools were used is in [`AI_USAGE.md`](AI_USAGE.md).

| Path | What |
|---|---|
| `notebook/soup_dpo_t4.ipynb` | the Colab cells in the order they ran on the T4, including the failed first run and the workaround; each cell's output is in the phase log it names (the tab holding the live outputs was closed before saving) |
| `configs/dpo_stream_t4.yaml` | the run under review; `control_recipe_lr.yaml` and `fixed_data.yaml` are the control and fix runs; `colab_*.yaml` are the exact configs run on Colab (only `base:` differs, see APPENDIX B) |
| `data/` | the 500-pair train set, the 100-pair holdout, the fixed set, and how each was made |
| `scripts/verify_training.py` | Part 2: tells a real run from one that changed nothing or learned a shortcut, with null and random controls |
| `scripts/memory_budget.py` | Part 1: the VRAM budget, computed before training |
| `scripts/audit_data.py` | Part 3: data quirks a run absorbs silently |
| `scripts/train_with_mem.py` | runs `soup train` and records measured CUDA memory (+ a read-only logits probe) |
| `scripts/run_logged.sh`, `scripts/ts.py` | raw timestamped logs + raw nvidia-smi for every phase |
| `logs/t4/`, `nvidia-smi/` | raw T4 evidence, one file pair per phase, failures included |
| `runs/` | verification and `soup ship` JSON, and the three trained adapters (+ TRL's reference snapshot) |
| `logs/local-rtx3060/` | local debug runs, including the two that failed |

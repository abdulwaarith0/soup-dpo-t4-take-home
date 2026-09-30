# Data

The task's own package (~500 pairs from Russian support tickets) was not required, so this uses
an open dataset that is as close as I could find: real Russian conversations, not a translation.

- **Source:** [`0x7o/oasst-ru-dpo-v1`](https://huggingface.co/datasets/0x7o/oasst-ru-dpo-v1) at revision
  `56af31c2bb770371974508a47b3e8193a73dc3af`: 1,322 prompt / chosen / rejected rows. The texts are
  Russian OpenAssistant conversations (human-written), but **the dataset card is empty**: it does not
  say how "chosen" and "rejected" were picked (OASST's human rankings, or something else), and it
  declares no licence (OpenAssistant itself is Apache-2.0). So the labels' quality is an assumption,
  not a known fact, and the report treats it as one.
- **Sampling:** shuffled with seed 1337; the first 500 rows are `train.jsonl`, the next 100 are
  `holdout.jsonl` (never trained on, used only by `scripts/verify_training.py`), and the first 40
  holdout questions become `../eval/task_eval_ru.jsonl` for `soup ship`.
- **Not cleaned, on purpose.** The task is about what fails silently on messy data, so every quirk
  of the source is kept and measured instead (`scripts/audit_data.py`). The biggest: every prompt,
  chosen and rejected string carries another model's chat-template text (`<|user|>`,
  `<|assistant|>`, `</s>`), which Qwen2.5's tokenizer reads as ordinary characters.
- **Not support tickets.** The topics are general (science, advice, code), so nothing here says how
  a model would do on the real ticket data; the report treats that as an open gap.

Rebuild with `python data/prepare_data.py`; it must print:

```
train.jsonl: 500 rows  sha256[:16]=9c2b4afb86c99846
holdout.jsonl: 100 rows  sha256[:16]=975ddce4a2943d8f
task_eval_ru.jsonl: 40 tasks
source: 0x7o/oasst-ru-dpo-v1@56af31c2bb770371974508a47b3e8193a73dc3af  rows: 1322  seed: 1337
```

`smoke.jsonl` is the first 16 rows of `train.jsonl`, used only for the local debug runs in
`../logs/local-rtx3060/`.

# How AI tools were used

The task allows AI tools and asks where they were used, what was accepted, and what was verified,
changed or redone. This is that record.

**Tool.** Claude Code (Anthropic's Claude, Opus 5.5) on my machine, with its browser extension driving
the Colab notebook in my own Chrome session. No other AI tool was used.

## What the AI did

- Read the soup-cli 0.75.1 source and docs to find the code paths that matter here: how streamed DPO
  builds its reference model, TRL 0.29.1's DPO loss, `soup ship`, `soup data lint / doctor / validate`,
  the pre-flight VRAM estimate and the snapshot check that stopped the first T4 run.
- Wrote the scripts (`data/prepare_data.py`, `scripts/*.py`, `scripts/run_logged.sh`) and the notebook.
- Ran the local debug runs on my RTX 3060 (`logs/local-rtx3060/`) and drove the T4 session in Colab.
- Drafted the report from the logs.
- Proposed the fix run (conversational format + length-balanced data); I chose to spend the remaining T4
  time on it rather than on `soup ship` for the random control.

## Where the AI was wrong, and how it was caught

Every one of these was caught by running something, not by re-reading.

| The AI's claim | What showed it was wrong | What changed |
|---|---|---|
| `pip install "soup-cli[train]"` gives 0.75.1 on Colab | the install failed on the T4: Colab's Python is 3.13 and 0.73.0+ requires <3.13 | pinned 0.75.1 in a Python 3.12 venv; the unpinned result (0.72.4) is recorded as a finding |
| that 0.72.4 path would also hit peft 0.21's empty-adapter bug | the pip dry run showed Colab ships peft 0.20.0 | claim dropped; the report cites Soup's release notes for what 0.72.x lacks on a T4 instead |
| the dataset's answers are "human-ranked" | the dataset card is empty | `data/README.md` says the labelling method is unknown |
| the first memory model assumed fp16 logits | a probe in the local run printed `torch.float32` | budget rebuilt on fp32 logits before the T4 run; the local runs that informed it are in `logs/local-rtx3060/` |
| the local pipeline check was enough to run on Colab | the first T4 run stopped on a symlink check that Windows never exercises | workaround documented in the notebook and the report |
| a local `soup ship` test was worth running on a 6 GB card | it spilled GPU memory into system RAM (Windows WDDM) and stalled the machine | stopped; ship was only run on the T4 |
| the verification's first pass rule (held-out accuracy CI above 0.5) was sound | the catalog-lr control passed it by 0.002 while being right on only 26% of chosen-shorter pairs; 65% of held-out pairs favour the longer answer, so a pure length heuristic passes | rule replaced by length-balanced accuracy; the old rule is still printed, labelled superseded (`scripts/verify_training.py` docstring) |
| a draft line called 0.72.4 "two weeks old" | checked the release date: 2026-08-03 | corrected before submission |
| (browser automation) typing into a Colab cell | twice the text went to the notebook in command mode, where letters are shortcuts; one inserted a stray cell | both stray cells deleted before anything ran; from then on the editor focus was checked before typing |

## What I verified, changed or redid myself

- The decisions were mine: the base model, the dataset, running on a real T4 rather than a local card, and
  spending the last T4 time on testing the fix instead of more `soup ship` runs.
- I went through every case in the table above myself and re-ran it, rather than taking the AI's account of it.
- I verified the pipeline end to end: the data rebuilds to the same hashes, the runs and their logs line up, and
  the numbers in the report match the files in `logs/t4/`, `nvidia-smi/` and `runs/`.
- I read `scripts/verify_training.py` and made sure I understood each check (artifact, load, the balanced verdict,
  the null and random controls) before trusting its output.
- I recomputed the length-balanced accuracies by hand from the verify JSON (for the main run,
  (0.723 + 0.257) / 2 = 0.49) and agreed with the rule change: a pure "longer wins" model would have passed the old rule.
- I re-derived the memory math (one fp32 logits copy = 4 × 1024 × 151,936 × 4 bytes = 2.32 GiB, four alive at the
  peak) and compared the budget with the measured peak in `logs/t4/10b_train_main.log`.
- I read `REPORT.md` and `APPENDIX.md` in full and checked the claims and wording against the logs before submitting.

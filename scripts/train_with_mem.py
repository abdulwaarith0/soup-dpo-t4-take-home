"""Run `soup train` in this process and report what the CUDA allocator really did.

`soup train` prints a PREDICTED peak before the model is built. This wrapper adds
the MEASURED side at exit, from the same process that trained:

  max_memory_allocated  bytes held by live tensors at the worst moment
  max_memory_reserved   bytes the caching allocator held from the driver

nvidia-smi (logged separately) sees reserved memory PLUS the CUDA context and
library workspaces, so the three numbers are expected to differ, and Part 1 of
the report explains by how much.

Usage: python scripts/train_with_mem.py --config configs/dpo_stream_t4.yaml
"""

import atexit
import json
import sys


def _report() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    gib = 1024**3
    stats = {
        "device": torch.cuda.get_device_name(0),
        "max_memory_allocated_gib": round(torch.cuda.max_memory_allocated() / gib, 3),
        "max_memory_reserved_gib": round(torch.cuda.max_memory_reserved() / gib, 3),
        "total_gib": round(torch.cuda.get_device_properties(0).total_memory / gib, 3),
    }
    print("MEASURED_CUDA_MEMORY " + json.dumps(stats), flush=True)


def _install_logits_probe(max_calls: int = 6) -> None:
    """Read-only instrumentation for Part 1: TRL's DPO loss calls selective_log_softmax
    once for the policy pass and once for the reference pass. Record the logits'
    dtype/shape and the allocated memory at each of the first few calls, then
    delegate unchanged. Nothing about the computation is altered."""
    import torch
    import trl.trainer.dpo_trainer as dpo_mod

    original = dpo_mod.selective_log_softmax
    calls = {"n": 0}
    mib = 1024**2

    def probed(logits, index):
        if calls["n"] < max_calls:
            calls["n"] += 1
            role = "policy" if logits.requires_grad else "reference"
            print(
                "LOGITS_PROBE " + json.dumps({
                    "call": calls["n"], "role": role, "dtype": str(logits.dtype),
                    "shape": list(logits.shape),
                    "tensor_mib": round(logits.numel() * logits.element_size() / mib, 1),
                    "allocated_mib": round(torch.cuda.memory_allocated() / mib, 1),
                    "max_allocated_so_far_mib": round(torch.cuda.max_memory_allocated() / mib, 1),
                }),
                flush=True,
            )
        return original(logits, index)

    dpo_mod.selective_log_softmax = probed


def main() -> None:
    atexit.register(_report)
    args = sys.argv[1:]
    if "--probe" in args:
        args.remove("--probe")
        _install_logits_probe()
    from soup_cli.cli import app

    sys.argv = ["soup", "train", *args]
    app()


if __name__ == "__main__":
    main()

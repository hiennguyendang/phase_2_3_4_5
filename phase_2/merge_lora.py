"""Merge the QLoRA adapter into the base model -> a standalone full checkpoint.

Lets step 7 load with just `--model <merged>` (no peft / no adapter). Merging
needs the base in bf16 (NOT 4-bit), so this loads on CPU by default to stay
within GPU memory; it is slow but safe. Needs ~30GB system RAM for a 7B model.

    pip install -q transformers peft accelerate
    python merge_lora.py --lora /kaggle/working/sg_lora --out /kaggle/working/sg_merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

import config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base weights")
    p.add_argument("--model", default=config.SG_LLM_MODEL, help="base model id/path")
    p.add_argument("--lora", type=Path, required=True, help="LoRA adapter dir (step 6 output)")
    p.add_argument("--out", type=Path, default=config.WORK_ROOT / "sg_merged")
    p.add_argument("--device", default="cpu", help="cpu (safe) or cuda (needs VRAM)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.lora.exists():
        raise SystemExit(f"[ERROR] LoRA dir not found: {args.lora}")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base {args.model} in bf16 on {args.device} ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map={"": args.device}, low_cpu_mem_usage=True,
    )
    print("Attaching + merging adapter ...")
    model = PeftModel.from_pretrained(base, str(args.lora))
    model = model.merge_and_unload()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out), safe_serialization=True)

    # tokenizer: prefer the one saved with the adapter, fall back to base
    try:
        tok = AutoTokenizer.from_pretrained(str(args.lora))
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.model)
    tok.save_pretrained(str(args.out))

    print(f"\nMerged model saved -> {args.out}")
    print(f"Use in step 7 with:  --model {args.out}   (drop --lora)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

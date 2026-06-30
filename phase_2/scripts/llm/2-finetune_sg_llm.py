"""Step 6 — QLoRA fine-tune the flat scene-graph extractor LLM (Qwen2.5-3B-Instruct).

4-bit nf4 + LoRA, completion-only loss (learn the assistant JSON only). The target is
the FLAT per-region findings JSON from build_sft_dataset.py (sg_schema), so a 3B is enough.

    pip install -q transformers trl peft bitsandbytes accelerate datasets
    python finetune_sg_llm.py --data-dir /kaggle/working/sg_sft \
        --out /kaggle/working/sg_lora --epochs 2

Kaggle (single 16GB T4): a 3B QLoRA fits with room to spare. The default --model comes
from config.SG_LLM_MODEL (3B); pass --model Qwen/Qwen2.5-7B-Instruct to compare. The
response_template below ("<|im_start|>assistant\\n") matches the Qwen2.5 chat template;
change it if you switch to a non-Qwen base. Adapters are saved to --out (step 7 loads them).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))  # phase_2/src

import argparse
import os
from pathlib import Path

import config


def _rclone(*a) -> None:
    import shutil
    import subprocess
    if not shutil.which("rclone"):
        return
    try:
        subprocess.run(["rclone", *a], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA finetune the SG LLM")
    p.add_argument("--data-dir", type=Path, default=config.WORK_ROOT / "sg_sft")
    p.add_argument("--out", type=Path, default=config.WORK_ROOT / "sg_lora")
    p.add_argument("--model", default=config.SG_LLM_MODEL)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-train-samples", type=int, default=0,
                   help="subsample the train split to this many rows (0 = all); SFT extraction "
                        "converges well on ~60-80k, cutting wall-clock on slow T4 4-bit")
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--resume", action="store_true",
                   help="continue from the last checkpoint in --out (pulled from Drive if --sync-remote)")
    p.add_argument("--sync-remote", default=None,
                   help="rclone remote for the run dir, e.g. dhint:sg_lora_runs/sg_lora")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, TrainerCallback)
    from trl import SFTConfig, SFTTrainer
    from trl import DataCollatorForCompletionOnlyLM

    train_path = args.data_dir / "train.jsonl"
    val_path = args.data_dir / "val.jsonl"
    if not train_path.exists():
        raise SystemExit(f"[ERROR] {train_path} not found (run build_sft_dataset.py).")

    # distributed (DDP via torchrun): each rank holds a full model copy on its OWN gpu.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = local_rank == 0

    # precision: bf16 ONLY on Ampere+ (sm_80). T4 is Turing (sm_75) with NO hardware bf16 -> bf16
    # there runs via slow emulation (≈minutes/step). Use fp16 on T4 -> tensor cores, ~10x faster.
    cap = torch.cuda.get_device_capability(local_rank)
    use_bf16 = cap[0] >= 8
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    if is_main:
        print(f"[precision] {torch.cuda.get_device_name(local_rank)} sm_{cap[0]}{cap[1]} -> "
              f"{'bf16' if use_bf16 else 'fp16'}")

    # Drive-resumable: pull prior checkpoints before training. Under DDP do it OUTSIDE this script
    # (the notebook pre-pulls once) to avoid all ranks racing on the same dir.
    remote = args.sync_remote.rstrip("/") if args.sync_remote else None
    args.out.mkdir(parents=True, exist_ok=True)
    if args.resume and remote and world_size == 1:
        print(f"[resume] pulling checkpoints from {remote}")
        _rclone("copy", remote, str(args.out), "--quiet")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
    )
    # single-gpu: "auto"; DDP: pin the whole (quantized) model to THIS rank's gpu
    device_map = {"": local_rank} if world_size > 1 else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map=device_map,
        torch_dtype=compute_dtype,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    data_files = {"train": str(train_path)}
    if val_path.exists():
        data_files["validation"] = str(val_path)
    ds = load_dataset("json", data_files=data_files)
    if args.max_train_samples and len(ds["train"]) > args.max_train_samples:
        ds["train"] = ds["train"].shuffle(seed=42).select(range(args.max_train_samples))
        if is_main:
            print(f"[subsample] train -> {len(ds['train']):,} rows")

    def fmt(batch):
        return tok.apply_chat_template(batch["messages"], tokenize=False)

    # completion-only loss: only tokens after the assistant header contribute.
    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n", tokenizer=tok,
    )

    sft_args = SFTConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16,          # T4 (Turing) has no hw bf16 -> fp16, else bf16 emulation is glacial
        group_by_length=True,       # batch similar-length samples -> far less padding -> faster
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},   # needed for DDP + grad ckpt
        ddp_find_unused_parameters=False,                          # LoRA: no unused params
        max_seq_length=args.max_len,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        peft_config=lora,
        formatting_func=fmt,
        data_collator=collator,
    )

    # push the run dir to Drive on every checkpoint (survives Kaggle session death); rank 0 only
    if remote:
        out_dir = str(args.out)

        class RcloneSync(TrainerCallback):
            def on_save(self, a, state, control, **kw):
                if state.is_world_process_zero:
                    _rclone("copy", out_dir, remote, "--transfers", "4", "--quiet")

        trainer.add_callback(RcloneSync())

    ckpt = None
    if args.resume and any(args.out.glob("checkpoint-*")):
        ckpt = True
        print(f"[resume] continuing from a checkpoint in {args.out}")
    trainer.train(resume_from_checkpoint=ckpt)

    if is_main:                          # save + push once (rank 0)
        trainer.save_model(str(args.out))
        tok.save_pretrained(str(args.out))
        if remote:
            _rclone("copy", str(args.out), remote, "--quiet")
            print(f"adapters pushed to {remote}")
        print(f"\nLoRA adapters saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

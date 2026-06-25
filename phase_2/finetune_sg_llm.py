"""Step 6 — QLoRA fine-tune the scene-graph LLM (Qwen2.5-7B-Instruct).

4-bit nf4 + LoRA, completion-only loss (learn the assistant JSON only).

    pip install -q transformers trl peft bitsandbytes accelerate datasets
    python finetune_sg_llm.py --data-dir /kaggle/working/sg_sft \
        --out /kaggle/working/sg_lora --epochs 2

Kaggle (single 16GB GPU): defaults below fit a 7B QLoRA. Drop --max-len or use a
smaller --model if you OOM. Adapters are saved to --out; merge or load on top of
the base model in step 7.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import config


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
    p.add_argument("--save-steps", type=int, default=200)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from trl import SFTConfig, SFTTrainer
    from trl import DataCollatorForCompletionOnlyLM

    train_path = args.data_dir / "train.jsonl"
    val_path = args.data_dir / "val.jsonl"
    if not train_path.exists():
        raise SystemExit(f"[ERROR] {train_path} not found (run build_sft_dataset.py).")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16,
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
        bf16=True,
        gradient_checkpointing=True,
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
    trainer.train()
    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))
    print(f"\nLoRA adapters saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

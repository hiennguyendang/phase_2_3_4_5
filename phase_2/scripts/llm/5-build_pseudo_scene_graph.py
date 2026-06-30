"""Step 7 — generate pseudo scene graphs for images WITH reports (CheXplus).

For each image: take the detector's 29 boxes (step 2 output) + the report, ask the
fine-tuned LLM for the FLAT findings (sg_schema), validate them against the closed
vocab (parse_flat drops hallucinations), map them to relation/cue strings
(compact_from_flat) and assemble a *_SceneGraph.json. NIH has no report -> no cues ->
intentionally excluded here (its boxes still come from the detector for C-KAN).

    python build_pseudo_scene_graph.py \
      --pred-dir /kaggle/working/pred \
      --metadata /kaggle/input/<chexplus>/chexplpus_metadata_final.jsonl \
      --lora /kaggle/working/sg_lora --out /kaggle/working/pseudo_sg \
      --update-metadata /kaggle/working/chexplus_with_scene.jsonl
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))  # phase_2/src

import argparse
import json
from pathlib import Path

import config
from scene_to_yolo import iter_jsonl
from sg_lib import assemble_scene_graph, available_regions
from sg_schema import SYSTEM_PROMPT, build_user_prompt, compact_from_flat, parse_flat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build pseudo scene graphs with the flat SG LLM")
    p.add_argument("--pred-dir", type=Path, required=True, help="per-image detector JSONs (step 2)")
    p.add_argument("--metadata", type=Path, required=True, help="dataset jsonl with reports")
    p.add_argument("--model", default=config.SG_LLM_MODEL)
    p.add_argument("--lora", type=Path, default=None, help="LoRA adapter dir (step 6)")
    p.add_argument("--out", type=Path, default=config.WORK_ROOT / "pseudo_sg")
    p.add_argument("--update-metadata", type=Path, default=None)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-4bit", action="store_true", help="load LLM in bf16 (needs more VRAM)")
    return p.parse_args()


def load_llm(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = dict(device_map="auto", torch_dtype=torch.bfloat16)
    if not args.no_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.lora is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.lora))
    model.eval()
    return model, tok


def main() -> int:
    args = parse_args()
    if not args.pred_dir.exists():
        raise SystemExit(f"[ERROR] pred-dir not found: {args.pred_dir}")

    # reports + ids from metadata
    meta: dict[str, dict] = {}
    for row in iter_jsonl(args.metadata):
        iid = str(row.get("image_id", "")).strip()
        if iid and str(row.get("report", "")).strip():
            meta[iid] = row
    print(f"images with report in metadata: {len(meta):,}")

    pred_files = sorted(args.pred_dir.glob("*.json"))
    jobs = []
    for pf in pred_files:
        if pf.name == "predictions.jsonl":
            continue
        try:
            pred = json.loads(pf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        iid = pred.get("image_id", pf.stem)
        row = meta.get(iid)
        if row is None:                      # no report -> skip (e.g. NIH)
            continue
        regions = available_regions(pred.get("objects", []))
        if not regions:
            continue
        jobs.append((iid, pred, row, regions))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"images to process (have boxes + report): {len(jobs):,}")
    if not jobs:
        raise SystemExit("[ERROR] nothing to process")

    import torch

    model, tok = load_llm(args)
    args.out.mkdir(parents=True, exist_ok=True)
    update_f = open(args.update_metadata, "w", encoding="utf-8") if args.update_metadata else None

    written = 0
    for start in range(0, len(jobs), args.batch):
        batch = jobs[start: start + args.batch]
        prompts = []
        for iid, pred, row, regions in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(str(row.get("report", "")), regions)},
            ]
            prompts.append(tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))

        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new_tokens, skip_special_tokens=True)

        for (iid, pred, row, regions), text in zip(batch, texts):
            flat = parse_flat(text)                  # validated against the closed vocab
            compact = compact_from_flat(flat)        # -> relation/cue strings, deterministic
            scene = assemble_scene_graph(
                iid, pred.get("objects", []), compact,
                viewpoint=row.get("viewpoint"),
                patient_id=row.get("patient_id"), study_id=row.get("study_id"),
                report=str(row.get("report", "")),
            )
            out_path = args.out / f"{iid}_SceneGraph.json"
            out_path.write_text(json.dumps(scene, ensure_ascii=False), encoding="utf-8")
            if update_f is not None:
                new_row = dict(row)
                new_row["scene_path"] = str(out_path)
                update_f.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            written += 1

        if written % 200 == 0 or start + args.batch >= len(jobs):
            print(f"  ...{written:,}/{len(jobs):,}")

    if update_f is not None:
        update_f.close()
    print(f"\nDONE. {written:,} pseudo scene graphs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

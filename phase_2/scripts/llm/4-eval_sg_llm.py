"""Step 6b — evaluate the flat scene-graph extractor (report -> findings).

Runs the fine-tuned LLM on the SFT val split and scores its output against the silver
target, answering the question from VERA_llm_parser_prior_report.md §8 ("measure per-finding
accuracy") and the "is 3B enough?" question (run once per --model, compare).

Metrics (all computed on parsed flat outputs):
  * format validity       : fraction of generations that yield a parseable JSON object.
  * presence (positive)    : micro/macro P/R/F1 over (region, finding) cells asserted "yes".
  * localization gap       : F1 ignoring region minus F1 with region (how much the model gets
                             the finding right but mislocalizes it).
  * progression (3-class)  : accuracy + confusion over gold cells that carry a comparison cue
                             and that the model also localized (+ coverage of those cells).
  * per-finding table      : P/R/F1 for the most frequent findings.

    pip install -q transformers peft bitsandbytes accelerate
    python eval_sg_llm.py --val /kaggle/working/sg_sft/val.jsonl \
        --lora /kaggle/working/sg_lora --out /kaggle/working/sg_eval.json --limit 2000
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))  # phase_2/src

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import config
from sg_schema import PROG_NAMES, SYSTEM_PROMPT_STRICT, parse_flat

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it


# ---------------------------------------------------------------------------
# helpers: turn a flat dict into comparable sets
# ---------------------------------------------------------------------------
def pos_cells(flat: dict) -> set[tuple[str, str]]:
    """(region, finding) pairs asserted present ("yes")."""
    return {(r, f["finding"]) for r, fs in flat.items() for f in fs
            if f.get("presence", "yes") == "yes"}


def unc_cells(flat: dict) -> set[tuple[str, str]]:
    """(region, finding) pairs flagged uncertain (hedged), regardless of polarity."""
    return {(r, f["finding"]) for r, fs in flat.items() for f in fs if f.get("uncertain")}


def prog_cells(flat: dict) -> dict[tuple[str, str], str]:
    """(region, finding) -> progression word, for present findings that carry one."""
    out = {}
    for r, fs in flat.items():
        for f in fs:
            if f.get("presence", "yes") == "yes" and f.get("progression") in PROG_NAMES:
                out[(r, f["finding"])] = f["progression"]
    return out


def _raw_parse_ok(text: str) -> bool:
    """Did the generation contain a parseable top-level JSON object (any content)?"""
    s = text.find("{")
    if s == -1:
        return False
    depth = 0
    for i in range(s, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    json.loads(text[s: i + 1])
                    return True
                except json.JSONDecodeError:
                    return False
    return False


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the flat SG extractor LLM")
    p.add_argument("--val", type=Path, default=config.WORK_ROOT / "sg_sft" / "val.jsonl")
    p.add_argument("--model", default=config.SG_LLM_MODEL)
    p.add_argument("--lora", type=Path, default=None, help="LoRA adapter dir (omit -> base model)")
    p.add_argument("--out", type=Path, default=config.WORK_ROOT / "sg_eval.json")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--prompt", choices=["sft", "strict"], default="sft",
                   help="sft = the system prompt baked in the data (for the finetuned model); "
                        "strict = the tighter zero-shot prompt with an inline example (for "
                        "un-finetuned base models)")
    p.add_argument("--limit", type=int, default=2000, help="eval only first N samples (0 = all)")
    p.add_argument("--top-findings", type=int, default=20)
    p.add_argument("--no-4bit", action="store_true")
    return p.parse_args()


def load_llm(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # bf16 only on Ampere+ (sm_80). T4 is Turing (sm_75): bf16 runs via slow emulation -> use fp16.
    dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    kwargs = dict(device_map="auto", torch_dtype=dt)
    if not args.no_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dt)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.lora is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.lora))
    model.eval()
    return model, tok


def main() -> int:
    args = parse_args()
    if not args.val.exists():
        raise SystemExit(f"[ERROR] val file not found: {args.val} (run build_sft_dataset.py)")

    rows = []
    for line in args.val.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        msgs = json.loads(line)["messages"]
        sys_m = next(m["content"] for m in msgs if m["role"] == "system")
        usr_m = next(m["content"] for m in msgs if m["role"] == "user")
        gold = next(m["content"] for m in msgs if m["role"] == "assistant")
        if args.prompt == "strict":            # zero-shot: tighter prompt + inline example
            sys_m = SYSTEM_PROMPT_STRICT
        rows.append((sys_m, usr_m, parse_flat(gold)))
        if args.limit and len(rows) >= args.limit:
            break
    print(f"eval samples: {len(rows):,}")

    import torch
    model, tok = load_llm(args)

    # accumulators
    n_format_ok = 0
    P_tp = P_fp = P_fn = 0                      # presence "yes", with region
    F_tp = F_fp = F_fn = 0                      # presence "yes", finding-only (no region)
    U_tp = U_fp = U_fn = 0                      # presence "uncertain", with region
    per_find = defaultdict(lambda: [0, 0, 0])   # finding -> [tp, fp, fn] (with region)
    prog_total = prog_correct = prog_covered = prog_gold = 0
    prog_conf = Counter()                       # (gold, pred) -> n

    starts = list(range(0, len(rows), args.batch))
    for start in tqdm(starts, desc="generate", unit="batch"):
        batch = rows[start: start + args.batch]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": s}, {"role": "user", "content": u}],
            tokenize=False, add_generation_prompt=True) for s, u, _ in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

        for (_s, _u, gold), text in zip(batch, texts):
            n_format_ok += int(_raw_parse_ok(text))
            pred = parse_flat(text)

            g_pos, p_pos = pos_cells(gold), pos_cells(pred)
            P_tp += len(g_pos & p_pos); P_fp += len(p_pos - g_pos); P_fn += len(g_pos - p_pos)
            for (_r, f) in (g_pos & p_pos):
                per_find[f][0] += 1
            for (_r, f) in (p_pos - g_pos):
                per_find[f][1] += 1
            for (_r, f) in (g_pos - p_pos):
                per_find[f][2] += 1

            g_find = {f for _r, f in g_pos}; p_find = {f for _r, f in p_pos}
            F_tp += len(g_find & p_find); F_fp += len(p_find - g_find); F_fn += len(g_find - p_find)

            g_unc, p_unc = unc_cells(gold), unc_cells(pred)
            U_tp += len(g_unc & p_unc); U_fp += len(p_unc - g_unc); U_fn += len(g_unc - p_unc)

            g_prog, p_prog = prog_cells(gold), prog_cells(pred)
            prog_gold += len(g_prog)
            for cell, gp in g_prog.items():
                if cell in p_prog:                      # model localized the cued cell
                    prog_covered += 1
                    prog_total += 1
                    pp = p_prog[cell]
                    prog_correct += int(pp == gp)
                    prog_conf[(gp, pp)] += 1

    # macro presence F1 over findings that actually appear in gold
    macro = [prf(tp, fp, fn)["f1"] for tp, fp, fn in per_find.values() if tp + fn > 0]
    report = {
        "model": args.model, "lora": str(args.lora) if args.lora else None,
        "n_samples": len(rows),
        "format_valid_rate": round(n_format_ok / max(1, len(rows)), 4),
        "presence_with_region": prf(P_tp, P_fp, P_fn),
        "presence_finding_only": prf(F_tp, F_fp, F_fn),
        "uncertain_with_region": prf(U_tp, U_fp, U_fn),
        "localization_gap_f1": round(prf(F_tp, F_fp, F_fn)["f1"] - prf(P_tp, P_fp, P_fn)["f1"], 4),
        "presence_macro_f1": round(sum(macro) / len(macro), 4) if macro else 0.0,
        "progression": {
            "gold_cued_cells": prog_gold,
            "coverage": round(prog_covered / max(1, prog_gold), 4),   # localized of cued
            "accuracy_on_covered": round(prog_correct / max(1, prog_total), 4),
            "confusion": {f"{g}->{p}": n for (g, p), n in sorted(prog_conf.items())},
        },
        "per_finding_top": [],
    }
    top = sorted(per_find.items(), key=lambda kv: -(kv[1][0] + kv[1][2]))[: args.top_findings]
    for finding, (tp, fp, fn) in top:
        report["per_finding_top"].append({"finding": finding, **prf(tp, fp, fn)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== EVAL ===")
    print(f"model            : {args.model}  (lora={report['lora']})")
    print(f"format valid     : {report['format_valid_rate']:.3f}")
    pr = report["presence_with_region"]
    print(f"presence (region): P {pr['precision']:.3f}  R {pr['recall']:.3f}  F1 {pr['f1']:.3f}")
    print(f"presence (macro) : {report['presence_macro_f1']:.3f}")
    uc = report["uncertain_with_region"]
    print(f"uncertain (hedge): P {uc['precision']:.3f}  R {uc['recall']:.3f}  F1 {uc['f1']:.3f}")
    print(f"localiz. gap F1  : {report['localization_gap_f1']:.3f}  "
          f"(finding-only F1 {report['presence_finding_only']['f1']:.3f})")
    pg = report["progression"]
    print(f"progression      : acc {pg['accuracy_on_covered']:.3f} on {prog_total} covered "
          f"(coverage {pg['coverage']:.3f} of {prog_gold} cued cells)")
    print(f"  confusion: {pg['confusion']}")
    print("top findings (finding: P/R/F1):")
    for d in report["per_finding_top"]:
        print(f"  {d['finding']:<40} {d['precision']:.2f}/{d['recall']:.2f}/{d['f1']:.2f}"
              f"  (tp{d['tp']} fp{d['fp']} fn{d['fn']})")
    print(f"\nreport -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

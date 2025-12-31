#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AI-Generated Code Header
# **Intent:** Set up a small, reproducible experiment to evaluate a biomedical LLM on medical QA datasets (subset + metrics).
# **Optimization:** Batches tokenization; supports limiting samples; optional quantization/device_map for feasibility.
# **Safety:** Clear dataset/model selection, capped sample sizes, prints failures with actionable hints.

"""
BioMistral Experiment: Medical QA Evaluation Harness
===================================================

This script implements a *practical* experiment scaffold inspired by the BioMistral paper:
  https://arxiv.org/abs/2402.10373

The paper evaluates across multiple established medical QA benchmarks. Full reproduction can be heavy.
This harness focuses on:
- selecting a dataset
- sampling a subset (e.g., 50-200 examples)
- generating answers
- computing a simple accuracy/EM-like score with normalization

Usage examples:
  python evaluate_medical_qa.py --dataset pubmed_qa --split validation --max-samples 50 --model <model_id>
  python evaluate_medical_qa.py --dataset medmcqa --split validation --max-samples 100 --model <model_id>

Auth:
  Set HF_TOKEN if your model is gated/private.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _die(msg: str, code: int = 1) -> "None":
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s


@dataclass(frozen=True)
class Example:
    prompt: str
    gold: str


def build_prompt(question: str, choices: Optional[List[str]] = None) -> str:
    # AI-SUGGESTION: Keep prompts minimal and consistent; some base models are not instruction-tuned.
    if choices:
        choices_block = "\n".join([f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)])
        return f"User: Answer the medical question by choosing the best option.\n\nQuestion: {question}\n\nOptions:\n{choices_block}\n\nAssistant:"
    return f"User: Answer the medical question.\n\nQuestion: {question}\n\nAssistant:"


def load_dataset_examples(dataset_key: str, split: str, max_samples: int) -> List[Example]:
    """
    Map common medical QA datasets into a unified prompt/gold format.

    Supported `dataset_key` values:
      - pubmed_qa
      - medmcqa

    Extend this mapping as needed (BioMistral paper evaluates many tasks).
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        _die(
            "Missing dependency: datasets.\n"
            "Install from this folder:\n"
            "  python -m pip install -r requirements.txt\n"
            f"Original error: {e}"
        )

    if dataset_key == "pubmed_qa":
        # Dataset: pubmed_qa (pqa_labeled)
        ds = load_dataset("pubmed_qa", "pqa_labeled", split=split)
        out: List[Example] = []
        for row in ds.select(range(min(max_samples, len(ds)))):
            question = row.get("question", "")
            # Some variants contain `context` with multiple fields; keep it short for feasibility.
            context = row.get("context", "")
            long_context = ""
            if isinstance(context, dict):
                # AI-SUGGESTION: avoid huge contexts; join snippets if present.
                snippets = context.get("contexts") or context.get("snippets") or []
                if isinstance(snippets, list) and snippets:
                    long_context = "\n".join([str(x) for x in snippets[:2]])
            elif isinstance(context, str):
                long_context = context
            if long_context:
                q = f"{question}\n\nContext:\n{long_context}"
            else:
                q = question
            gold = row.get("final_decision", "")  # typically: "yes"/"no"/"maybe"
            out.append(Example(prompt=build_prompt(q), gold=str(gold)))
        return out

    if dataset_key == "medmcqa":
        ds = load_dataset("medmcqa", split=split)
        out = []
        for row in ds.select(range(min(max_samples, len(ds)))):
            question = str(row.get("question", ""))
            choices = [str(row.get("opa", "")), str(row.get("opb", "")), str(row.get("opc", "")), str(row.get("opd", ""))]
            # `cop` is correct option index 0-3
            cop = row.get("cop", None)
            if cop is None:
                continue
            gold = ["A", "B", "C", "D"][int(cop)]
            out.append(Example(prompt=build_prompt(question, choices=choices), gold=gold))
        return out

    _die(f"Unknown dataset_key: {dataset_key}. Supported: pubmed_qa, medmcqa")


def generate_answers(
    model_id: str,
    prompts: List[str],
    hf_token: Optional[str],
    device_map: str,
    quant: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        _die(f"Missing dependency: transformers. Original error: {e}")

    torch_dtype = None
    try:
        import torch
        torch_dtype = torch.float16 if device_map != "cpu" else None
    except Exception:
        torch_dtype = None

    quantization_config = None
    if quant in ("8bit", "4bit"):
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            _die(f"Quantization requested but bitsandbytes/transformers support not available. Original error: {e}")
        if quant == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            import torch
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model_kwargs: Dict[str, Any] = {"device_map": device_map, "low_cpu_mem_usage": True, "token": hf_token}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    answers: List[str] = []
    try:
        import torch
        for p in prompts:
            inputs = tokenizer(p, return_tensors="pt")
            # Best-effort move if model on single device.
            try:
                model_device = getattr(model, "device", None)
                if model_device is not None and str(model_device) != "cpu":
                    inputs = {k: v.to(model_device) for k, v in inputs.items()}
            except Exception:
                pass

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0),
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            # Extract portion after Assistant:
            ans = text.split("Assistant:", 1)[-1].strip() if "Assistant:" in text else text
            answers.append(ans)
    except Exception as e:
        _die(f"Generation loop failed. Original error: {e}")

    return answers


def score_predictions(dataset_key: str, preds: List[str], golds: List[str]) -> Tuple[float, Dict[str, Any]]:
    correct = 0
    details: Dict[str, Any] = {}

    if dataset_key == "medmcqa":
        # Extract first letter A-D from model output.
        def extract_choice(s: str) -> str:
            s_norm = normalize_text(s)
            # Try common patterns like "A", "answer: A", etc.
            m = re.search(r"\b([abcd])\b", s_norm)
            return m.group(1).upper() if m else ""

        extracted = [extract_choice(p) for p in preds]
        for p, g in zip(extracted, golds):
            if p == g:
                correct += 1
        details["extracted"] = extracted
        return correct / max(1, len(golds)), details

    # pubmed_qa (yes/no/maybe)
    def extract_ynm(s: str) -> str:
        s_norm = normalize_text(s)
        if "yes" in s_norm:
            return "yes"
        if "no" in s_norm:
            return "no"
        if "maybe" in s_norm:
            return "maybe"
        return ""

    extracted = [extract_ynm(p) for p in preds]
    gold_norm = [normalize_text(g) for g in golds]
    for p, g in zip(extracted, gold_norm):
        if p == g:
            correct += 1
    details["extracted"] = extracted
    return correct / max(1, len(golds)), details


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BioMistral-style medical QA evaluation scaffold (subset-based).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="BioMistral/BioMistral-7B",
        help="HF model id to evaluate (default matches the BioMistral-7B model card).",
    )
    parser.add_argument("--dataset", required=True, choices=["pubmed_qa", "medmcqa"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--quant", default="none", choices=["none", "8bit", "4bit"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--out", default="", help="Optional path to save a JSONL of predictions.")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    examples = load_dataset_examples(args.dataset, args.split, args.max_samples)
    prompts = [e.prompt for e in examples]
    golds = [e.gold for e in examples]

    preds = generate_answers(
        model_id=args.model,
        prompts=prompts,
        hf_token=hf_token,
        device_map=args.device_map,
        quant=args.quant,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    acc, details = score_predictions(args.dataset, preds, golds)
    print(f"Dataset: {args.dataset} split={args.split} n={len(golds)}")
    print(f"Accuracy (simple extraction): {acc:.4f}")

    if args.out:
        import json

        with open(args.out, "w", encoding="utf-8") as f:
            for i, (p, g, pred) in enumerate(zip(prompts, golds, preds)):
                f.write(json.dumps({"i": i, "prompt": p, "gold": g, "pred": pred}, ensure_ascii=False) + "\n")
        print(f"Saved predictions to: {args.out}")

    # Print a few samples for quick inspection
    for i in range(min(3, len(golds))):
        print("\n--- Sample", i, "---")
        print("GOLD:", golds[i])
        print("PRED:", preds[i][:400])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



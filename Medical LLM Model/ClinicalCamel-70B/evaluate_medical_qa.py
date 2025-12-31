#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AI-Generated Code Header
# **Intent:** Evaluate Clinical Camel on small subsets of medical QA datasets (PubMedQA/MedMCQA) using a robust prompt strategy.
# **Optimization:** Subset-based evaluation; supports `device_map=auto` and optional quantization for feasibility.
# **Safety:** Avoids huge downloads by default; clear error messages for gated access and OOM.

"""
Clinical Camel Experiment: Medical QA (subset)
=============================================

Model:
  https://huggingface.co/wanglab/ClinicalCamel-70B

This is a lightweight evaluation scaffold (subset-based), inspired by typical medical QA evaluations.

Examples:
  python evaluate_medical_qa.py --dataset pubmed_qa --max-samples 50
  python evaluate_medical_qa.py --dataset medmcqa --max-samples 100
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
    question: str
    choices: Optional[List[str]]
    gold: str


def load_examples(dataset_key: str, split: str, max_samples: int) -> List[Example]:
    try:
        from datasets import load_dataset
    except Exception as e:
        _die(f"Missing datasets. Install: python -m pip install -r requirements.txt\n{e}")

    if dataset_key == "pubmed_qa":
        ds = load_dataset("pubmed_qa", "pqa_labeled", split=split)
        out: List[Example] = []
        for row in ds.select(range(min(max_samples, len(ds)))):
            q = str(row.get("question", ""))
            gold = str(row.get("final_decision", ""))  # yes/no/maybe
            out.append(Example(question=q, choices=None, gold=gold))
        return out

    if dataset_key == "medmcqa":
        ds = load_dataset("medmcqa", split=split)
        out: List[Example] = []
        for row in ds.select(range(min(max_samples, len(ds)))):
            q = str(row.get("question", ""))
            choices = [str(row.get("opa", "")), str(row.get("opb", "")), str(row.get("opc", "")), str(row.get("opd", ""))]
            cop = row.get("cop", None)
            if cop is None:
                continue
            gold = ["A", "B", "C", "D"][int(cop)]
            out.append(Example(question=q, choices=choices, gold=gold))
        return out

    _die(f"Unknown dataset: {dataset_key}")


def build_prompt(ex: Example) -> str:
    if ex.choices:
        opts = "\n".join([f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(ex.choices)])
        return (
            "User: Answer the medical question by choosing the best option.\n\n"
            f"Question: {ex.question}\n\nOptions:\n{opts}\n\n"
            "Return only the option letter (A/B/C/D).\n\nAssistant:"
        )
    return (
        "User: Answer the medical question.\n\n"
        f"Question: {ex.question}\n\n"
        "Answer with yes/no/maybe if applicable.\n\nAssistant:"
    )


def extract_answer(dataset_key: str, text: str) -> str:
    if dataset_key == "medmcqa":
        m = re.search(r"\b([ABCD])\b", text.strip())
        return m.group(1) if m else ""
    t = normalize_text(text)
    if "yes" in t:
        return "yes"
    if "no" in t:
        return "no"
    if "maybe" in t:
        return "maybe"
    return ""


def score(dataset_key: str, preds: List[str], golds: List[str]) -> float:
    correct = 0
    if dataset_key == "medmcqa":
        for p, g in zip(preds, golds):
            if p == g:
                correct += 1
        return correct / max(1, len(golds))
    golds_n = [normalize_text(g) for g in golds]
    for p, g in zip(preds, golds_n):
        if p == g:
            correct += 1
    return correct / max(1, len(golds))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Subset evaluation for Clinical Camel (Transformers).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="wanglab/ClinicalCamel-70B")
    parser.add_argument("--dataset", required=True, choices=["pubmed_qa", "medmcqa"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--quant", default="none", choices=["none", "8bit", "4bit"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        _die(f"Missing transformers/torch. Install: python -m pip install -r requirements.txt\n{e}")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    quantization_config = None
    if args.quant in ("8bit", "4bit"):
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            _die(f"Quantization requested but BitsAndBytesConfig unavailable. Install bitsandbytes.\n{e}")
        if args.quant == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    model_kwargs: Dict[str, Any] = {
        "device_map": args.device_map,
        "low_cpu_mem_usage": True,
        "token": hf_token,
        "torch_dtype": torch_dtype,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    examples = load_examples(args.dataset, args.split, args.max_samples)
    golds = [e.gold for e in examples]

    preds: List[str] = []
    for ex in examples:
        prompt = build_prompt(ex)
        inputs = tokenizer(prompt, return_tensors="pt")
        try:
            model_device = getattr(model, "device", None)
            if model_device is not None and str(model_device) != "cpu":
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
        except Exception:
            pass

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        gen_text = decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded
        preds.append(extract_answer(args.dataset, gen_text))

    acc = score(args.dataset, preds, golds)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset} split={args.split} n={len(golds)}")
    print(f"Accuracy (simple extraction): {acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



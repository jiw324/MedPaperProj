#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AI-Generated Code Header
# **Intent:** Provide a lightweight CLI/REPL to run Bio+ClinicalBERT for masked LM or embeddings.
# **Optimization:** Keeps defaults simple (fill-mask) and uses CPU/GPU automatically via Transformers device_map.
# **Safety:** Handles HF token via env vars; avoids printing secrets; guards Windows UTF-8 console issues.

"""
Bio+ClinicalBERT Text CLI
=========================

Supported tasks:
- fill-mask (default): predict tokens for `[MASK]`
- embeddings: return pooled sentence embeddings (prints vector length, not full vector)

Usage (PowerShell):
  cd "...\Medical LLM Model\Bio+ClinicalBERT"
  $env:HF_TOKEN="hf_..."   # if gated/private; do NOT paste token into chat
  python run_bio_clinicalbert_cli.py --task fill-mask --model emilyalsentzer/Bio_ClinicalBERT

Commands in the REPL:
- /exit or /quit  -> leave
- /reset          -> no-op (kept for consistency)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional


def _configure_windows_utf8() -> None:
    if sys.platform != "win32":
        return
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    except Exception:
        pass


def _die(msg: str, code: int = 1) -> "None":
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _load_pipeline(task: str, model_id: str, hf_token: Optional[str], device_map: str):
    try:
        from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, pipeline
    except Exception as e:
        _die(
            "Missing dependency: transformers.\n"
            "Install from this folder:\n"
            "  python -m pip install -r requirements.txt\n"
            f"Original error: {e}"
        )

    # // AI-SUGGESTION: pass token so gated repos work without exposing it.
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    if task == "fill-mask":
        model = AutoModelForMaskedLM.from_pretrained(model_id, device_map=device_map, token=hf_token)
    elif task == "embeddings":
        model = AutoModel.from_pretrained(model_id, device_map=device_map, token=hf_token)
    else:
        _die(f"Unsupported task: {task}")

    return pipeline(task if task == "fill-mask" else "feature-extraction", model=model, tokenizer=tok, device_map=device_map)


def main() -> int:
    _configure_windows_utf8()

    parser = argparse.ArgumentParser(
        description="Bio+ClinicalBERT CLI for fill-mask or embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="emilyalsentzer/Bio_ClinicalBERT", help="HF model id to load.")
    parser.add_argument("--task", default="fill-mask", choices=["fill-mask", "embeddings"], help="Inference task.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map (auto/cpu/cuda...).")
    parser.add_argument("--top-k", type=int, default=5, help="Top predictions for fill-mask.")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    try:
        nlp = _load_pipeline(args.task, args.model, hf_token, args.device_map)
    except Exception as e:
        _die(f"Failed to load model/pipeline. If gated, set HF_TOKEN. Original error: {e}")

    print("\nBio+ClinicalBERT CLI")
    print(f"Model: {args.model}")
    print(f"Task:  {args.task}")
    print("Commands: /exit, /quit, /reset\n")

    while True:
        try:
            text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0

        if not text:
            continue
        low = text.lower()
        if low in ("/exit", "/quit"):
            return 0
        if low == "/reset":
            # // AI-SUGGESTION: placeholder for future context handling.
            print("[OK] Nothing to reset (stateless mode).\n")
            continue

        if args.task == "fill-mask":
            if "[MASK]" not in text:
                print("[WARN] No [MASK] token found. Inserted at end.\n")
                text = text.rstrip() + " [MASK]"
            try:
                results = nlp(text, top_k=args.top_k)
            except Exception as e:
                _die(f"Inference failed: {e}")

            if not isinstance(results, list):
                results = [results]
            print("Predictions:")
            for r in results:
                token = r.get("token_str", "").strip()
                score = r.get("score")
                print(f"  - {token} (p={score:.4f})")
            print()
        else:  # embeddings
            try:
                emb = nlp(text)
            except Exception as e:
                _die(f"Inference failed: {e}")
            # // AI-SUGGESTION: avoid printing full vectors to keep output readable.
            if isinstance(emb, list):
                try:
                    length = len(emb[0][0])
                except Exception:
                    length = None
            else:
                length = getattr(emb, "shape", None)
            print(f"[OK] Embedding computed. Shape/length: {length}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



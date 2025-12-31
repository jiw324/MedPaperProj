#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AI-Generated Code Header
# **Intent:** Provide a lightweight CLI/REPL to run Clinical LLaMA-2 variants for causal text generation.
# **Optimization:** Uses `device_map=auto` by default; optional dtype override; keeps defaults modest.
# **Safety:** Handles HF token via env vars; avoids printing secrets; guards Windows UTF-8 console issues.

"""
Clinical LLaMA-2 Variants Text CLI
==================================

Usage (PowerShell):
  cd "...\Medical LLM Model\Clinical LLaMA-2 variants"
  $env:HF_TOKEN="hf_..."   # if gated/private; do NOT paste token into chat
  python run_clinical_llama2_cli.py --model wanglab/Clinical-LLaMA-2-7B --prompt "Hello"

Commands in the REPL:
- /exit or /quit  -> leave
- /reset          -> clear context (stateless here, kept for consistency)
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
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


def _load_model(model_id: str, hf_token: Optional[str], device_map: str, dtype: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        _die(
            "Missing dependency: transformers.\n"
            "Install from this folder:\n"
            "  python -m pip install -r requirements.txt\n"
            f"Original error: {e}"
        )

    torch_dtype = None
    if dtype != "auto":
        try:
            import torch
            torch_dtype = getattr(torch, dtype)
        except Exception:
            torch_dtype = None

    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model_kwargs = {"device_map": device_map, "low_cpu_mem_usage": True, "token": hf_token}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    return tok, model


def generate(prompt: str, tokenizer, model, max_new_tokens: int, temperature: float, top_p: float) -> str:
    try:
        import torch
        inputs = tokenizer(prompt, return_tensors="pt")
        if getattr(model, "device", None) and str(model.device) != "cpu":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        _die(f"Generation failed: {e}")
    return ""


def main() -> int:
    _configure_windows_utf8()

    parser = argparse.ArgumentParser(
        description="Clinical LLaMA-2 variants CLI for causal text generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="wanglab/Clinical-LLaMA-2-7B", help="HF model id to load.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map (auto/cpu/cuda...).")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--prompt", default=None, help="Optional single-shot prompt (skips REPL).")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    try:
        tokenizer, model = _load_model(args.model, hf_token, args.device_map, args.dtype)
    except Exception as e:
        _die(f"Failed to load model/tokenizer. If gated, set HF_TOKEN. Original error: {e}")

    def run_once(user_text: str) -> None:
        full = generate(user_text, tokenizer, model, args.max_new_tokens, args.temperature, args.top_p)
        print("\nAssistant>")
        print(textwrap.fill(full, width=100))
        print()

    if args.prompt:
        run_once(args.prompt)
        return 0

    print("\nClinical LLaMA-2 Variants CLI")
    print(f"Model: {args.model}")
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
            # // AI-SUGGESTION: stateless; nothing to reset today.
            print("[OK] Context cleared (stateless).\n")
            continue

        run_once(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



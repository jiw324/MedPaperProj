#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AI-Generated Code Header
# **Intent:** Provide a text-based interface (CLI/REPL) to run BioMistral-style biomedical LLM inference via Transformers.
# **Optimization:** Optional `device_map=auto` and optional 4/8-bit quantization (bitsandbytes) for GPU memory efficiency.
# **Safety:** Handles Hugging Face auth tokens, prints actionable errors, guards Windows console encoding issues.

"""
BioMistral Text CLI
===================

This is a lightweight terminal REPL for running a biomedical model (e.g., "BioMistral") using Transformers.

The BioMistral paper:
  https://arxiv.org/abs/2402.10373

Model card (BioMistral-7B):
  https://huggingface.co/BioMistral/BioMistral-7B

Auth:
- If your chosen model is gated/private, set HF_TOKEN before running.

PowerShell:
  $env:HF_TOKEN="hf_..."
  python run_biomistral_cli.py --model <model_id>
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from dataclasses import dataclass
from typing import Optional


# AI-SUGGESTION: Avoid UnicodeEncodeError on some Windows terminals by forcing UTF-8 output.
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


_configure_windows_utf8()


def _die(msg: str, code: int = 1) -> "None":
    print(msg, file=sys.stderr)
    raise SystemExit(code)


@dataclass(frozen=True)
class GenCfg:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95


def build_prompt(user_text: str, system: Optional[str]) -> str:
    """
    Many biomedical base models are not chat-tuned. Keep a simple prompt format.
    """
    system_block = f"System: {system.strip()}\n\n" if system and system.strip() else ""
    return f"{system_block}User: {user_text.strip()}\nAssistant:"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BioMistral-style biomedical LLM text interface (Transformers).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="BioMistral/BioMistral-7B",
        help="Hugging Face model id to load (default matches the BioMistral-7B model card).",
    )
    parser.add_argument("--device-map", default="auto", help="Transformers device_map (auto/cpu/cuda...).")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--quant", default="none", choices=["none", "8bit", "4bit"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--system",
        default="You are a helpful biomedical assistant. Provide educational information and encourage consulting clinicians for medical decisions.",
    )
    parser.add_argument("--no-system", action="store_true")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        _die(
            "Missing dependency: transformers.\n"
            "Install from this folder:\n"
            "  python -m pip install -r requirements.txt\n"
            f"Original error: {e}"
        )

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    torch_dtype = None
    if args.dtype != "auto":
        try:
            import torch
            torch_dtype = getattr(torch, args.dtype)
        except Exception:
            torch_dtype = None

    quantization_config = None
    if args.quant in ("8bit", "4bit"):
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            _die(
                "Quantization requested but BitsAndBytesConfig not available.\n"
                "Install bitsandbytes and ensure a compatible CUDA setup.\n"
                f"Original error: {e}"
            )
        if args.quant == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            import torch
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    except Exception as e:
        _die(
            "Failed to load tokenizer. If the model is gated/private, set HF_TOKEN.\n"
            f"Original error: {e}"
        )

    model_kwargs = {"device_map": args.device_map, "low_cpu_mem_usage": True, "token": hf_token}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    except Exception as e:
        _die(
            "Failed to load model. If this is OOM, try a smaller model or quantization.\n"
            f"Original error: {e}"
        )

    system = None if args.no_system else args.system
    gen = GenCfg(args.max_new_tokens, args.temperature, args.top_p)

    print("\n" + "=" * 72)
    print("BioMistral Experiment — Text CLI")
    print(f"Paper: https://arxiv.org/abs/2402.10373")
    print(f"Model: {args.model}")
    print("Commands: /exit, /reset")
    print("=" * 72 + "\n")

    context = ""
    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0

        if not user_text:
            continue
        if user_text.lower() in ("/exit", "/quit"):
            return 0
        if user_text.lower() == "/reset":
            context = ""
            print("[OK] Context cleared.\n")
            continue

        prompt = build_prompt((context + "\n" + user_text).strip() if context else user_text, system)
        inputs = tokenizer(prompt, return_tensors="pt")

        # AI-SUGGESTION: Best-effort move inputs if model is on a single device.
        try:
            model_device = getattr(model, "device", None)
            if model_device is not None and str(model_device) != "cpu":
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
        except Exception:
            pass

        try:
            import torch
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=gen.max_new_tokens,
                    temperature=gen.temperature,
                    do_sample=(gen.temperature > 0),
                    top_p=gen.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
        except Exception as e:
            _die(f"Generation failed.\nOriginal error: {e}")

        full_text = tokenizer.decode(out[0], skip_special_tokens=True)
        assistant_text = full_text.split("Assistant:", 1)[-1].strip() if "Assistant:" in full_text else full_text

        print("\nAssistant>")
        print(textwrap.fill(assistant_text, width=100))
        print()

        context = (context + "\n" + user_text).strip()
        if len(context) > 4000:
            context = context[-4000:]


if __name__ == "__main__":
    raise SystemExit(main())



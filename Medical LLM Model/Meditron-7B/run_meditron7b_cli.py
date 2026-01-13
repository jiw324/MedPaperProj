#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AI-Generated Code Header
# **Intent:** Provide a text-based interface (CLI/REPL) to run Meditron-7B via Transformers.
# **Optimization:** Supports `device_map=auto` and optional 4/8-bit quantization (bitsandbytes) for GPU memory efficiency.
# **Safety:** Handles HF auth tokens; avoids printing tokens; mitigates Windows console encoding issues.

"""
Meditron-7B Text CLI
====================

Model (default):
  epfl-llm/meditron-7b

Usage (PowerShell):
  cd "...\Medical LLM Model\Meditron-7B"
  $env:HF_TOKEN="hf_..."   # only if gated
  python run_meditron7b_cli.py --model epfl-llm/meditron-7b --device-map auto

Commands:
  /exit   quit
  /reset  clear conversation
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from dataclasses import dataclass
from typing import Dict, List


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


def _fallback_prompt(messages: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        else:
            parts.append(f"Assistant: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _build_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    # AI-SUGGESTION: Prefer chat_template if the tokenizer provides one; otherwise fall back to a simple prompt.
    try:
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass
    return _fallback_prompt(messages)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Meditron-7B (Transformers) text interface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="epfl-llm/meditron-7b")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--quant", default="none", choices=["none", "8bit", "4bit"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--system",
        default="You are a helpful medical language model. Provide educational information and encourage consulting clinicians for medical decisions.",
    )
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        _die(
            "Missing dependencies. Install from this folder:\n"
            "  python -m pip install -r requirements.txt\n"
            f"Original error: {e}"
        )

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    torch_dtype = None
    if args.dtype != "auto":
        torch_dtype = getattr(torch, args.dtype, None)

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

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    except Exception as e:
        _die(
            "Failed to load tokenizer. If gated, ensure access is granted and HF_TOKEN is set.\n"
            f"Model page: https://huggingface.co/{args.model}\n"
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
            "Failed to load model. If OOM, try --quant 4bit/8bit and ensure you have sufficient VRAM.\n"
            f"Original error: {e}"
        )

    gen = GenCfg(args.max_new_tokens, args.temperature, args.top_p)
    messages: List[Dict[str, str]] = [{"role": "system", "content": args.system}]

    print("\n" + "=" * 72)
    print("Meditron-7B — Text CLI")
    print(f"Model: {args.model}")
    print("Commands: /exit, /reset")
    print("=" * 72 + "\n")

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
            messages = [{"role": "system", "content": args.system}]
            print("[OK] Conversation reset.\n")
            continue

        messages.append({"role": "user", "content": user_text})
        prompt = _build_prompt(tokenizer, messages)

        inputs = tokenizer(prompt, return_tensors="pt")
        try:
            model_device = getattr(model, "device", None)
            if model_device is not None and str(model_device) != "cpu":
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
        except Exception:
            pass

        try:
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

        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        assistant_text = decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()
        if not assistant_text:
            assistant_text = decoded.strip()

        print("\nAssistant>")
        print(textwrap.fill(assistant_text, width=100))
        print()

        messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    raise SystemExit(main())








#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AI-Generated Code Header
# **Intent:** Provide a text-based interface (CLI/REPL) to run OpenBioLLM (Llama-3) using the correct chat template.
# **Optimization:** Uses `device_map=auto` and bf16 (when available) for GPU efficiency; optional 4/8-bit quantization.
# **Safety:** Handles HF auth tokens; avoids printing tokens; mitigates Windows console encoding issues.

"""
OpenBioLLM Text CLI (Llama-3 chat template)
==========================================

Model card (link provided by user):
  https://huggingface.co/aaditya/Llama3-OpenBioLLM-70B

Important:
- The model card recommends using the **Llama-3 Instruct chat template** via `tokenizer.apply_chat_template(...)`.
- This model may require accepting a license / gated access depending on underlying base model requirements.

Usage (PowerShell):
  cd "...\Medical LLM Model\OpenBioLLM"
  $env:HF_TOKEN="hf_..."    # if needed; do NOT paste token in chat
  python run_openbiollm_cli.py
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from dataclasses import dataclass
from typing import List, Optional


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
    temperature: float = 0.0
    top_p: float = 0.9


def _get_terminators(tokenizer) -> List[int]:
    # Llama-3 commonly uses <|eot_id|> as end-of-turn.
    terminators = []
    if getattr(tokenizer, "eos_token_id", None) is not None:
        terminators.append(tokenizer.eos_token_id)
    try:
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int) and eot >= 0:
            terminators.append(eot)
    except Exception:
        pass
    # Dedup
    return list(dict.fromkeys(terminators))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OpenBioLLM (Llama-3) text interface using apply_chat_template.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="aaditya/Llama3-OpenBioLLM-70B",
        help="Hugging Face model id to load.",
    )
    parser.add_argument("--device-map", default="auto", help="Transformers device_map (auto/cpu/cuda...).")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--quant", default="none", choices=["none", "8bit", "4bit"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--system",
        default=(
            "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge "
            "and practical experience. Provide educational information; do not provide medical diagnosis."
        ),
    )
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as e:
        _die(
            "Missing dependencies. Install from this folder:\n"
            "  python -m pip install -r requirements.txt\n"
            f"Original error: {e}"
        )

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    torch_dtype = None
    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "float32":
        torch_dtype = torch.float32
    else:
        # AI-SUGGESTION: Prefer bf16 on modern GPUs; otherwise transformers may choose.
        torch_dtype = None

    quantization_config = None
    if args.quant == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif args.quant == "4bit":
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
            "Failed to load tokenizer. If gated/license-restricted, ensure you've accepted terms and set HF_TOKEN.\n"
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
            "Failed to load model. If OOM, try --quant 4bit/8bit or a smaller model.\n"
            f"Original error: {e}"
        )

    gen = GenCfg(args.max_new_tokens, args.temperature, args.top_p)
    terminators = _get_terminators(tokenizer)

    print("\n" + "=" * 72)
    print("OpenBioLLM — Text CLI")
    print(f"Model: {args.model}")
    print("Commands: /exit, /reset")
    print("=" * 72 + "\n")

    # Maintain a chat message list so the template is applied correctly.
    messages = [{"role": "system", "content": args.system}]

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

        # AI-SUGGESTION: model card suggests apply_chat_template(..., add_generation_prompt=True)
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            _die(
                "Tokenizer does not support apply_chat_template as expected.\n"
                f"Original error: {e}"
            )

        inputs = tokenizer(prompt, return_tensors="pt")
        # If model is on single device, move inputs to it.
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
                    do_sample=True,
                    temperature=gen.temperature,
                    top_p=gen.top_p,
                    eos_token_id=terminators if terminators else None,
                    pad_token_id=tokenizer.eos_token_id,
                )
        except Exception as e:
            _die(f"Generation failed.\nOriginal error: {e}")

        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        # Heuristic: assistant text is whatever was generated after the prompt.
        assistant_text = decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded
        if not assistant_text:
            assistant_text = decoded.strip()

        print("\nAssistant>")
        print(textwrap.fill(assistant_text, width=100))
        print()

        messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    raise SystemExit(main())





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AI-Generated Code Header
# **Intent:** Load a Meditron model from Hugging Face via Transformers and provide a text-based chat interface (CLI/REPL).
# **Optimization:** Supports optional `device_map=auto` and optional 4/8-bit quantization when available (GPU + bitsandbytes).
# **Safety:** Validates auth/token presence for gated repos, prints actionable errors, guards Windows console encoding issues.

"""
Text-based Meditron CLI
======================

This is a simple REPL (terminal chat) for Meditron models using Hugging Face Transformers.

Key notes:
- Meditron-70B is typically a gated model on Hugging Face and requires authentication.
- 70B is extremely large; you usually need multi-GPU or heavy quantization to run it.

Authentication (recommended):
1) Request access on the model page (if gated).
2) Create a token (Read) at Hugging Face settings.
3) Set env var before running:

PowerShell:
  $env:HF_TOKEN="hf_..."
  python chat_cli.py --model epfl-llm/meditron-70b

Cmd.exe:
  set HF_TOKEN=hf_...
  python chat_cli.py --model epfl-llm/meditron-70b
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
        # Python 3.7+: reconfigure is best when available
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    except Exception:
        # Best-effort only; do not fail startup on console quirks.
        pass


_configure_windows_utf8()


def _die(msg: str, code: int = 1) -> "None":
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _detect_torch_device() -> str:
    try:
        import torch
    except Exception:
        return "unknown"
    if torch.cuda.is_available():
        return "cuda"
    # Apple MPS not relevant on Windows, but harmless to check.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95


def build_prompt(user_text: str, system: Optional[str]) -> str:
    """
    Meditron base models are not always instruction-tuned.
    This simple prompt format is a reasonable default for experimentation.
    """
    system_block = ""
    if system and system.strip():
        system_block = f"System: {system.strip()}\n\n"
    return f"{system_block}User: {user_text.strip()}\nAssistant:"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Text-based chat interface for Meditron models (Transformers).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="epfl-llm/meditron-70b",
        help="Hugging Face model id (e.g., epfl-llm/meditron-70b or epfl-llm/meditron-7b).",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map (use 'auto' for multi-device placement; use 'cpu' to force CPU).",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype to request (effective mainly on GPU).",
    )
    parser.add_argument(
        "--quant",
        default="none",
        choices=["none", "8bit", "4bit"],
        help="Optional quantization (requires bitsandbytes; typically requires CUDA).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--system",
        default="You are a helpful medical language model. Provide educational information and encourage consulting a clinician for medical decisions.",
        help="System message prefix (optional).",
    )
    parser.add_argument(
        "--no-system",
        action="store_true",
        help="Disable system message.",
    )
    args = parser.parse_args()

    # --- Dependency imports ---
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        _die(
            "Missing dependency: transformers.\n"
            "Install from this folder:\n"
            "  python -m pip install -r requirements.txt\n"
            f"Original error: {e}"
        )

    # --- HF authentication for gated models ---
    # AI-SUGGESTION: Prefer env var token to avoid interactive CLI login (which can break on Windows encoding).
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    # --- Torch / dtype ---
    torch_device = _detect_torch_device()
    if torch_device == "cpu" and args.model.endswith("70b"):
        print(
            "[WARN] You appear to be on CPU. Meditron-70B is extremely large and will likely not fit.\n"
            "       Consider a smaller model id (e.g. epfl-llm/meditron-7b) or use a GPU + quantization."
        )

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
                "Install bitsandbytes and a compatible setup.\n"
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

    # --- Load tokenizer/model (per user request: these lines are the core load calls) ---
    # AI-SUGGESTION: Pass `token=...` for gated repos. If you omit token, HF will require cached login.
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    except Exception as e:
        if "gated" in str(e).lower() or "restricted" in str(e).lower() or "401" in str(e):
            _die(
                "Access denied (gated/restricted model). You must request access and authenticate.\n\n"
                "Do this:\n"
                f"- Request access on the model page: https://huggingface.co/{args.model}\n"
                "- Create a token (Read) at: https://huggingface.co/settings/tokens\n"
                "- Set env var before running:\n"
                "    PowerShell:  $env:HF_TOKEN=\"hf_...\"\n"
                "    CMD:         set HF_TOKEN=hf_...\n\n"
                f"Original error: {e}"
            )
        _die(f"Failed to load tokenizer.\nOriginal error: {e}")

    model_kwargs = {
        "device_map": args.device_map,
        "low_cpu_mem_usage": True,
        "token": hf_token,
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    except Exception as e:
        if "gated" in str(e).lower() or "restricted" in str(e).lower() or "401" in str(e):
            _die(
                "Access denied (gated/restricted model). You must request access and authenticate.\n\n"
                "Do this:\n"
                f"- Request access on the model page: https://huggingface.co/{args.model}\n"
                "- Create a token (Read) at: https://huggingface.co/settings/tokens\n"
                "- Set env var before running:\n"
                "    PowerShell:  $env:HF_TOKEN=\"hf_...\"\n"
                "    CMD:         set HF_TOKEN=hf_...\n\n"
                f"Original error: {e}"
            )
        _die(
            "Failed to load model.\n"
            "If this looks like an OOM issue, try:\n"
            "  - a smaller model (epfl-llm/meditron-7b)\n"
            "  - --quant 8bit or --quant 4bit (GPU + bitsandbytes)\n"
            f"\nOriginal error: {e}"
        )

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    system = None if args.no_system else args.system

    print("\n" + "=" * 72)
    print("Meditron CLI (text interface)")
    print(f"Model: {args.model}")
    print("Type '/exit' to quit, '/reset' to clear context.")
    print("=" * 72 + "\n")

    # Simple rolling context (not a full chat template; keeps it lightweight).
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

        prompt = build_prompt(context + ("\n" if context else "") + user_text, system)

        inputs = tokenizer(prompt, return_tensors="pt")
        try:
            # If model is sharded across devices via device_map, inputs can stay on CPU.
            # If model is on a single device, move tensors to that device.
            model_device = getattr(model, "device", None)
            if model_device is not None and str(model_device) != "cpu":
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
        except Exception:
            pass

        # Generate
        try:
            import torch
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=gen_cfg.max_new_tokens,
                    temperature=gen_cfg.temperature,
                    do_sample=(gen_cfg.temperature > 0),
                    top_p=gen_cfg.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
        except Exception as e:
            _die(f"Generation failed.\nOriginal error: {e}")

        full_text = tokenizer.decode(out[0], skip_special_tokens=True)
        # Best-effort extraction of assistant portion:
        assistant_text = full_text.split("Assistant:", 1)[-1].strip() if "Assistant:" in full_text else full_text

        print("\nAssistant>")
        print(textwrap.fill(assistant_text, width=100))
        print()

        # Update context (keep it bounded to avoid infinite growth)
        context = (context + "\n" + user_text).strip()
        if len(context) > 4000:
            context = context[-4000:]


if __name__ == "__main__":
    raise SystemExit(main())



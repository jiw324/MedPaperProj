#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AI-Generated Code Header
# **Intent:** Verify Hugging Face authentication + gated access for Clinical Camel without downloading large weights.
# **Optimization:** Downloads only `config.json` (tiny) to confirm permissions.
# **Safety:** Does not print tokens; prints HF account identity; mitigates Windows console encoding issues.

"""
Clinical Camel Access Checker
=============================

Model:
  https://huggingface.co/wanglab/ClinicalCamel-70B

Usage (PowerShell):
  cd "...\Medical LLM Model\Clinical Camel"
  $env:HF_TOKEN="hf_..."   # if the model is gated; do NOT paste token into chat
  python check_access.py --model wanglab/ClinicalCamel-70B
"""

from __future__ import annotations

import argparse
import os
import sys


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


def main() -> int:
    _configure_windows_utf8()
    parser = argparse.ArgumentParser(description="Check HF access by downloading config.json only.")
    parser.add_argument("--model", default="wanglab/ClinicalCamel-70B")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download, whoami
    except Exception as e:
        print(f"Missing huggingface-hub. Install: python -m pip install -r requirements.txt\n{e}", file=sys.stderr)
        return 1

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    try:
        info = whoami(token=token) if token else whoami()
        print(f"[OK] Logged in as: {info.get('name')}")
    except Exception as e:
        print(f"[WARN] whoami() failed (set HF_TOKEN if needed): {e}")

    try:
        path = hf_hub_download(repo_id=args.model, filename="config.json", token=token)
        print(f"[OK] Access confirmed. Downloaded: {path}")
        return 0
    except Exception as e:
        print(f"[ERROR] Access failed for {args.model}: {e}", file=sys.stderr)
        print(f"Model page: https://huggingface.co/{args.model}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())



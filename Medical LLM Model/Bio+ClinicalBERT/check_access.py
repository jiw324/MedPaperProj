#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AI-Generated Code Header
# **Intent:** Verify Hugging Face authentication + access for Bio+ClinicalBERT without downloading large weights.
# **Optimization:** Downloads only a tiny file (`config.json` by default) to confirm permissions.
# **Safety:** Does not print tokens; identifies the logged-in HF account; handles Windows UTF-8 console quirks.

"""
Bio+ClinicalBERT Access Checker
===============================

Models (examples):
- https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
- https://huggingface.co/emilyalsentzer/clinicalbert

Usage (PowerShell):
  cd "...\Medical LLM Model\Bio+ClinicalBERT"
  $env:HF_TOKEN="hf_..."   # optional; do NOT paste tokens into chat
  python check_access.py --model emilyalsentzer/Bio_ClinicalBERT
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

    parser = argparse.ArgumentParser(description="Check HF access to Bio+ClinicalBERT by downloading a tiny file only.")
    parser.add_argument("--model", default="emilyalsentzer/Bio_ClinicalBERT", help="HF model id to check access for.")
    parser.add_argument(
        "--file",
        default="config.json",
        help="Small file to download for the access check (default: config.json).",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download, whoami
    except Exception as e:
        print(
            "Missing dependency: huggingface-hub\n"
            "Install from this folder:\n"
            "  python -m pip install -r requirements.txt\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        return 1

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    try:
        info = whoami(token=token) if token else whoami()
        print(f"[OK] Logged in as: {info.get('name')} (type={info.get('type')})")
    except Exception as e:
        print("[WARN] Could not call whoami(). If the repo is gated, set HF_TOKEN first.")
        print(f"       whoami() error: {e}")

    try:
        # // AI-SUGGESTION: limit to a small file so the access check avoids heavy downloads.
        path = hf_hub_download(repo_id=args.model, filename=args.file, token=token)
        print(f"[OK] Access confirmed. Downloaded: {path}")
        return 0
    except Exception as e:
        print(f"[ERROR] Access failed for {args.model}: {e}", file=sys.stderr)
        print(f"Model page: https://huggingface.co/{args.model}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())



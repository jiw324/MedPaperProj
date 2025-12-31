#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AI-Generated Code Header
# **Intent:** Verify Hugging Face authentication + gated access for a Meditron model without downloading large weights.
# **Optimization:** Downloads only `config.json` (tiny) to confirm permissions; avoids model load.
# **Safety:** Does not print tokens; prints account identity via `whoami`; handles Windows encoding quirks.

"""
Meditron Access Checker
======================

This script checks whether your current Hugging Face authentication has access to a (possibly gated) model.
It performs a small download of `config.json` only.

Usage (PowerShell):
  cd "...\Medical LLM Model\Meditron"
  $env:HF_TOKEN="hf_..."   # do NOT paste token into chat
  python check_access.py --model epfl-llm/meditron-70b

Notes:
- GPU vs CPU does not affect access checks.
- If this prints a 403 "not in authorized list", the account behind your token is not approved for that model.
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

    parser = argparse.ArgumentParser(description="Check HF access to a Meditron model (downloads config.json only).")
    parser.add_argument("--model", default="epfl-llm/meditron-70b", help="HF model id to check.")
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

    # Identify account (does not reveal token)
    try:
        info = whoami(token=token) if token else whoami()
        print(f"[OK] Logged in as: {info.get('name')} (type={info.get('type')})")
    except Exception as e:
        print("[WARN] Could not call whoami(). If the repo is gated, set HF_TOKEN first.")
        print(f"       whoami() error: {e}")

    # Tiny download to confirm access
    try:
        path = hf_hub_download(repo_id=args.model, filename="config.json", token=token)
        print(f"[OK] Access confirmed. Downloaded: {path}")
        return 0
    except Exception as e:
        print("[ERROR] Access check failed.")
        print(f"Model: {args.model}")
        print(f"Error: {e}")
        print(
            "\nCommon fixes:\n"
            "- Ensure you are approved on the model page while logged into the SAME account as your token\n"
            f"  https://huggingface.co/{args.model}\n"
            "- Revoke and recreate a new 'Read' token, then set it via $env:HF_TOKEN\n",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())



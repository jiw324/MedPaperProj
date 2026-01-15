"""
gpt51_api.py
// AI-Generated Code Header
// **Intent:** Provide a minimal, robust API wrapper + CLI to run GPT-5.1 on JSON chat prompts and save generations.
// **Optimization:** Retries with exponential backoff; writes JSONL atomically.
// **Safety:** Validates prompt JSON, avoids leaking API keys, handles timeouts/rate limits gracefully.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_MODEL = "gpt-5.1"

def _model_key(model: str) -> str:
    # AI-SUGGESTION: Normalize model ids that may include version suffixes.
    m = (model or "").strip()
    if m.startswith("gpt-5.1"):
        return "gpt-5.1"
    return m


def _apply_attack_override(
    *,
    messages: List[Dict[str, str]],
    row: Dict[str, Any],
    model: str,
) -> List[Dict[str, str]]:
    if str(row.get("condition", "")).lower() != "attack":
        return messages
    mapping = row.get("attack_messages_by_model")
    if not isinstance(mapping, dict):
        return messages
    key = _model_key(model)
    attack_msg = mapping.get(key) or mapping.get("default")
    if not isinstance(attack_msg, str) or not attack_msg.strip():
        return messages
    if len(messages) >= 3:
        messages[-1] = {"role": "user", "content": attack_msg}
    else:
        messages.append({"role": "user", "content": attack_msg})
    return messages


def _die(msg: str, code: int = 1) -> "None":
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _load_openai_client():
    """
    Lazily import OpenAI SDK.

    Install:
      python -m pip install openai
    Env:
      OPENAI_API_KEY=...
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: openai.\n"
            "Install:\n"
            "  python -m pip install openai\n"
            f"Original error: {e}"
        ) from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def parse_messages_json(prompt_text: str) -> List[Dict[str, str]]:
    """
    Parse a JSON array of messages like:
      [{"role":"user","content":"..."}, ...]
    """
    try:
        obj = json.loads(prompt_text)
    except Exception as e:
        raise ValueError(f"Prompt is not valid JSON: {e}") from e

    if not isinstance(obj, list) or not obj:
        raise ValueError("Prompt JSON must be a non-empty list of {role, content} objects.")

    out: List[Dict[str, str]] = []
    for i, item in enumerate(obj):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt message {i} is not an object.")
        role = item.get("role")
        content = item.get("content")
        if role not in ("system", "user", "assistant", "developer"):
            raise ValueError(f"Prompt message {i} has invalid role: {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"Prompt message {i} has non-string content.")
        out.append({"role": role, "content": content})
    return out


@dataclass(frozen=True)
class GPTResult:
    text: str
    raw: Dict[str, Any]


def call_gpt51_chat(
    *,
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout_s: int = 120,
    max_retries: int = 6,
    retry_base_s: float = 1.0,
    retry_jitter_s: float = 0.3,
) -> GPTResult:
    """
    Call GPT-5.1 using the OpenAI Chat Completions API via the official SDK.
    Retries transient errors with exponential backoff + jitter.
    """
    client = _load_openai_client()
    rng = random.Random(0)

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                # AI-SUGGESTION: GPT-5 models may require max_completion_tokens instead of max_tokens.
                max_completion_tokens=max_tokens,
                timeout=timeout_s,
            )
            text = (resp.choices[0].message.content or "").strip()
            raw = resp.model_dump() if hasattr(resp, "model_dump") else json.loads(resp.json())
            return GPTResult(text=text, raw=raw)
        except Exception as e:
            last_err = e
            if attempt >= max_retries:
                break
            sleep_s = retry_base_s * (2**attempt) + rng.random() * retry_jitter_s
            time.sleep(sleep_s)

    raise RuntimeError(f"OpenAI request failed after {max_retries + 1} attempts: {last_err}") from last_err


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def run_prompts_jsonl(
    *,
    prompts_jsonl: Path,
    out_jsonl: Path,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    max_retries: int,
    limit: int,
) -> None:
    rows_out: List[Dict[str, Any]] = []
    n = 0
    for row in iter_jsonl(prompts_jsonl):
        if limit > 0 and n >= limit:
            break
        case_id = str(row.get("case_id", ""))
        prompt_text = row.get("prompt")
        if not isinstance(prompt_text, str):
            rows_out.append({"case_id": case_id, "error": "missing prompt", "output_text": ""})
            n += 1
            continue
        try:
            messages = parse_messages_json(prompt_text)
            messages = _apply_attack_override(messages=messages, row=row, model=model)
            res = call_gpt51_chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                max_retries=max_retries,
            )
            rows_out.append({"case_id": case_id, "error": "", "output_text": res.text, "raw": res.raw})
        except Exception as e:
            rows_out.append({"case_id": case_id, "error": str(e), "output_text": ""})
        n += 1
        if n % 25 == 0:
            print(f"[gpt5.1] processed {n} prompts", file=sys.stderr)

    write_jsonl(out_jsonl, rows_out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run GPT-5.1 on prompts.jsonl produced by allergy_omission_experiment.py")
    ap.add_argument(
        "--prompts_jsonl",
        type=Path,
        default=Path("Research_setup/output/allergy_omission/results/prompts.jsonl"),
    )
    ap.add_argument(
        "--out_jsonl",
        type=Path,
        default=Path("Research_setup/output/allergy_omission/results/generations_gpt51.jsonl"),
    )
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.0)
    # AI-SUGGESTION: Keep CLI flag name stable; internally mapped to max_completion_tokens for GPT-5.
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--timeout_s", type=int, default=120)
    ap.add_argument("--max_retries", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="If >0, only run first N prompts.")
    args = ap.parse_args(argv)

    if not args.prompts_jsonl.exists():
        _die(f"prompts_jsonl not found: {args.prompts_jsonl}")

    run_prompts_jsonl(
        prompts_jsonl=args.prompts_jsonl,
        out_jsonl=args.out_jsonl,
        model=args.model,
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        timeout_s=int(args.timeout_s),
        max_retries=int(args.max_retries),
        limit=int(args.limit),
    )
    print(f"Wrote: {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



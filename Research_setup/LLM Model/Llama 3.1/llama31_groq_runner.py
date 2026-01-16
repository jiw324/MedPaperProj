"""
llama31_groq_runner.py
// AI-Generated Code Header
// **Intent:** Run Llama 3.1 remotely on Groq using prompts.jsonl (JSON message arrays) and write generations JSONL (pure I/O runner).
// **Optimization:** Retries with exponential backoff; sequential processing; minimal dependencies.
// **Safety:** Validates prompt JSON, avoids leaking API keys, writes outputs atomically. No task-specific scoring/mutation here.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


# AI-SUGGESTION: Prefer a per-runner env var to avoid collisions with other Groq runners.
DEFAULT_MODEL = (
    os.environ.get("LLAMA31_GROQ_MODEL")
    or os.environ.get("GROQ_MODEL")
    or "llama-3.1-8b-instant"  # AI-SUGGESTION: requested model id; override if needed.
).strip()


def _die(msg: str, code: int = 1) -> "None":
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def parse_messages_json(prompt_text: str) -> List[Dict[str, str]]:
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


def write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _load_groq_client(*, api_key: str):
    try:
        from groq import Groq  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: groq.\n"
            "Install:\n"
            "  python -m pip install groq\n"
            f"Original error: {e}"
        ) from e
    return Groq(api_key=api_key)


def groq_chat_with_backoff(
    *,
    client,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    max_retries: int,
    backoff_base_s: float,
    verbose: bool,
) -> str:
    def _extract_message_text(msg: Any) -> str:
        parts: List[str] = []
        reasoning = getattr(msg, "reasoning", None)
        if isinstance(reasoning, str) and reasoning.strip():
            parts.append(reasoning.strip())
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
        return "\n\n".join(parts).strip()

    rng = random.Random(0)
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            # AI-SUGGESTION: If max_tokens <= 0, omit the token limit (still capped by Groq/model).
            try:
                if max_tokens and max_tokens > 0:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                        timeout=timeout_s,
                    )
                else:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        timeout=timeout_s,
                    )
            except TypeError:
                # AI-SUGGESTION: Backwards compat with older Groq SDK args.
                if max_tokens and max_tokens > 0:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout_s,
                    )
                else:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        timeout=timeout_s,
                    )

            return _extract_message_text(resp.choices[0].message)
        except Exception as e:
            last_err = e
            if attempt >= max_retries:
                break
            sleep_s = backoff_base_s * (2**attempt) + rng.random() * 0.25
            if verbose:
                print(
                    f"[llama31-groq] retry={attempt+1}/{max_retries} sleep={sleep_s:.2f}s err={e}",
                    file=sys.stderr,
                )
            time.sleep(sleep_s)
    raise RuntimeError(f"Groq request failed after {max_retries + 1} attempts: {last_err}") from last_err


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run Llama 3.1 on Groq using prompts.jsonl and write generations JSONL (pure I/O runner)."
    )
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Groq model name for Llama 3.1.")
    ap.add_argument("--max_tokens", type=int, default=0, help="0 = omit limit (capped by Groq/model).")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sleep_s", type=float, default=0.0)
    ap.add_argument("--timeout_s", type=int, default=60)
    ap.add_argument("--max_retries", type=int, default=6)
    ap.add_argument("--backoff_base_s", type=float, default=1.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--print_messages", action="store_true")
    ap.add_argument(
        "--prompts_jsonl",
        type=Path,
        default=Path("Research_setup/output/allergy_omission/results/prompts.jsonl"),
    )
    ap.add_argument(
        "--out_jsonl",
        type=Path,
        default=Path("Research_setup/output/allergy_omission/results/generations_llama31_groq.jsonl"),
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    if not args.prompts_jsonl.exists():
        _die(f"prompts_jsonl not found: {args.prompts_jsonl}")
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        _die("GROQ_API_KEY is not set.")

    client = _load_groq_client(api_key=api_key)

    out_rows: List[Dict[str, Any]] = []
    n = 0
    for row in iter_jsonl(args.prompts_jsonl):
        if args.limit > 0 and n >= args.limit:
            break
        case_id = str(row.get("case_id", ""))
        prompt_text = row.get("prompt")
        if not isinstance(prompt_text, str):
            out_rows.append({"case_id": case_id, "error": "missing prompt", "output_text": ""})
            n += 1
            continue
        try:
            # AI-SUGGESTION: Runner does not mutate prompts; any attack/control instructions must be embedded in prompts.jsonl.
            msgs = parse_messages_json(prompt_text)
            if args.print_messages or args.verbose:
                print(
                    f"[llama31-groq] case={case_id} messages={json.dumps(msgs, ensure_ascii=False)}",
                    file=sys.stderr,
                )

            out_text = groq_chat_with_backoff(
                client=client,
                model=str(args.model),
                messages=msgs,
                temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
                timeout_s=int(args.timeout_s),
                max_retries=int(args.max_retries),
                backoff_base_s=float(args.backoff_base_s),
                verbose=bool(args.verbose),
            )
            out_rows.append(
                {
                    "case_id": case_id,
                    "error": "",
                    "output_text": out_text,
                }
            )
        except Exception as e:
            out_rows.append({"case_id": case_id, "error": str(e), "output_text": ""})
        n += 1
        if float(args.sleep_s) > 0:
            time.sleep(float(args.sleep_s))

    write_jsonl_atomic(args.out_jsonl, out_rows)
    print(f"Wrote: {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



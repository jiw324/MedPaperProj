"""
claude45_api.py
// AI-Generated Code Header
// **Intent:** Provide a minimal, robust API wrapper + CLI to run Claude Sonnet 4.5 on JSON chat prompts and save generations.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# AI-SUGGESTION: Default to Claude Sonnet 4.5 model id.
# Ref: Anthropic announcement / API model id.
DEFAULT_MODEL = "claude-sonnet-4-5"

def _model_key(model: str) -> str:
    # AI-SUGGESTION: Normalize model ids; allow future suffixes.
    m = (model or "").strip()
    if m.startswith("claude-sonnet-4-5"):
        return "claude-sonnet-4-5"
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


def _load_anthropic_client():
    """
    Lazily import Anthropic SDK.

    Install:
      python -m pip install anthropic
    Env:
      ANTHROPIC_API_KEY=...
    """
    try:
        import anthropic  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: anthropic.\n"
            "Install:\n"
            "  python -m pip install anthropic\n"
            f"Original error: {e}"
        ) from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    return anthropic.Anthropic(api_key=api_key)


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


def to_anthropic_parts(messages: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """
    Convert OpenAI-style messages into Anthropic's format:
      - system: list of content blocks, e.g. [{"type":"text","text":"..."}]
      - messages: list of {role: "user"|"assistant", content: [{"type":"text","text":"..."}]}

    We merge any "system"/"developer" messages into the system string.
    """
    system_chunks: List[str] = []
    out_msgs: List[Dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role in ("system", "developer"):
            system_chunks.append(content)
            continue
        # Anthropic supports "user" and "assistant" roles.
        if role not in ("user", "assistant"):
            continue
        # AI-SUGGESTION: Use explicit content blocks for compatibility with newer Anthropic API validation.
        out_msgs.append({"role": role, "content": [{"type": "text", "text": content}]})

    system_text = "\n\n".join([c.strip() for c in system_chunks if c.strip()]).strip()
    if not out_msgs:
        raise ValueError("No user/assistant messages found after conversion to Anthropic format.")
    system_blocks: List[Dict[str, str]] = [{"type": "text", "text": system_text}] if system_text else []
    return system_blocks, out_msgs


@dataclass(frozen=True)
class ClaudeResult:
    text: str
    raw: Dict[str, Any]


def call_claude45(
    *,
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout_s: int = 120,
    max_retries: int = 6,
    retry_base_s: float = 1.0,
    retry_jitter_s: float = 0.3,
) -> ClaudeResult:
    """
    Call Claude Sonnet 4.5 via the Anthropic Messages API.
    Retries transient errors with exponential backoff + jitter.
    """
    client = _load_anthropic_client()
    rng = random.Random(0)

    system_blocks, anthropic_messages = to_anthropic_parts(messages)

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.messages.create(
                model=model,
                # AI-SUGGESTION: Anthropic now expects `system` as a list of content blocks for some models/SDK versions.
                system=system_blocks,
                messages=anthropic_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_s,
            )
            # resp.content is typically a list of content blocks; join text blocks.
            text_parts: List[str] = []
            for block in getattr(resp, "content", []) or []:
                if getattr(block, "type", None) == "text":
                    text_parts.append(getattr(block, "text", "") or "")
            text = "".join(text_parts).strip()

            raw = resp.model_dump() if hasattr(resp, "model_dump") else json.loads(resp.json())
            return ClaudeResult(text=text, raw=raw)
        except Exception as e:
            last_err = e
            if attempt >= max_retries:
                break
            sleep_s = retry_base_s * (2**attempt) + rng.random() * retry_jitter_s
            time.sleep(sleep_s)

    raise RuntimeError(f"Anthropic request failed after {max_retries + 1} attempts: {last_err}") from last_err


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
            msgs = parse_messages_json(prompt_text)
            msgs = _apply_attack_override(messages=msgs, row=row, model=model)
            res = call_claude45(
                messages=msgs,
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
            print(f"[claude45] processed {n} prompts", file=sys.stderr)

    write_jsonl(out_jsonl, rows_out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run Claude Sonnet 4.5 on prompts.jsonl produced by allergy_omission_experiment.py"
    )
    ap.add_argument(
        "--prompts_jsonl",
        type=Path,
        default=Path("Research_setup/output/allergy_omission/results/prompts.jsonl"),
    )
    ap.add_argument(
        "--out_jsonl",
        type=Path,
        default=Path("Research_setup/output/allergy_omission/results/generations_claude_sonnet45.jsonl"),
    )
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.0)
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



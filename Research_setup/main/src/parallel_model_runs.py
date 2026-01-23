"""
parallel_model_runs.py
// AI-Generated Code Header
// **Intent:** Run 6 model runners in parallel (subprocess-per-model) against existing prompts JSONL files.
// **Optimization:** Parallelizes at the model level (safe, coarse-grained concurrency) and writes per-model logs.
// **Safety:** Validates input prompt paths, isolates outputs per model, supports per-process timeout, and returns non-zero on failures.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed


# AI-SUGGESTION: Default set requested by user: 6 different models (exclude "generic").
DEFAULT_MODEL_KEYS: Tuple[str, ...] = (
    "llama31_groq",
    "llama33_groq",
    "qwen3_groq",
    "gpt41",
    "gpt51",
    "claude45",
)


@dataclass(frozen=True)
class Job:
    name: str
    cmd: List[str]
    cwd: Path
    log_path: Path
    timeout_s: int


@dataclass(frozen=True)
class JobResult:
    name: str
    exit_code: int
    duration_s: float
    log_path: Path
    error: str


def _die(msg: str, code: int = 2) -> "None":
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _ensure_file(path: Path, *, label: str) -> None:
    if not path.exists() or not path.is_file():
        _die(f"{label} not found: {path}")


def _runner_cmd(
    *,
    py: str,
    runner_script: Path,
    prompts_jsonl: Path,
    out_jsonl: Path,
    model: str,
    limit: int,
    extra_args: Sequence[str],
) -> List[str]:
    # AI-SUGGESTION: Use list args (no shell) for safer quoting on Windows.
    cmd = [py, str(runner_script), "--model", model, "--prompts_jsonl", str(prompts_jsonl), "--out_jsonl", str(out_jsonl)]
    cmd.extend(list(extra_args))
    if limit is not None:
        cmd.extend(["--limit", str(int(limit))])
    return cmd


def _build_jobs(
    *,
    pipeline: str,
    root: Path,
    repo_root: Path,
    py: str,
    model_ids: Dict[str, str],
    limit: int,
    timeout_s: int,
    logs_dir: Path,
    model_keys: Sequence[str],
) -> List[Job]:
    pipeline = (pipeline or "").strip().lower()
    root = root.resolve()
    repo_root = repo_root.resolve()

    # Runner scripts (relative to Research_setup/).
    runners = {
        "llama31_groq": repo_root / "LLM Model" / "Llama 3.1" / "llama31_groq_runner.py",
        "llama33_groq": repo_root / "LLM Model" / "Llama 3.3" / "llama33_groq_runner.py",
        # AI-SUGGESTION: Repo names this "bedrock_runner" but it is invoked for Qwen in all pipelines.
        "qwen3_groq": repo_root / "LLM Model" / "Qwen 3" / "qwen3_bedrock_runner.py",
        "gpt41": repo_root / "LLM Model" / "GPT-4.1" / "gpt41_api.py",
        "gpt51": repo_root / "LLM Model" / "GPT-5.1" / "gpt51_api.py",
        "claude45": repo_root / "LLM Model" / "Claude 4.5" / "claude45_api.py",
    }

    jobs: List[Job] = []
    for mk in model_keys:
        if mk not in runners:
            _die(f"Unknown model_key: {mk}")
        runner_script = runners[mk]
        _ensure_file(runner_script, label="runner_script")

        # Resolve prompt/out paths per pipeline (mirrors Research_setup/Makefile naming).
        if pipeline in ("allergy", "allergy_omission"):
            prompts_by_model = {
                "llama31_groq": root / "prompts_llama31_groq.jsonl",
                "llama33_groq": root / "prompts_llama33_groq.jsonl",
                "qwen3_groq": root / "prompts_qwen3_groq.jsonl",
                "gpt41": root / "prompts_gpt41.jsonl",
                "gpt51": root / "prompts_gpt51.jsonl",
                "claude45": root / "prompts_claude45.jsonl",
            }
            out_by_model = {
                "llama31_groq": root / "generations_llama31_groq.jsonl",
                "llama33_groq": root / "generations_llama33_groq.jsonl",
                "qwen3_groq": root / "generations_qwen3_bedrock.jsonl",
                "gpt41": root / "generations_gpt41.jsonl",
                "gpt51": root / "generations_gpt51.jsonl",
                "claude45": root / "generations_claude_sonnet45.jsonl",
            }
        elif pipeline in ("medication", "medication_omission"):
            prompts_by_model = {
                "llama31_groq": root / "prompts_llama31_groq.jsonl",
                "llama33_groq": root / "prompts_llama33_groq.jsonl",
                "qwen3_groq": root / "prompts_qwen3_groq.jsonl",
                "gpt41": root / "prompts_gpt41.jsonl",
                "gpt51": root / "prompts_gpt51.jsonl",
                "claude45": root / "prompts_claude45.jsonl",
            }
            out_by_model = {
                "llama31_groq": root / "generations_llama31_groq.jsonl",
                "llama33_groq": root / "generations_llama33_groq.jsonl",
                "qwen3_groq": root / "generations_qwen3_bedrock.jsonl",
                "gpt41": root / "generations_gpt41.jsonl",
                "gpt51": root / "generations_gpt51.jsonl",
                "claude45": root / "generations_claude_sonnet45.jsonl",
            }
        elif pipeline in ("defense_allergy", "defense_allergy_omission"):
            prompts_by_model = {k: (root / k / "prompts.jsonl") for k in DEFAULT_MODEL_KEYS}
            out_by_model = {
                "llama31_groq": root / "llama31_groq" / "generations_llama31_groq.jsonl",
                "llama33_groq": root / "llama33_groq" / "generations_llama33_groq.jsonl",
                "qwen3_groq": root / "qwen3_groq" / "generations_qwen3_groq.jsonl",
                "gpt41": root / "gpt41" / "generations_gpt41.jsonl",
                "gpt51": root / "gpt51" / "generations_gpt51.jsonl",
                "claude45": root / "claude45" / "generations_claude45.jsonl",
            }
        elif pipeline in ("defense_medication", "defense_medication_omission"):
            prompts_by_model = {k: (root / k / "prompts.jsonl") for k in DEFAULT_MODEL_KEYS}
            out_by_model = {
                "llama31_groq": root / "llama31_groq" / "generations_llama31_groq.jsonl",
                "llama33_groq": root / "llama33_groq" / "generations_llama33_groq.jsonl",
                "qwen3_groq": root / "qwen3_groq" / "generations_qwen3_groq.jsonl",
                "gpt41": root / "gpt41" / "generations_gpt41.jsonl",
                "gpt51": root / "gpt51" / "generations_gpt51.jsonl",
                "claude45": root / "claude45" / "generations_claude45.jsonl",
            }
        else:
            _die(f"Unknown --pipeline: {pipeline}")

        prompts_jsonl = prompts_by_model[mk]
        out_jsonl = out_by_model[mk]
        _ensure_file(prompts_jsonl, label=f"prompts_jsonl[{mk}]")
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)

        # Runner-specific args.
        extra_args: List[str] = []
        if mk in ("llama31_groq", "llama33_groq", "qwen3_groq"):
            extra_args.extend(["--max_tokens", "0"])

        cmd = _runner_cmd(
            py=py,
            runner_script=runner_script,
            prompts_jsonl=prompts_jsonl,
            out_jsonl=out_jsonl,
            model=model_ids[mk],
            limit=limit,
            extra_args=extra_args,
        )

        jobs.append(
            Job(
                name=mk,
                cmd=cmd,
                cwd=repo_root,
                log_path=(logs_dir / f"{pipeline}_{mk}.log"),
                timeout_s=int(timeout_s),
            )
        )

    return jobs


def _run_job(job: Job) -> JobResult:
    t0 = time.time()
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with job.log_path.open("w", encoding="utf-8", newline="\n") as logf:
            logf.write(f"// AI-SUGGESTION: cmd={job.cmd}\n")
            logf.flush()
            p = subprocess.Popen(
                job.cmd,
                cwd=str(job.cwd),
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                rc = int(p.wait(timeout=job.timeout_s if job.timeout_s > 0 else None))
                return JobResult(
                    name=job.name,
                    exit_code=rc,
                    duration_s=(time.time() - t0),
                    log_path=job.log_path,
                    error="" if rc == 0 else f"exit_code={rc}",
                )
            except subprocess.TimeoutExpired:
                try:
                    p.kill()
                except Exception:
                    pass
                return JobResult(
                    name=job.name,
                    exit_code=124,
                    duration_s=(time.time() - t0),
                    log_path=job.log_path,
                    error=f"timeout_after_{job.timeout_s}s",
                )
    except Exception as e:
        return JobResult(
            name=job.name,
            exit_code=1,
            duration_s=(time.time() - t0),
            log_path=job.log_path,
            error=str(e),
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run 6 model runners in parallel (subprocess-per-model).")
    ap.add_argument(
        "--pipeline",
        required=True,
        choices=["allergy", "medication", "defense_allergy", "defense_medication"],
        help="Which pipeline naming convention to use for prompts/out file paths.",
    )
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Pipeline root dir. Examples: output/allergy_omission/results or output/defense_allergy_omission",
    )
    ap.add_argument(
        "--repo_root",
        type=Path,
        default=Path("."),
        help="Path to Research_setup/ (used as working dir for runner scripts).",
    )
    ap.add_argument("--limit", type=int, default=0, help="If >0, only run first N prompts in each model runner.")
    ap.add_argument("--timeout_s", type=int, default=0, help="If >0, kill any runner process after this many seconds.")
    ap.add_argument("--max_parallel", type=int, default=6, help="Max concurrent model processes.")
    ap.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODEL_KEYS),
        help="Comma-separated model keys to run (subset of the 6).",
    )

    # Default model ids (mirror Makefile defaults, but allow override).
    ap.add_argument("--llama31_model", type=str, default="llama-3.1-8b-instant")
    ap.add_argument("--llama33_model", type=str, default="llama-3.3-70b-versatile")
    ap.add_argument("--qwen3_model", type=str, default="qwen/qwen3-32b")
    ap.add_argument("--gpt41_model", type=str, default="gpt-4.1")
    ap.add_argument("--gpt51_model", type=str, default="gpt-5.1")
    ap.add_argument("--claude45_model", type=str, default="claude-sonnet-4-5")
    ap.add_argument("--py", type=str, default=sys.executable, help="Python executable to use for child runners.")

    args = ap.parse_args(argv)

    model_keys = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    if not model_keys:
        _die("--models produced an empty list.")
    for mk in model_keys:
        if mk not in DEFAULT_MODEL_KEYS:
            _die(f"--models includes unsupported key: {mk}. Supported: {list(DEFAULT_MODEL_KEYS)}")

    model_ids = {
        "llama31_groq": str(args.llama31_model),
        "llama33_groq": str(args.llama33_model),
        "qwen3_groq": str(args.qwen3_model),
        "gpt41": str(args.gpt41_model),
        "gpt51": str(args.gpt51_model),
        "claude45": str(args.claude45_model),
    }

    # Logs live under <root>/logs/ so they're next to the outputs.
    logs_dir = Path(args.root) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    jobs = _build_jobs(
        pipeline=str(args.pipeline),
        root=Path(args.root),
        repo_root=Path(args.repo_root),
        py=str(args.py),
        model_ids=model_ids,
        limit=int(args.limit),
        timeout_s=int(args.timeout_s),
        logs_dir=logs_dir,
        model_keys=model_keys,
    )

    max_parallel = max(1, int(args.max_parallel))
    t0 = time.time()
    results: List[JobResult] = []

    with ThreadPoolExecutor(max_workers=min(max_parallel, len(jobs))) as ex:
        futs = {ex.submit(_run_job, j): j for j in jobs}
        for fut in as_completed(futs):
            results.append(fut.result())

    # Print a concise summary (stderr for visibility).
    results_sorted = sorted(results, key=lambda r: r.name)
    ok = [r for r in results_sorted if r.exit_code == 0]
    bad = [r for r in results_sorted if r.exit_code != 0]
    elapsed = time.time() - t0
    print(f"[parallel] pipeline={args.pipeline} ok={len(ok)}/{len(results_sorted)} elapsed_s={elapsed:.1f}", file=sys.stderr)
    for r in results_sorted:
        status = "OK" if r.exit_code == 0 else f"FAIL({r.error})"
        print(f"- {r.name}: {status} time_s={r.duration_s:.1f} log={r.log_path}", file=sys.stderr)

    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())



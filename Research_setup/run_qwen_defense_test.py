"""
run_qwen_defense_test.py
// AI-Generated Code Header
// **Intent:** Convenience script to run ONLY the Qwen defense pipeline end-to-end:
//            prompts.jsonl -> generations_qwen3_groq.jsonl -> scoring summary.json.
//            Works for both allergy and medication defense tasks.
// **Optimization:** Uses existing experiment + runner scripts (no duplicated logic).
// **Safety:** Validates paths and exits non-zero on failures; prints the produced summary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


RESEARCH_SETUP_DIR = Path(__file__).resolve().parent


def _die(msg: str, code: int = 2) -> "None":
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _load_summary(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("summary.json must be a list")
    clean = next((x for x in data if isinstance(x, dict) and x.get("condition") == "clean"), None)
    attack = next((x for x in data if isinstance(x, dict) and x.get("condition") == "attack"), None)
    return {"clean": clean, "attack": attack, "raw": data}


def _run(cmd: list[str]) -> None:
    # AI-SUGGESTION: Use the current python executable for consistency with venv.
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(RESEARCH_SETUP_DIR), check=False)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run ONLY Qwen defense pipeline (prompts -> generations -> scoring).")
    ap.add_argument("--task", choices=["allergy", "medication"], required=True)
    ap.add_argument("--max_patients", type=int, default=1, help="Default 1 (smoke test). Use 1000 for full run.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=2, help="Runner limit. Use 0 to run all prompts.")
    ap.add_argument("--qwen_model", type=str, default="qwen/qwen3-32b", help="Groq model id for Qwen.")
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("output"),
        help="Base output dir under Research_setup/. Final dir will be <out_dir>/defense_<task>_omission/qwen3_groq",
    )
    args = ap.parse_args(argv)

    task = str(args.task)
    subset_dir = (
        RESEARCH_SETUP_DIR / "data/subset/data_subset_allergyomissionsubset"
        if task == "allergy"
        else RESEARCH_SETUP_DIR / "data/subset/data_subset_medicationomissionsubset"
    )
    if not subset_dir.exists():
        _die(f"Subset dir not found: {subset_dir}")

    patient_ids_txt = subset_dir / "patient_ids.txt"
    patients_csv = subset_dir / "patients_subset.csv"
    items_csv = subset_dir / ("allergies_subset.csv" if task == "allergy" else "medications_subset.csv")
    for p in (patient_ids_txt, patients_csv, items_csv):
        if not p.exists():
            _die(f"Required input missing: {p}")

    out_dir = (RESEARCH_SETUP_DIR / args.out_dir / f"defense_{task}_omission" / "qwen3_groq").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts_path = out_dir / "prompts.jsonl"
    gens_path = out_dir / "generations_qwen3_groq.jsonl"
    summary_path = out_dir / "summary.json"

    # 1) Prompts
    _run(
        [
            sys.executable,
            "main/src/defense_omission_experiment.py",
            "--task",
            task,
            "--model_key",
            "qwen3_groq",
            "--patient_ids_txt",
            str(patient_ids_txt),
            "--patients_csv",
            str(patients_csv),
            "--items_csv",
            str(items_csv),
            "--max_patients",
            str(int(args.max_patients)),
            "--seed",
            str(int(args.seed)),
            "--out_dir",
            str(out_dir),
            "--prompts_out_jsonl",
            str(prompts_path),
        ]
    )

    # 2) Run Qwen runner
    _run(
        [
            sys.executable,
            "LLM Model/Qwen 3/qwen3_bedrock_runner.py",
            "--model",
            str(args.qwen_model),
            "--prompts_jsonl",
            str(prompts_path),
            "--out_jsonl",
            str(gens_path),
            "--max_tokens",
            "0",
            "--limit",
            str(int(args.limit)),
        ]
    )

    # 3) Score
    _run(
        [
            sys.executable,
            "main/src/defense_omission_experiment.py",
            "--task",
            task,
            "--model_key",
            "qwen3_groq",
            "--patient_ids_txt",
            str(patient_ids_txt),
            "--patients_csv",
            str(patients_csv),
            "--items_csv",
            str(items_csv),
            "--max_patients",
            str(int(args.max_patients)),
            "--seed",
            str(int(args.seed)),
            "--out_dir",
            str(out_dir),
            "--generations_jsonl",
            str(gens_path),
        ]
    )

    if not summary_path.exists():
        _die(f"Expected summary not found: {summary_path}")

    summ = _load_summary(summary_path)
    atk = summ.get("attack") or {}
    asr = atk.get("attack_success_rate")
    dsr = (1.0 - float(asr)) if asr not in (None, "") else ""
    print(f"Summary path: {summary_path}")
    print(f"Attack success rate: {asr}")
    print(f"Defense success rate: {dsr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
compute_defense_success_rates.py
// AI-Generated Code Header
// **Intent:** Aggregate defense success metrics across models for a defense run folder (defense_allergy_omission or defense_medication_omission).
// **Optimization:** Prefer summary.json when available; fallback to streaming results.csv.
// **Safety:** Validates inputs, tolerates partial runs (missing files), and writes outputs atomically.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _die(msg: str, code: int = 2) -> "None":
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Agg:
    model_dir: str
    model_name: str
    n_total_clean: int
    n_eval_clean: int
    n_missing_clean: int
    trusted_correct_rate_clean: Optional[float]
    parse_error_rate_clean: Optional[float]
    n_total_attack: int
    n_eval_attack: int
    n_missing_attack: int
    trusted_correct_rate_attack: Optional[float]
    parse_error_rate_attack: Optional[float]
    attack_success_rate: Optional[float]

    @property
    def defense_success_rate(self) -> Optional[float]:
        # AI-SUGGESTION: In this repo, "attack_success_rate" means attacker succeeded (defense failed).
        # Defense success rate is the complement.
        return (1.0 - self.attack_success_rate) if self.attack_success_rate is not None else None


def _summ_from_summary_json(summary_json: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data = _load_json(summary_json)
    if not isinstance(data, list):
        raise ValueError(f"summary.json must be a list: {summary_json}")
    clean = next((x for x in data if isinstance(x, dict) and x.get("condition") == "clean"), None)
    attack = next((x for x in data if isinstance(x, dict) and x.get("condition") == "attack"), None)
    if not isinstance(clean, dict) or not isinstance(attack, dict):
        raise ValueError(f"summary.json missing clean/attack objects: {summary_json}")
    return clean, attack


def _summ_from_results_csv(results_csv: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Fallback if summary.json is missing.
    """
    def init() -> Dict[str, Any]:
        return {
            "n_total": 0,
            "n_evaluated": 0,
            "n_missing_generations": 0,
            "trusted_correct_hits": 0,
            "parse_error_hits": 0,
            "attack_success_hits": 0,
        }

    clean = init()
    attack = init()
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            cond = (row.get("condition") or "").strip().lower()
            if cond not in ("clean", "attack"):
                continue
            bucket = clean if cond == "clean" else attack
            bucket["n_total"] += 1
            parse_error = (row.get("parse_error") or "").strip()
            if parse_error == "missing_generation":
                bucket["n_missing_generations"] += 1
                continue
            bucket["n_evaluated"] += 1
            if parse_error:
                bucket["parse_error_hits"] += 1
            if (row.get("trusted_correct") or "").strip().lower() == "true":
                bucket["trusted_correct_hits"] += 1
            if cond == "attack" and (row.get("attack_success") or "").strip().lower() == "true":
                bucket["attack_success_hits"] += 1

    def finalize(bucket: Dict[str, Any], *, is_attack: bool) -> Dict[str, Any]:
        n_total = int(bucket["n_total"])
        n_eval = int(bucket["n_evaluated"])
        n_missing = int(bucket["n_missing_generations"])
        trusted_correct_rate = (bucket["trusted_correct_hits"] / n_eval) if n_eval else ""
        parse_error_rate = (bucket["parse_error_hits"] / n_eval) if n_eval else ""
        if is_attack:
            attack_success_rate = (bucket["attack_success_hits"] / n_eval) if n_eval else ""
        else:
            attack_success_rate = ""
        return {
            "n_total": n_total,
            "n_evaluated": n_eval,
            "n_missing_generations": n_missing,
            "trusted_correct_rate": trusted_correct_rate,
            "parse_error_rate": parse_error_rate,
            "attack_success_rate": attack_success_rate,
        }

    return finalize(clean, is_attack=False), finalize(attack, is_attack=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate defense success rates across models in a defense run folder.")
    ap.add_argument(
        "--defense_dir",
        type=Path,
        default=Path("Research_setup/final/defense_allergy_omission"),
        help="Folder containing model subfolders with summary.json/results.csv.",
    )
    ap.add_argument(
        "--out_csv",
        type=Path,
        default=None,
        help="Optional override. Defaults to <defense_dir>/defense_success_rates.csv",
    )
    ap.add_argument(
        "--out_json",
        type=Path,
        default=None,
        help="Optional override. Defaults to <defense_dir>/defense_success_rates.json",
    )
    args = ap.parse_args(argv)

    defense_dir: Path = args.defense_dir
    if not defense_dir.exists():
        _die(f"defense_dir not found: {defense_dir}")

    model_dirs = sorted([p for p in defense_dir.iterdir() if p.is_dir()])
    if not model_dirs:
        _die(f"No model subfolders found in: {defense_dir}")

    out_csv = args.out_csv or (defense_dir / "defense_success_rates.csv")
    out_json = args.out_json or (defense_dir / "defense_success_rates.json")

    out_rows: List[Dict[str, Any]] = []
    for md in model_dirs:
        model_name = md.name
        summary_json = md / "summary.json"
        results_csv = md / "results.csv"

        if summary_json.exists():
            clean_s, attack_s = _summ_from_summary_json(summary_json)
        elif results_csv.exists():
            clean_s, attack_s = _summ_from_results_csv(results_csv)
        else:
            # AI-SUGGESTION: tolerate partial runs
            out_rows.append(
                {
                    "model_name": model_name,
                    "model_dir": md.as_posix(),
                    "error": "missing_summary_and_results",
                }
            )
            continue

        agg = Agg(
            model_dir=md.as_posix(),
            model_name=model_name,
            n_total_clean=int(clean_s.get("n_total") or 0),
            n_eval_clean=int(clean_s.get("n_evaluated") or 0),
            n_missing_clean=int(clean_s.get("n_missing_generations") or 0),
            trusted_correct_rate_clean=_safe_float(clean_s.get("trusted_correct_rate")),
            parse_error_rate_clean=_safe_float(clean_s.get("parse_error_rate")),
            n_total_attack=int(attack_s.get("n_total") or 0),
            n_eval_attack=int(attack_s.get("n_evaluated") or 0),
            n_missing_attack=int(attack_s.get("n_missing_generations") or 0),
            trusted_correct_rate_attack=_safe_float(attack_s.get("trusted_correct_rate")),
            parse_error_rate_attack=_safe_float(attack_s.get("parse_error_rate")),
            attack_success_rate=_safe_float(attack_s.get("attack_success_rate")),
        )

        out_rows.append(
            {
                "model_name": agg.model_name,
                "model_dir": agg.model_dir,
                "clean_n_total": agg.n_total_clean,
                "clean_n_evaluated": agg.n_eval_clean,
                "clean_n_missing_generations": agg.n_missing_clean,
                "clean_trusted_correct_rate": agg.trusted_correct_rate_clean if agg.trusted_correct_rate_clean is not None else "",
                "clean_parse_error_rate": agg.parse_error_rate_clean if agg.parse_error_rate_clean is not None else "",
                "attack_n_total": agg.n_total_attack,
                "attack_n_evaluated": agg.n_eval_attack,
                "attack_n_missing_generations": agg.n_missing_attack,
                "attack_trusted_correct_rate": agg.trusted_correct_rate_attack if agg.trusted_correct_rate_attack is not None else "",
                "attack_parse_error_rate": agg.parse_error_rate_attack if agg.parse_error_rate_attack is not None else "",
                "attack_success_rate": agg.attack_success_rate if agg.attack_success_rate is not None else "",
                "defense_success_rate": agg.defense_success_rate if agg.defense_success_rate is not None else "",
                "error": "",
            }
        )

    fieldnames = [
        "model_name",
        "model_dir",
        "clean_n_total",
        "clean_n_evaluated",
        "clean_n_missing_generations",
        "clean_trusted_correct_rate",
        "clean_parse_error_rate",
        "attack_n_total",
        "attack_n_evaluated",
        "attack_n_missing_generations",
        "attack_trusted_correct_rate",
        "attack_parse_error_rate",
        "attack_success_rate",
        "defense_success_rate",
        "error",
    ]

    _atomic_write_csv(out_csv, out_rows, fieldnames=fieldnames)
    _atomic_write_text(out_json, json.dumps(out_rows, ensure_ascii=False, indent=2) + "\n")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



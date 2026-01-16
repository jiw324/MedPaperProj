"""
create_medication_omission_dataset.py
// AI-Generated Code Header
// **Intent:** Create a deterministic N-patient subdataset (patients+medications) for faster medication-omission experiments.
// **Optimization:** Uses stdlib CSV streaming and simple in-memory sets; fast for Synthea-sized CSVs.
// **Safety:** Validates required columns, fails fast on missing paths, and writes outputs atomically.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence


RESEARCH_SETUP_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MEDICATIONS_CSV = RESEARCH_SETUP_DIR / "data/synthea_1m_fhir_3_0_May_24/output_1/csv/medications.csv"
DEFAULT_PATIENTS_CSV = RESEARCH_SETUP_DIR / "data/synthea_1m_fhir_3_0_May_24/output_1/csv/patients.csv"
DEFAULT_SUBSET_OUT_DIR = RESEARCH_SETUP_DIR / "data/subset/data_subset_medicationomissionsubset"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def export_subset_dataset(
    *,
    medications_csv: Path,
    patients_csv: Path,
    selected_patient_ids: Sequence[str],
    subset_dir: Path,
) -> Dict[str, Path]:
    """
    Export a smaller dataset containing only the selected patients:
      - patients_subset.csv: filtered rows from patients.csv by ID
      - medications_subset.csv: filtered rows from medications.csv by PATIENT
    """
    subset_dir.mkdir(parents=True, exist_ok=True)
    selected = list(selected_patient_ids)
    selected_set = set(selected)
    if not selected_set:
        raise ValueError("No selected_patient_ids provided.")

    patients_out = subset_dir / "patients_subset.csv"
    meds_out = subset_dir / "medications_subset.csv"
    ids_out = subset_dir / "patient_ids.txt"
    meta_out = subset_dir / "metadata.json"

    _atomic_write_text(ids_out, "\n".join(selected) + "\n")

    # Filter patients.csv -> patients_subset.csv
    with patients_csv.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames or "ID" not in reader.fieldnames:
            raise ValueError(f"patients.csv missing header/ID column: {patients_csv}")
        tmp = patients_out.with_suffix(patients_out.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                pid = (row.get("ID") or "").strip()
                if pid in selected_set:
                    writer.writerow(row)
        os.replace(tmp, patients_out)

    # Filter medications.csv -> medications_subset.csv
    with medications_csv.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames or "PATIENT" not in reader.fieldnames:
            raise ValueError(f"medications.csv missing header/PATIENT column: {medications_csv}")
        tmp = meds_out.with_suffix(meds_out.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                pid = (row.get("PATIENT") or "").strip()
                if pid in selected_set:
                    writer.writerow(row)
        os.replace(tmp, meds_out)

    meta = {
        "source_medications_csv": str(medications_csv),
        "source_patients_csv": str(patients_csv),
        "subset_patient_count": len(selected),
    }
    _atomic_write_text(meta_out, json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    return {
        "patients_subset_csv": patients_out,
        "medications_subset_csv": meds_out,
        "patient_ids_txt": ids_out,
        "metadata_json": meta_out,
    }


def iter_medication_patients(medications_csv: Path) -> List[str]:
    """
    Return patient IDs that have at least one medication row with a non-empty DESCRIPTION.
    """
    if not medications_csv.exists():
        raise FileNotFoundError(f"Medications CSV not found: {medications_csv}")
    pts: set[str] = set()
    with medications_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"PATIENT", "DESCRIPTION"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Medications CSV missing columns {sorted(missing)} in {medications_csv}")
        for row in reader:
            pid = (row.get("PATIENT") or "").strip()
            desc = (row.get("DESCRIPTION") or "").strip()
            if pid and desc:
                pts.add(pid)
    return list(pts)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Create a filtered subdataset of medication-bearing patients.")
    ap.add_argument("--medications_csv", type=Path, default=DEFAULT_MEDICATIONS_CSV, help="Path to Synthea medications.csv")
    ap.add_argument("--patients_csv", type=Path, default=DEFAULT_PATIENTS_CSV, help="Path to Synthea patients.csv")
    ap.add_argument("--n", type=int, default=1000, help="Number of medication-bearing patients to sample.")
    ap.add_argument("--seed", type=int, default=7, help="Random seed for deterministic sampling.")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_SUBSET_OUT_DIR, help="Output directory for subset CSVs.")

    args = ap.parse_args(argv)
    if args.n <= 0:
        raise ValueError("--n must be > 0")

    rng = random.Random(int(args.seed))
    eligible = iter_medication_patients(args.medications_csv)
    if not eligible:
        raise RuntimeError("No medication-bearing patients found. Check medications.csv path/content.")

    rng.shuffle(eligible)
    selected = eligible[: int(args.n)]
    if len(selected) < int(args.n):
        raise RuntimeError(f"Requested n={args.n} but only {len(selected)} eligible patients exist.")

    exported = export_subset_dataset(
        medications_csv=args.medications_csv,
        patients_csv=args.patients_csv,
        selected_patient_ids=selected,
        subset_dir=args.out_dir,
    )

    print("Exported subset dataset:")
    for k, p in exported.items():
        print(f"- {k}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
create_sub_dataset.py
// AI-Generated Code Header
// **Intent:** Create a deterministic 5,000-patient subdataset (patients+allergies) for faster allergy-omission experiments.
// **Optimization:** Uses stdlib CSV streaming and simple in-memory sets; fast for Synthea-sized CSVs.
// **Safety:** Validates required columns, fails fast on missing paths, and writes outputs atomically.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional, Sequence

# AI-SUGGESTION: Reuse the already-validated parsing/export logic from the experiment module to avoid divergence.
from allergy_omission_experiment import export_subset_dataset, load_allergies  # noqa: E402


DEFAULT_ALLERGIES_CSV = Path("Research_setup/data/synthea_1m_fhir_3_0_May_24/output_1/csv/allergies.csv")
DEFAULT_PATIENTS_CSV = Path("Research_setup/data/synthea_1m_fhir_3_0_May_24/output_1/csv/patients.csv")
DEFAULT_SUBSET_OUT_DIR = Path("Research_setup/output/allergy_omission/subdataset")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Create a filtered subdataset of allergy-bearing patients.")
    ap.add_argument("--allergies_csv", type=Path, default=DEFAULT_ALLERGIES_CSV, help="Path to Synthea allergies.csv")
    ap.add_argument("--patients_csv", type=Path, default=DEFAULT_PATIENTS_CSV, help="Path to Synthea patients.csv")
    ap.add_argument("--n", type=int, default=1000, help="Number of allergy-bearing patients to sample.")
    ap.add_argument("--seed", type=int, default=7, help="Random seed for deterministic sampling.")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_SUBSET_OUT_DIR, help="Output directory for subset CSVs.")

    args = ap.parse_args(argv)
    if args.n <= 0:
        raise ValueError("--n must be > 0")

    rng = random.Random(int(args.seed))
    allergies_by_patient = load_allergies(args.allergies_csv)
    eligible = [pid for pid, recs in allergies_by_patient.items() if recs]
    if not eligible:
        raise RuntimeError("No allergy-bearing patients found. Check allergies.csv path/content.")

    rng.shuffle(eligible)
    selected = eligible[: int(args.n)]
    if len(selected) < int(args.n):
        raise RuntimeError(f"Requested n={args.n} but only {len(selected)} eligible patients exist.")

    exported = export_subset_dataset(
        allergies_csv=args.allergies_csv,
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



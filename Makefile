# Makefile
# // AI-SUGGESTION: Uses Python for directory creation so this works on Windows (PowerShell/cmd) with GNU Make installed.

PY ?= python

OUT_ROOT := Research_setup/output/allergy_omission
SUBSET_DIR := $(OUT_ROOT)/subdataset
RESULT_DIR := $(OUT_ROOT)/results

ALLERGIES_SRC := Research_setup/data/synthea_1m_fhir_3_0_May_24/output_1/csv/allergies.csv
PATIENTS_SRC := Research_setup/data/synthea_1m_fhir_3_0_May_24/output_1/csv/patients.csv

N ?= 5000
SEED ?= 7
MAX_PATIENTS ?= 5000

.PHONY: all prepare subdataset experiment clean
.PHONY: prompts experiment_all_models

# Full end-to-end run (includes running all 4 models).
all: experiment_all_models

prepare:
	@$(PY) -c "import os; os.makedirs(r'$(SUBSET_DIR)', exist_ok=True); os.makedirs(r'$(RESULT_DIR)', exist_ok=True)"

subdataset: prepare
	@$(PY) Research_setup/src/create_sub_dataset.py --allergies_csv $(ALLERGIES_SRC) --patients_csv $(PATIENTS_SRC) --n $(N) --seed $(SEED) --out_dir $(SUBSET_DIR)

# Default: build prompts and template results WITHOUT running any model.
prompts: subdataset
	@$(PY) Research_setup/src/allergy_omission_experiment.py --allergies_csv $(SUBSET_DIR)/allergies_subset.csv --patients_csv $(SUBSET_DIR)/patients_subset.csv --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir $(RESULT_DIR)

experiment: prompts

# Run all 4 models (heavy). Use this in your inference environment.
experiment_all_models: subdataset
	@$(PY) Research_setup/src/allergy_omission_experiment.py --allergies_csv $(SUBSET_DIR)/allergies_subset.csv --patients_csv $(SUBSET_DIR)/patients_subset.csv --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir $(RESULT_DIR) --run_all_models

clean:
	@$(PY) -c "import shutil; shutil.rmtree(r'$(OUT_ROOT)', ignore_errors=True)"



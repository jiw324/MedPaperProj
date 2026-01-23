# Research_setup/Makefile
# // AI-SUGGESTION: Run this Makefile from the Research_setup folder: `cd Research_setup && make allergy`

PY ?= python

# -----------------------------
# Paths (relative to Research_setup/)
# -----------------------------

OUT_ROOT := output/allergy_omission
RESULT_DIR := $(OUT_ROOT)/results

SUBSET_DIR := data/subset/data_subset_allergyomissionsubset
MED_SUBSET_DIR := data/subset/data_subset_medicationomissionsubset

ALLERGIES_SRC := data/synthea_1m_fhir_3_0_May_24/output_1/csv/allergies.csv
PATIENTS_SRC := data/synthea_1m_fhir_3_0_May_24/output_1/csv/patients.csv
MEDICATIONS_SRC := data/synthea_1m_fhir_3_0_May_24/output_1/csv/medications.csv

# -----------------------------
# Experiment parameters
# -----------------------------

N ?= 1000
SEED ?= 7
MAX_PATIENTS ?= 1000
MED_MAX_PATIENTS ?= 1000

# Shared runner limit. 0 = run all prompts.
RUN_LIMIT ?= 0

# Default model ids (override at make-time if desired).
LLAMA31_MODEL ?= llama-3.1-8b-instant
LLAMA33_MODEL ?= llama-3.3-70b-versatile
QWEN3_MODEL ?= qwen/qwen3-32b
GPT41_MODEL ?= gpt-4.1
GPT51_MODEL ?= gpt-5.1
CLAUDE45_MODEL ?= claude-sonnet-4-5

.PHONY: prepare_allergy allergy_subdataset allergy_prompts
.PHONY: allergy_prompts_llama31 allergy_prompts_llama33 allergy_prompts_qwen3 allergy_prompts_gpt41 allergy_prompts_gpt51 allergy_prompts_claude45 allergy_prompts_all
.PHONY: allergy_llama31 allergy_llama33 allergy_qwen3 allergy_gpt41 allergy_gpt51 allergy_claude45 allergy_models
.PHONY: allergy_models_parallel allergy_parallel
.PHONY: allergy_score_llama31 allergy_score_llama33 allergy_score_qwen3 allergy_score_gpt41 allergy_score_gpt51 allergy_score_claude45 allergy_score_all
.PHONY: allergy allergy_test clean
.PHONY: prepare_medication medication_subdataset medication_prompts
.PHONY: medication_prompts_llama31 medication_prompts_llama33 medication_prompts_qwen3 medication_prompts_gpt41 medication_prompts_gpt51 medication_prompts_claude45 medication_prompts_all
.PHONY: medication_llama31 medication_llama33 medication_qwen3 medication_gpt41 medication_gpt51 medication_claude45 medication_models
.PHONY: medication_models_parallel medication_parallel
.PHONY: medication_score_llama31 medication_score_llama33 medication_score_qwen3 medication_score_gpt41 medication_score_gpt51 medication_score_claude45 medication_score_all
.PHONY: medication medication_test
.PHONY: defense_allergy_prompts_llama31 defense_allergy_prompts_llama33 defense_allergy_prompts_qwen3 defense_allergy_prompts_gpt41 defense_allergy_prompts_gpt51 defense_allergy_prompts_claude45 defense_allergy_prompts_all
.PHONY: defense_allergy_llama31 defense_allergy_llama33 defense_allergy_qwen3 defense_allergy_gpt41 defense_allergy_gpt51 defense_allergy_claude45 defense_allergy_models
.PHONY: defense_allergy_models_parallel defense_allergy_parallel
.PHONY: defense_allergy_score_llama31 defense_allergy_score_llama33 defense_allergy_score_qwen3 defense_allergy_score_gpt41 defense_allergy_score_gpt51 defense_allergy_score_claude45 defense_allergy_score_all
.PHONY: defense_allergy defense_allergy_test
.PHONY: defense_allergy_qwen
.PHONY: defense_medication_prompts_llama31 defense_medication_prompts_llama33 defense_medication_prompts_qwen3 defense_medication_prompts_gpt41 defense_medication_prompts_gpt51 defense_medication_prompts_claude45 defense_medication_prompts_all
.PHONY: defense_medication_llama31 defense_medication_llama33 defense_medication_qwen3 defense_medication_gpt41 defense_medication_gpt51 defense_medication_claude45 defense_medication_models
.PHONY: defense_medication_models_parallel defense_medication_parallel
.PHONY: defense_medication_score_llama31 defense_medication_score_llama33 defense_medication_score_qwen3 defense_medication_score_gpt41 defense_medication_score_gpt51 defense_medication_score_claude45 defense_medication_score_all
.PHONY: defense_medication defense_medication_test
.PHONY: defense_medication_qwen

prepare_allergy:
	@$(PY) -c "import os; os.makedirs(r'$(SUBSET_DIR)', exist_ok=True); os.makedirs(r'$(RESULT_DIR)', exist_ok=True)"

allergy_subdataset: prepare_allergy
	@$(PY) data/src/create_allergy_omission_dataset.py --allergies_csv "$(ALLERGIES_SRC)" --patients_csv "$(PATIENTS_SRC)" --n $(N) --seed $(SEED) --out_dir "$(SUBSET_DIR)"

allergy_prompts: allergy_subdataset
	@$(PY) main/src/allergy_omission_experiment.py --allergies_csv "$(SUBSET_DIR)/allergies_subset.csv" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(RESULT_DIR)" --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt"

# AI-SUGGESTION: Model-specific prompt files so each model sees a tuned variant.
allergy_prompts_llama31: allergy_subdataset
	@$(PY) main/src/allergy_omission_experiment.py --allergies_csv "$(SUBSET_DIR)/allergies_subset.csv" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(RESULT_DIR)" --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --model_key llama31_groq --prompts_out_jsonl "$(RESULT_DIR)/prompts_llama31_groq.jsonl"

allergy_prompts_llama33: allergy_subdataset
	@$(PY) main/src/allergy_omission_experiment.py --allergies_csv "$(SUBSET_DIR)/allergies_subset.csv" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(RESULT_DIR)" --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --model_key llama33_groq --prompts_out_jsonl "$(RESULT_DIR)/prompts_llama33_groq.jsonl"

allergy_prompts_qwen3: allergy_subdataset
	@$(PY) main/src/allergy_omission_experiment.py --allergies_csv "$(SUBSET_DIR)/allergies_subset.csv" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(RESULT_DIR)" --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --model_key qwen3_groq --prompts_out_jsonl "$(RESULT_DIR)/prompts_qwen3_groq.jsonl"

allergy_prompts_gpt41: allergy_subdataset
	@$(PY) main/src/allergy_omission_experiment.py --allergies_csv "$(SUBSET_DIR)/allergies_subset.csv" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(RESULT_DIR)" --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --model_key gpt41 --prompts_out_jsonl "$(RESULT_DIR)/prompts_gpt41.jsonl"

allergy_prompts_gpt51: allergy_subdataset
	@$(PY) main/src/allergy_omission_experiment.py --allergies_csv "$(SUBSET_DIR)/allergies_subset.csv" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(RESULT_DIR)" --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --model_key gpt51 --prompts_out_jsonl "$(RESULT_DIR)/prompts_gpt51.jsonl"

allergy_prompts_claude45: allergy_subdataset
	@$(PY) main/src/allergy_omission_experiment.py --allergies_csv "$(SUBSET_DIR)/allergies_subset.csv" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(RESULT_DIR)" --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --model_key claude45 --prompts_out_jsonl "$(RESULT_DIR)/prompts_claude45.jsonl"

allergy_prompts_all: allergy_prompts_llama31 allergy_prompts_llama33 allergy_prompts_qwen3 allergy_prompts_gpt41 allergy_prompts_gpt51 allergy_prompts_claude45

# -----------------------------
# Model runs (prompts.jsonl -> generations_*.jsonl)
# -----------------------------

allergy_llama31: allergy_prompts_llama31
	@$(PY) "LLM Model/Llama 3.1/llama31_groq_runner.py" --model "$(LLAMA31_MODEL)" --prompts_jsonl "$(RESULT_DIR)/prompts_llama31_groq.jsonl" --out_jsonl "$(RESULT_DIR)/generations_llama31_groq.jsonl" --max_tokens 0 --limit $(RUN_LIMIT)

allergy_llama33: allergy_prompts_llama33
	@$(PY) "LLM Model/Llama 3.3/llama33_groq_runner.py" --model "$(LLAMA33_MODEL)" --prompts_jsonl "$(RESULT_DIR)/prompts_llama33_groq.jsonl" --out_jsonl "$(RESULT_DIR)/generations_llama33_groq.jsonl" --max_tokens 0 --limit $(RUN_LIMIT)

allergy_qwen3: allergy_prompts_qwen3
	@$(PY) "LLM Model/Qwen 3/qwen3_bedrock_runner.py" --model "$(QWEN3_MODEL)" --prompts_jsonl "$(RESULT_DIR)/prompts_qwen3_groq.jsonl" --out_jsonl "$(RESULT_DIR)/generations_qwen3_bedrock.jsonl" --max_tokens 0 --limit $(RUN_LIMIT)

allergy_gpt41: allergy_prompts_gpt41
	@$(PY) "LLM Model/GPT-4.1/gpt41_api.py" --model "$(GPT41_MODEL)" --prompts_jsonl "$(RESULT_DIR)/prompts_gpt41.jsonl" --out_jsonl "$(RESULT_DIR)/generations_gpt41.jsonl" --limit $(RUN_LIMIT)

allergy_gpt51: allergy_prompts_gpt51
	@$(PY) "LLM Model/GPT-5.1/gpt51_api.py" --model "$(GPT51_MODEL)" --prompts_jsonl "$(RESULT_DIR)/prompts_gpt51.jsonl" --out_jsonl "$(RESULT_DIR)/generations_gpt51.jsonl" --limit $(RUN_LIMIT)

allergy_claude45: allergy_prompts_claude45
	@$(PY) "LLM Model/Claude 4.5/claude45_api.py" --model "$(CLAUDE45_MODEL)" --prompts_jsonl "$(RESULT_DIR)/prompts_claude45.jsonl" --out_jsonl "$(RESULT_DIR)/generations_claude_sonnet45.jsonl" --limit $(RUN_LIMIT)

allergy_models: allergy_llama31 allergy_llama33 allergy_qwen3 allergy_gpt41 allergy_gpt51 allergy_claude45

# AI-SUGGESTION: Run the 6 model runners concurrently (subprocess-per-model), instead of sequential make prerequisites.
# Requires that the model-specific prompt files already exist (allergy_prompts_all).
allergy_models_parallel: allergy_prompts_all
	@$(PY) main/src/parallel_model_runs.py --pipeline allergy --root "$(RESULT_DIR)" --limit $(RUN_LIMIT) --max_parallel 6

# -----------------------------
# Scoring (generations_*.jsonl -> scored results)
# We write scored outputs under output/allergy_omission/scored/<model>/
# -----------------------------

SCORED_ROOT := $(OUT_ROOT)/scored

allergy_score_llama31:
	@$(PY) -c "import os; os.makedirs(r'$(SCORED_ROOT)/llama31_groq', exist_ok=True)"
	@$(PY) main/src/allergy_omission_experiment.py --allergies_csv "$(SUBSET_DIR)/allergies_subset.csv" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --max_patients $(MAX_PATIENTS) --seed $(SEED) --model_key llama31_groq --out_dir "$(SCORED_ROOT)/llama31_groq" --generations_jsonl "$(RESULT_DIR)/generations_llama31_groq.jsonl"

allergy_score_llama33:
	@$(PY) -c "import os; os.makedirs(r'$(SCORED_ROOT)/llama33_groq', exist_ok=True)"
	@$(PY) main/src/allergy_omission_experiment.py --allergies_csv "$(SUBSET_DIR)/allergies_subset.csv" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --max_patients $(MAX_PATIENTS) --seed $(SEED) --model_key llama33_groq --out_dir "$(SCORED_ROOT)/llama33_groq" --generations_jsonl "$(RESULT_DIR)/generations_llama33_groq.jsonl"

allergy_score_qwen3:
	@$(PY) -c "import os; os.makedirs(r'$(SCORED_ROOT)/qwen3_groq', exist_ok=True)"
	@$(PY) main/src/allergy_omission_experiment.py --allergies_csv "$(SUBSET_DIR)/allergies_subset.csv" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --max_patients $(MAX_PATIENTS) --seed $(SEED) --model_key qwen3_groq --out_dir "$(SCORED_ROOT)/qwen3_groq" --generations_jsonl "$(RESULT_DIR)/generations_qwen3_bedrock.jsonl"

allergy_score_gpt41:
	@$(PY) -c "import os; os.makedirs(r'$(SCORED_ROOT)/gpt41', exist_ok=True)"
	@$(PY) main/src/allergy_omission_experiment.py --allergies_csv "$(SUBSET_DIR)/allergies_subset.csv" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --max_patients $(MAX_PATIENTS) --seed $(SEED) --model_key gpt41 --out_dir "$(SCORED_ROOT)/gpt41" --generations_jsonl "$(RESULT_DIR)/generations_gpt41.jsonl"

allergy_score_gpt51:
	@$(PY) -c "import os; os.makedirs(r'$(SCORED_ROOT)/gpt51', exist_ok=True)"
	@$(PY) main/src/allergy_omission_experiment.py --allergies_csv "$(SUBSET_DIR)/allergies_subset.csv" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --max_patients $(MAX_PATIENTS) --seed $(SEED) --model_key gpt51 --out_dir "$(SCORED_ROOT)/gpt51" --generations_jsonl "$(RESULT_DIR)/generations_gpt51.jsonl"

allergy_score_claude45:
	@$(PY) -c "import os; os.makedirs(r'$(SCORED_ROOT)/claude45', exist_ok=True)"
	@$(PY) main/src/allergy_omission_experiment.py --allergies_csv "$(SUBSET_DIR)/allergies_subset.csv" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --max_patients $(MAX_PATIENTS) --seed $(SEED) --model_key claude45 --out_dir "$(SCORED_ROOT)/claude45" --generations_jsonl "$(RESULT_DIR)/generations_claude_sonnet45.jsonl"

allergy_score_all: allergy_score_llama31 allergy_score_llama33 allergy_score_qwen3 allergy_score_gpt41 allergy_score_gpt51 allergy_score_claude45

# Full pipeline
allergy: allergy_models allergy_score_all

# AI-SUGGESTION: Full pipeline with parallel model execution.
allergy_parallel: allergy_models_parallel allergy_score_all

# Smoke-test pipeline (limit 2 prompts)
allergy_test: RUN_LIMIT=2
allergy_test: allergy

# -----------------------------
# Medication omission pipeline
# -----------------------------

MED_OUT_ROOT := output/medication_omission
MED_RESULT_DIR := $(MED_OUT_ROOT)/results
MED_SCORED_ROOT := $(MED_OUT_ROOT)/scored

# -----------------------------
# Defense pipelines (general to allergy + medication)
# Uses main/src/defense_omission_experiment.py
# -----------------------------

DEF_ALLERGY_OUT_ROOT := output/defense_allergy_omission
DEF_MED_OUT_ROOT := output/defense_medication_omission

prepare_medication:
	@$(PY) -c "import os; os.makedirs(r'$(MED_SUBSET_DIR)', exist_ok=True); os.makedirs(r'$(MED_RESULT_DIR)', exist_ok=True)"

medication_subdataset: prepare_medication
	@$(PY) data/src/create_medication_omission_dataset.py --medications_csv "$(MEDICATIONS_SRC)" --patients_csv "$(PATIENTS_SRC)" --n $(N) --seed $(SEED) --out_dir "$(MED_SUBSET_DIR)"

medication_prompts: medication_subdataset
	@$(PY) main/src/medication_omission_experiment.py --medications_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --out_dir "$(MED_RESULT_DIR)"

medication_prompts_llama31: medication_subdataset
	@$(PY) main/src/medication_omission_experiment.py --medications_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --out_dir "$(MED_RESULT_DIR)" --model_key llama31_groq --prompts_out_jsonl "$(MED_RESULT_DIR)/prompts_llama31_groq.jsonl"

medication_prompts_llama33: medication_subdataset
	@$(PY) main/src/medication_omission_experiment.py --medications_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --out_dir "$(MED_RESULT_DIR)" --model_key llama33_groq --prompts_out_jsonl "$(MED_RESULT_DIR)/prompts_llama33_groq.jsonl"

medication_prompts_qwen3: medication_subdataset
	@$(PY) main/src/medication_omission_experiment.py --medications_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --out_dir "$(MED_RESULT_DIR)" --model_key qwen3_groq --prompts_out_jsonl "$(MED_RESULT_DIR)/prompts_qwen3_groq.jsonl"

medication_prompts_gpt41: medication_subdataset
	@$(PY) main/src/medication_omission_experiment.py --medications_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --out_dir "$(MED_RESULT_DIR)" --model_key gpt41 --prompts_out_jsonl "$(MED_RESULT_DIR)/prompts_gpt41.jsonl"

medication_prompts_gpt51: medication_subdataset
	@$(PY) main/src/medication_omission_experiment.py --medications_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --out_dir "$(MED_RESULT_DIR)" --model_key gpt51 --prompts_out_jsonl "$(MED_RESULT_DIR)/prompts_gpt51.jsonl"

medication_prompts_claude45: medication_subdataset
	@$(PY) main/src/medication_omission_experiment.py --medications_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --out_dir "$(MED_RESULT_DIR)" --model_key claude45 --prompts_out_jsonl "$(MED_RESULT_DIR)/prompts_claude45.jsonl"

medication_prompts_all: medication_prompts_llama31 medication_prompts_llama33 medication_prompts_qwen3 medication_prompts_gpt41 medication_prompts_gpt51 medication_prompts_claude45

# Model runs (prompts.jsonl -> generations_*.jsonl)
medication_llama31: medication_prompts
	@$(PY) "LLM Model/Llama 3.1/llama31_groq_runner.py" --model "$(LLAMA31_MODEL)" --prompts_jsonl "$(MED_RESULT_DIR)/prompts_llama31_groq.jsonl" --out_jsonl "$(MED_RESULT_DIR)/generations_llama31_groq.jsonl" --max_tokens 0 --limit $(RUN_LIMIT)

medication_llama33: medication_prompts
	@$(PY) "LLM Model/Llama 3.3/llama33_groq_runner.py" --model "$(LLAMA33_MODEL)" --prompts_jsonl "$(MED_RESULT_DIR)/prompts_llama33_groq.jsonl" --out_jsonl "$(MED_RESULT_DIR)/generations_llama33_groq.jsonl" --max_tokens 0 --limit $(RUN_LIMIT)

medication_qwen3: medication_prompts
	@$(PY) "LLM Model/Qwen 3/qwen3_bedrock_runner.py" --model "$(QWEN3_MODEL)" --prompts_jsonl "$(MED_RESULT_DIR)/prompts_qwen3_groq.jsonl" --out_jsonl "$(MED_RESULT_DIR)/generations_qwen3_bedrock.jsonl" --max_tokens 0 --limit $(RUN_LIMIT)

medication_gpt41: medication_prompts
	@$(PY) "LLM Model/GPT-4.1/gpt41_api.py" --model "$(GPT41_MODEL)" --prompts_jsonl "$(MED_RESULT_DIR)/prompts_gpt41.jsonl" --out_jsonl "$(MED_RESULT_DIR)/generations_gpt41.jsonl" --limit $(RUN_LIMIT)

medication_gpt51: medication_prompts
	@$(PY) "LLM Model/GPT-5.1/gpt51_api.py" --model "$(GPT51_MODEL)" --prompts_jsonl "$(MED_RESULT_DIR)/prompts_gpt51.jsonl" --out_jsonl "$(MED_RESULT_DIR)/generations_gpt51.jsonl" --limit $(RUN_LIMIT)

medication_claude45: medication_prompts
	@$(PY) "LLM Model/Claude 4.5/claude45_api.py" --model "$(CLAUDE45_MODEL)" --prompts_jsonl "$(MED_RESULT_DIR)/prompts_claude45.jsonl" --out_jsonl "$(MED_RESULT_DIR)/generations_claude_sonnet45.jsonl" --limit $(RUN_LIMIT)

medication_models: medication_prompts_all medication_llama31 medication_llama33 medication_qwen3 medication_gpt41 medication_gpt51 medication_claude45

# AI-SUGGESTION: Run the 6 medication model runners concurrently.
medication_models_parallel: medication_prompts_all
	@$(PY) main/src/parallel_model_runs.py --pipeline medication --root "$(MED_RESULT_DIR)" --limit $(RUN_LIMIT) --max_parallel 6

# Scoring (generations_*.jsonl -> scored results)
medication_score_llama31:
	@$(PY) -c "import os; os.makedirs(r'$(MED_SCORED_ROOT)/llama31_groq', exist_ok=True)"
	@$(PY) main/src/medication_omission_experiment.py --medications_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --model_key llama31_groq --out_dir "$(MED_SCORED_ROOT)/llama31_groq" --generations_jsonl "$(MED_RESULT_DIR)/generations_llama31_groq.jsonl"

medication_score_llama33:
	@$(PY) -c "import os; os.makedirs(r'$(MED_SCORED_ROOT)/llama33_groq', exist_ok=True)"
	@$(PY) main/src/medication_omission_experiment.py --medications_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --model_key llama33_groq --out_dir "$(MED_SCORED_ROOT)/llama33_groq" --generations_jsonl "$(MED_RESULT_DIR)/generations_llama33_groq.jsonl"

medication_score_qwen3:
	@$(PY) -c "import os; os.makedirs(r'$(MED_SCORED_ROOT)/qwen3_groq', exist_ok=True)"
	@$(PY) main/src/medication_omission_experiment.py --medications_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --model_key qwen3_groq --out_dir "$(MED_SCORED_ROOT)/qwen3_groq" --generations_jsonl "$(MED_RESULT_DIR)/generations_qwen3_bedrock.jsonl"

medication_score_gpt41:
	@$(PY) -c "import os; os.makedirs(r'$(MED_SCORED_ROOT)/gpt41', exist_ok=True)"
	@$(PY) main/src/medication_omission_experiment.py --medications_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --model_key gpt41 --out_dir "$(MED_SCORED_ROOT)/gpt41" --generations_jsonl "$(MED_RESULT_DIR)/generations_gpt41.jsonl"

medication_score_gpt51:
	@$(PY) -c "import os; os.makedirs(r'$(MED_SCORED_ROOT)/gpt51', exist_ok=True)"
	@$(PY) main/src/medication_omission_experiment.py --medications_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --model_key gpt51 --out_dir "$(MED_SCORED_ROOT)/gpt51" --generations_jsonl "$(MED_RESULT_DIR)/generations_gpt51.jsonl"

medication_score_claude45:
	@$(PY) -c "import os; os.makedirs(r'$(MED_SCORED_ROOT)/claude45', exist_ok=True)"
	@$(PY) main/src/medication_omission_experiment.py --medications_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --model_key claude45 --out_dir "$(MED_SCORED_ROOT)/claude45" --generations_jsonl "$(MED_RESULT_DIR)/generations_claude_sonnet45.jsonl"

medication_score_all: medication_score_llama31 medication_score_llama33 medication_score_qwen3 medication_score_gpt41 medication_score_gpt51 medication_score_claude45

medication: medication_models medication_score_all

# AI-SUGGESTION: Full pipeline with parallel model execution.
medication_parallel: medication_models_parallel medication_score_all

medication_test: RUN_LIMIT=2
medication_test: medication

# -----------------------------
# Defense: Allergy omission (trusted/untrusted JSON + verifier scoring)
# -----------------------------

defense_allergy_prompts_llama31: allergy_subdataset
	@$(PY) main/src/defense_omission_experiment.py --task allergy --model_key llama31_groq --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --items_csv "$(SUBSET_DIR)/allergies_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_ALLERGY_OUT_ROOT)/llama31_groq"

defense_allergy_prompts_llama33: allergy_subdataset
	@$(PY) main/src/defense_omission_experiment.py --task allergy --model_key llama33_groq --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --items_csv "$(SUBSET_DIR)/allergies_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_ALLERGY_OUT_ROOT)/llama33_groq"

defense_allergy_prompts_qwen3: allergy_subdataset
	@$(PY) main/src/defense_omission_experiment.py --task allergy --model_key qwen3_groq --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --items_csv "$(SUBSET_DIR)/allergies_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_ALLERGY_OUT_ROOT)/qwen3_groq"

defense_allergy_prompts_gpt41: allergy_subdataset
	@$(PY) main/src/defense_omission_experiment.py --task allergy --model_key gpt41 --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --items_csv "$(SUBSET_DIR)/allergies_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_ALLERGY_OUT_ROOT)/gpt41"

defense_allergy_prompts_gpt51: allergy_subdataset
	@$(PY) main/src/defense_omission_experiment.py --task allergy --model_key gpt51 --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --items_csv "$(SUBSET_DIR)/allergies_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_ALLERGY_OUT_ROOT)/gpt51"

defense_allergy_prompts_claude45: allergy_subdataset
	@$(PY) main/src/defense_omission_experiment.py --task allergy --model_key claude45 --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --items_csv "$(SUBSET_DIR)/allergies_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_ALLERGY_OUT_ROOT)/claude45"

defense_allergy_prompts_all: defense_allergy_prompts_llama31 defense_allergy_prompts_llama33 defense_allergy_prompts_qwen3 defense_allergy_prompts_gpt41 defense_allergy_prompts_gpt51 defense_allergy_prompts_claude45

defense_allergy_llama31: defense_allergy_prompts_llama31
	@$(PY) "LLM Model/Llama 3.1/llama31_groq_runner.py" --model "$(LLAMA31_MODEL)" --prompts_jsonl "$(DEF_ALLERGY_OUT_ROOT)/llama31_groq/prompts.jsonl" --out_jsonl "$(DEF_ALLERGY_OUT_ROOT)/llama31_groq/generations_llama31_groq.jsonl" --max_tokens 0 --limit $(RUN_LIMIT)

defense_allergy_llama33: defense_allergy_prompts_llama33
	@$(PY) "LLM Model/Llama 3.3/llama33_groq_runner.py" --model "$(LLAMA33_MODEL)" --prompts_jsonl "$(DEF_ALLERGY_OUT_ROOT)/llama33_groq/prompts.jsonl" --out_jsonl "$(DEF_ALLERGY_OUT_ROOT)/llama33_groq/generations_llama33_groq.jsonl" --max_tokens 0 --limit $(RUN_LIMIT)

defense_allergy_qwen3: defense_allergy_prompts_qwen3
	@$(PY) "LLM Model/Qwen 3/qwen3_bedrock_runner.py" --model "$(QWEN3_MODEL)" --prompts_jsonl "$(DEF_ALLERGY_OUT_ROOT)/qwen3_groq/prompts.jsonl" --out_jsonl "$(DEF_ALLERGY_OUT_ROOT)/qwen3_groq/generations_qwen3_groq.jsonl" --max_tokens 0 --limit $(RUN_LIMIT)

defense_allergy_gpt41: defense_allergy_prompts_gpt41
	@$(PY) "LLM Model/GPT-4.1/gpt41_api.py" --model "$(GPT41_MODEL)" --prompts_jsonl "$(DEF_ALLERGY_OUT_ROOT)/gpt41/prompts.jsonl" --out_jsonl "$(DEF_ALLERGY_OUT_ROOT)/gpt41/generations_gpt41.jsonl" --limit $(RUN_LIMIT)

defense_allergy_gpt51: defense_allergy_prompts_gpt51
	@$(PY) "LLM Model/GPT-5.1/gpt51_api.py" --model "$(GPT51_MODEL)" --prompts_jsonl "$(DEF_ALLERGY_OUT_ROOT)/gpt51/prompts.jsonl" --out_jsonl "$(DEF_ALLERGY_OUT_ROOT)/gpt51/generations_gpt51.jsonl" --limit $(RUN_LIMIT)

defense_allergy_claude45: defense_allergy_prompts_claude45
	@$(PY) "LLM Model/Claude 4.5/claude45_api.py" --model "$(CLAUDE45_MODEL)" --prompts_jsonl "$(DEF_ALLERGY_OUT_ROOT)/claude45/prompts.jsonl" --out_jsonl "$(DEF_ALLERGY_OUT_ROOT)/claude45/generations_claude45.jsonl" --limit $(RUN_LIMIT)

defense_allergy_models: defense_allergy_prompts_all defense_allergy_llama31 defense_allergy_llama33 defense_allergy_qwen3 defense_allergy_gpt41 defense_allergy_gpt51 defense_allergy_claude45

# AI-SUGGESTION: Run the 6 defense-allergy model runners concurrently.
defense_allergy_models_parallel: defense_allergy_prompts_all
	@$(PY) main/src/parallel_model_runs.py --pipeline defense_allergy --root "$(DEF_ALLERGY_OUT_ROOT)" --limit $(RUN_LIMIT) --max_parallel 6

defense_allergy_score_llama31:
	@$(PY) main/src/defense_omission_experiment.py --task allergy --model_key llama31_groq --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --items_csv "$(SUBSET_DIR)/allergies_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_ALLERGY_OUT_ROOT)/llama31_groq" --generations_jsonl "$(DEF_ALLERGY_OUT_ROOT)/llama31_groq/generations_llama31_groq.jsonl"

defense_allergy_score_llama33:
	@$(PY) main/src/defense_omission_experiment.py --task allergy --model_key llama33_groq --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --items_csv "$(SUBSET_DIR)/allergies_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_ALLERGY_OUT_ROOT)/llama33_groq" --generations_jsonl "$(DEF_ALLERGY_OUT_ROOT)/llama33_groq/generations_llama33_groq.jsonl"

defense_allergy_score_qwen3:
	@$(PY) main/src/defense_omission_experiment.py --task allergy --model_key qwen3_groq --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --items_csv "$(SUBSET_DIR)/allergies_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_ALLERGY_OUT_ROOT)/qwen3_groq" --generations_jsonl "$(DEF_ALLERGY_OUT_ROOT)/qwen3_groq/generations_qwen3_groq.jsonl"

defense_allergy_score_gpt41:
	@$(PY) main/src/defense_omission_experiment.py --task allergy --model_key gpt41 --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --items_csv "$(SUBSET_DIR)/allergies_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_ALLERGY_OUT_ROOT)/gpt41" --generations_jsonl "$(DEF_ALLERGY_OUT_ROOT)/gpt41/generations_gpt41.jsonl"

defense_allergy_score_gpt51:
	@$(PY) main/src/defense_omission_experiment.py --task allergy --model_key gpt51 --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --items_csv "$(SUBSET_DIR)/allergies_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_ALLERGY_OUT_ROOT)/gpt51" --generations_jsonl "$(DEF_ALLERGY_OUT_ROOT)/gpt51/generations_gpt51.jsonl"

defense_allergy_score_claude45:
	@$(PY) main/src/defense_omission_experiment.py --task allergy --model_key claude45 --patient_ids_txt "$(SUBSET_DIR)/patient_ids.txt" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --items_csv "$(SUBSET_DIR)/allergies_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_ALLERGY_OUT_ROOT)/claude45" --generations_jsonl "$(DEF_ALLERGY_OUT_ROOT)/claude45/generations_claude45.jsonl"

defense_allergy_score_all: defense_allergy_score_llama31 defense_allergy_score_llama33 defense_allergy_score_qwen3 defense_allergy_score_gpt41 defense_allergy_score_gpt51 defense_allergy_score_claude45

defense_allergy: defense_allergy_models defense_allergy_score_all

# AI-SUGGESTION: Full pipeline with parallel model execution.
defense_allergy_parallel: defense_allergy_models_parallel defense_allergy_score_all

defense_allergy_test: RUN_LIMIT=2
defense_allergy_test: MAX_PATIENTS=1
defense_allergy_test: defense_allergy

# Qwen-only defense run (allergy)
defense_allergy_qwen: defense_allergy_prompts_qwen3 defense_allergy_qwen3 defense_allergy_score_qwen3

# -----------------------------
# Defense: Medication omission (trusted/untrusted JSON + verifier scoring)
# -----------------------------

defense_medication_prompts_llama31: medication_subdataset
	@$(PY) main/src/defense_omission_experiment.py --task medication --model_key llama31_groq --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --items_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_MED_OUT_ROOT)/llama31_groq"

defense_medication_prompts_llama33: medication_subdataset
	@$(PY) main/src/defense_omission_experiment.py --task medication --model_key llama33_groq --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --items_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_MED_OUT_ROOT)/llama33_groq"

defense_medication_prompts_qwen3: medication_subdataset
	@$(PY) main/src/defense_omission_experiment.py --task medication --model_key qwen3_groq --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --items_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_MED_OUT_ROOT)/qwen3_groq"

defense_medication_prompts_gpt41: medication_subdataset
	@$(PY) main/src/defense_omission_experiment.py --task medication --model_key gpt41 --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --items_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_MED_OUT_ROOT)/gpt41"

defense_medication_prompts_gpt51: medication_subdataset
	@$(PY) main/src/defense_omission_experiment.py --task medication --model_key gpt51 --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --items_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_MED_OUT_ROOT)/gpt51"

defense_medication_prompts_claude45: medication_subdataset
	@$(PY) main/src/defense_omission_experiment.py --task medication --model_key claude45 --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --items_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_MED_OUT_ROOT)/claude45"

defense_medication_prompts_all: defense_medication_prompts_llama31 defense_medication_prompts_llama33 defense_medication_prompts_qwen3 defense_medication_prompts_gpt41 defense_medication_prompts_gpt51 defense_medication_prompts_claude45

defense_medication_llama31: defense_medication_prompts_llama31
	@$(PY) "LLM Model/Llama 3.1/llama31_groq_runner.py" --model "$(LLAMA31_MODEL)" --prompts_jsonl "$(DEF_MED_OUT_ROOT)/llama31_groq/prompts.jsonl" --out_jsonl "$(DEF_MED_OUT_ROOT)/llama31_groq/generations_llama31_groq.jsonl" --max_tokens 0 --limit $(RUN_LIMIT)

defense_medication_llama33: defense_medication_prompts_llama33
	@$(PY) "LLM Model/Llama 3.3/llama33_groq_runner.py" --model "$(LLAMA33_MODEL)" --prompts_jsonl "$(DEF_MED_OUT_ROOT)/llama33_groq/prompts.jsonl" --out_jsonl "$(DEF_MED_OUT_ROOT)/llama33_groq/generations_llama33_groq.jsonl" --max_tokens 0 --limit $(RUN_LIMIT)

defense_medication_qwen3: defense_medication_prompts_qwen3
	@$(PY) "LLM Model/Qwen 3/qwen3_bedrock_runner.py" --model "$(QWEN3_MODEL)" --prompts_jsonl "$(DEF_MED_OUT_ROOT)/qwen3_groq/prompts.jsonl" --out_jsonl "$(DEF_MED_OUT_ROOT)/qwen3_groq/generations_qwen3_groq.jsonl" --max_tokens 0 --limit $(RUN_LIMIT)

defense_medication_gpt41: defense_medication_prompts_gpt41
	@$(PY) "LLM Model/GPT-4.1/gpt41_api.py" --model "$(GPT41_MODEL)" --prompts_jsonl "$(DEF_MED_OUT_ROOT)/gpt41/prompts.jsonl" --out_jsonl "$(DEF_MED_OUT_ROOT)/gpt41/generations_gpt41.jsonl" --limit $(RUN_LIMIT)

defense_medication_gpt51: defense_medication_prompts_gpt51
	@$(PY) "LLM Model/GPT-5.1/gpt51_api.py" --model "$(GPT51_MODEL)" --prompts_jsonl "$(DEF_MED_OUT_ROOT)/gpt51/prompts.jsonl" --out_jsonl "$(DEF_MED_OUT_ROOT)/gpt51/generations_gpt51.jsonl" --limit $(RUN_LIMIT)

defense_medication_claude45: defense_medication_prompts_claude45
	@$(PY) "LLM Model/Claude 4.5/claude45_api.py" --model "$(CLAUDE45_MODEL)" --prompts_jsonl "$(DEF_MED_OUT_ROOT)/claude45/prompts.jsonl" --out_jsonl "$(DEF_MED_OUT_ROOT)/claude45/generations_claude45.jsonl" --limit $(RUN_LIMIT)

defense_medication_models: defense_medication_prompts_all defense_medication_llama31 defense_medication_llama33 defense_medication_qwen3 defense_medication_gpt41 defense_medication_gpt51 defense_medication_claude45

# AI-SUGGESTION: Run the 6 defense-medication model runners concurrently.
defense_medication_models_parallel: defense_medication_prompts_all
	@$(PY) main/src/parallel_model_runs.py --pipeline defense_medication --root "$(DEF_MED_OUT_ROOT)" --limit $(RUN_LIMIT) --max_parallel 6

defense_medication_score_llama31:
	@$(PY) main/src/defense_omission_experiment.py --task medication --model_key llama31_groq --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --items_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_MED_OUT_ROOT)/llama31_groq" --generations_jsonl "$(DEF_MED_OUT_ROOT)/llama31_groq/generations_llama31_groq.jsonl"

defense_medication_score_llama33:
	@$(PY) main/src/defense_omission_experiment.py --task medication --model_key llama33_groq --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --items_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_MED_OUT_ROOT)/llama33_groq" --generations_jsonl "$(DEF_MED_OUT_ROOT)/llama33_groq/generations_llama33_groq.jsonl"

defense_medication_score_qwen3:
	@$(PY) main/src/defense_omission_experiment.py --task medication --model_key qwen3_groq --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --items_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_MED_OUT_ROOT)/qwen3_groq" --generations_jsonl "$(DEF_MED_OUT_ROOT)/qwen3_groq/generations_qwen3_groq.jsonl"

defense_medication_score_gpt41:
	@$(PY) main/src/defense_omission_experiment.py --task medication --model_key gpt41 --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --items_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_MED_OUT_ROOT)/gpt41" --generations_jsonl "$(DEF_MED_OUT_ROOT)/gpt41/generations_gpt41.jsonl"

defense_medication_score_gpt51:
	@$(PY) main/src/defense_omission_experiment.py --task medication --model_key gpt51 --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --items_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_MED_OUT_ROOT)/gpt51" --generations_jsonl "$(DEF_MED_OUT_ROOT)/gpt51/generations_gpt51.jsonl"

defense_medication_score_claude45:
	@$(PY) main/src/defense_omission_experiment.py --task medication --model_key claude45 --patient_ids_txt "$(MED_SUBSET_DIR)/patient_ids.txt" --patients_csv "$(MED_SUBSET_DIR)/patients_subset.csv" --items_csv "$(MED_SUBSET_DIR)/medications_subset.csv" --max_patients $(MED_MAX_PATIENTS) --seed $(SEED) --out_dir "$(DEF_MED_OUT_ROOT)/claude45" --generations_jsonl "$(DEF_MED_OUT_ROOT)/claude45/generations_claude45.jsonl"

defense_medication_score_all: defense_medication_score_llama31 defense_medication_score_llama33 defense_medication_score_qwen3 defense_medication_score_gpt41 defense_medication_score_gpt51 defense_medication_score_claude45

defense_medication: defense_medication_models defense_medication_score_all

# AI-SUGGESTION: Full pipeline with parallel model execution.
defense_medication_parallel: defense_medication_models_parallel defense_medication_score_all

defense_medication_test: RUN_LIMIT=2
defense_medication_test: MED_MAX_PATIENTS=1
defense_medication_test: defense_medication

# Qwen-only defense run (medication)
defense_medication_qwen: defense_medication_prompts_qwen3 defense_medication_qwen3 defense_medication_score_qwen3

clean:
	@$(PY) -c "import shutil; shutil.rmtree(r'output', ignore_errors=True)"




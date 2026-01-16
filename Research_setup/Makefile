# Research_setup/Makefile
# // AI-SUGGESTION: Run this Makefile from the Research_setup folder: `cd Research_setup && make allergy`

PY ?= python

# -----------------------------
# Paths (relative to Research_setup/)
# -----------------------------

OUT_ROOT := output/allergy_omission
RESULT_DIR := $(OUT_ROOT)/results

SUBSET_DIR := data/subset/data_subset_allergyomissionsubset

ALLERGIES_SRC := data/synthea_1m_fhir_3_0_May_24/output_1/csv/allergies.csv
PATIENTS_SRC := data/synthea_1m_fhir_3_0_May_24/output_1/csv/patients.csv

# -----------------------------
# Experiment parameters
# -----------------------------

N ?= 1000
SEED ?= 7
MAX_PATIENTS ?= 1000

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
.PHONY: allergy_llama31 allergy_llama33 allergy_qwen3 allergy_gpt41 allergy_gpt51 allergy_claude45 allergy_models
.PHONY: allergy_score_llama31 allergy_score_llama33 allergy_score_qwen3 allergy_score_gpt41 allergy_score_gpt51 allergy_score_claude45 allergy_score_all
.PHONY: allergy allergy_test clean

prepare_allergy:
	@$(PY) -c "import os; os.makedirs(r'$(SUBSET_DIR)', exist_ok=True); os.makedirs(r'$(RESULT_DIR)', exist_ok=True)"

allergy_subdataset: prepare_allergy
	@$(PY) data/src/create_allergy_omission_dataset.py --allergies_csv "$(ALLERGIES_SRC)" --patients_csv "$(PATIENTS_SRC)" --n $(N) --seed $(SEED) --out_dir "$(SUBSET_DIR)"

allergy_prompts: allergy_subdataset
	@$(PY) main/src/allergy_omission_experiment.py --allergies_csv "$(SUBSET_DIR)/allergies_subset.csv" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(RESULT_DIR)"

# -----------------------------
# Model runs (prompts.jsonl -> generations_*.jsonl)
# -----------------------------

allergy_llama31: allergy_prompts
	@$(PY) "LLM Model/Llama 3.1/llama31_groq_runner.py" --model "$(LLAMA31_MODEL)" --prompts_jsonl "$(RESULT_DIR)/prompts.jsonl" --out_jsonl "$(RESULT_DIR)/generations_llama31_groq.jsonl" --max_tokens 0 --limit $(RUN_LIMIT)

allergy_llama33: allergy_prompts
	@$(PY) "LLM Model/Llama 3.3/llama33_groq_runner.py" --model "$(LLAMA33_MODEL)" --prompts_jsonl "$(RESULT_DIR)/prompts.jsonl" --out_jsonl "$(RESULT_DIR)/generations_llama33_groq.jsonl" --max_tokens 0 --limit $(RUN_LIMIT)

allergy_qwen3: allergy_prompts
	@$(PY) "LLM Model/Qwen 3/qwen3_bedrock_runner.py" --model "$(QWEN3_MODEL)" --prompts_jsonl "$(RESULT_DIR)/prompts.jsonl" --out_jsonl "$(RESULT_DIR)/generations_qwen3_bedrock.jsonl" --max_tokens 0 --limit $(RUN_LIMIT)

allergy_gpt41: allergy_prompts
	@$(PY) "LLM Model/GPT-4.1/gpt41_api.py" --model "$(GPT41_MODEL)" --prompts_jsonl "$(RESULT_DIR)/prompts.jsonl" --out_jsonl "$(RESULT_DIR)/generations_gpt41.jsonl" --limit $(RUN_LIMIT)

allergy_gpt51: allergy_prompts
	@$(PY) "LLM Model/GPT-5.1/gpt51_api.py" --model "$(GPT51_MODEL)" --prompts_jsonl "$(RESULT_DIR)/prompts.jsonl" --out_jsonl "$(RESULT_DIR)/generations_gpt51.jsonl" --limit $(RUN_LIMIT)

allergy_claude45: allergy_prompts
	@$(PY) "LLM Model/Claude 4.5/claude45_api.py" --model "$(CLAUDE45_MODEL)" --prompts_jsonl "$(RESULT_DIR)/prompts.jsonl" --out_jsonl "$(RESULT_DIR)/generations_claude_sonnet45.jsonl" --limit $(RUN_LIMIT)

allergy_models: allergy_llama31 allergy_llama33 allergy_qwen3 allergy_gpt41 allergy_gpt51 allergy_claude45

# -----------------------------
# Scoring (generations_*.jsonl -> scored results)
# We write scored outputs under output/allergy_omission/scored/<model>/
# -----------------------------

SCORED_ROOT := $(OUT_ROOT)/scored

allergy_score_llama31:
	@$(PY) -c "import os; os.makedirs(r'$(SCORED_ROOT)/llama31_groq', exist_ok=True)"
	@$(PY) main/src/allergy_omission_experiment.py --allergies_csv "$(SUBSET_DIR)/allergies_subset.csv" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(SCORED_ROOT)/llama31_groq" --generations_jsonl "$(RESULT_DIR)/generations_llama31_groq.jsonl"

allergy_score_llama33:
	@$(PY) -c "import os; os.makedirs(r'$(SCORED_ROOT)/llama33_groq', exist_ok=True)"
	@$(PY) main/src/allergy_omission_experiment.py --allergies_csv "$(SUBSET_DIR)/allergies_subset.csv" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(SCORED_ROOT)/llama33_groq" --generations_jsonl "$(RESULT_DIR)/generations_llama33_groq.jsonl"

allergy_score_qwen3:
	@$(PY) -c "import os; os.makedirs(r'$(SCORED_ROOT)/qwen3_groq', exist_ok=True)"
	@$(PY) main/src/allergy_omission_experiment.py --allergies_csv "$(SUBSET_DIR)/allergies_subset.csv" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(SCORED_ROOT)/qwen3_groq" --generations_jsonl "$(RESULT_DIR)/generations_qwen3_bedrock.jsonl"

allergy_score_gpt41:
	@$(PY) -c "import os; os.makedirs(r'$(SCORED_ROOT)/gpt41', exist_ok=True)"
	@$(PY) main/src/allergy_omission_experiment.py --allergies_csv "$(SUBSET_DIR)/allergies_subset.csv" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(SCORED_ROOT)/gpt41" --generations_jsonl "$(RESULT_DIR)/generations_gpt41.jsonl"

allergy_score_gpt51:
	@$(PY) -c "import os; os.makedirs(r'$(SCORED_ROOT)/gpt51', exist_ok=True)"
	@$(PY) main/src/allergy_omission_experiment.py --allergies_csv "$(SUBSET_DIR)/allergies_subset.csv" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(SCORED_ROOT)/gpt51" --generations_jsonl "$(RESULT_DIR)/generations_gpt51.jsonl"

allergy_score_claude45:
	@$(PY) -c "import os; os.makedirs(r'$(SCORED_ROOT)/claude45', exist_ok=True)"
	@$(PY) main/src/allergy_omission_experiment.py --allergies_csv "$(SUBSET_DIR)/allergies_subset.csv" --patients_csv "$(SUBSET_DIR)/patients_subset.csv" --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir "$(SCORED_ROOT)/claude45" --generations_jsonl "$(RESULT_DIR)/generations_claude_sonnet45.jsonl"

allergy_score_all: allergy_score_llama31 allergy_score_llama33 allergy_score_qwen3 allergy_score_gpt41 allergy_score_gpt51 allergy_score_claude45

# Full pipeline
allergy: allergy_models allergy_score_all

# Smoke-test pipeline (limit 2 prompts)
allergy_test: RUN_LIMIT=2
allergy_test: allergy

clean:
	@$(PY) -c "import shutil; shutil.rmtree(r'output/allergy_omission', ignore_errors=True)"




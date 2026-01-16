# Makefile
# // AI-SUGGESTION: Uses Python for directory creation so this works on Windows (PowerShell/cmd) with GNU Make installed.

PY ?= python

OUT_ROOT := Research_setup/output/allergy_omission
SUBSET_DIR := $(OUT_ROOT)/subdataset
RESULT_DIR := $(OUT_ROOT)/results

ALLERGIES_SRC := Research_setup/data/synthea_1m_fhir_3_0_May_24/output_1/csv/allergies.csv
PATIENTS_SRC := Research_setup/data/synthea_1m_fhir_3_0_May_24/output_1/csv/patients.csv

N ?= 1000
SEED ?= 7
MAX_PATIENTS ?= 1000

.PHONY: all prepare subdataset experiment clean
.PHONY: prompt prompts experiment_all_models
.PHONY: llama31 llama33 qwen3 gpt41 gpt51 claude45 runners test

# AI-SUGGESTION: Shared limit flag across all runner targets. 0 = run all prompts.
RUN_LIMIT ?= 0

# AI-SUGGESTION: Default model ids (override at make-time if desired).
LLAMA31_MODEL ?= llama-3.1-8b-instant
LLAMA33_MODEL ?= llama-3.3-70b-versatile
QWEN3_MODEL ?= qwen/qwen3-32b
GPT41_MODEL ?= gpt-4.1
GPT51_MODEL ?= gpt-5.1
CLAUDE45_MODEL ?= claude-sonnet-4-5

# Full end-to-end run (runs Meditron-7B).
all: experiment_all_models

prepare:
	@$(PY) -c "import os; os.makedirs(r'$(SUBSET_DIR)', exist_ok=True); os.makedirs(r'$(RESULT_DIR)', exist_ok=True)"

subdataset: prepare
	@$(PY) Research_setup/src/create_sub_dataset.py --allergies_csv $(ALLERGIES_SRC) --patients_csv $(PATIENTS_SRC) --n $(N) --seed $(SEED) --out_dir $(SUBSET_DIR)

# Default: build prompts and template results WITHOUT running any model.
prompts: subdataset
	@$(PY) Research_setup/src/allergy_omission_experiment.py --allergies_csv $(SUBSET_DIR)/allergies_subset.csv --patients_csv $(SUBSET_DIR)/patients_subset.csv --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir $(RESULT_DIR)

prompt: prompts

experiment: prompts

# Run model inference (Meditron-7B by default). Use this in your inference environment.
experiment_all_models: subdataset
	@$(PY) Research_setup/src/allergy_omission_experiment.py --allergies_csv $(SUBSET_DIR)/allergies_subset.csv --patients_csv $(SUBSET_DIR)/patients_subset.csv --max_patients $(MAX_PATIENTS) --seed $(SEED) --out_dir $(RESULT_DIR) --run_all_models

# -----------------------------
# LLM Model runners (API/Groq)
# -----------------------------

# AI-SUGGESTION: Groq runners require GROQ_API_KEY.
llama31: prompts
	@$(PY) "Medical LLM Model/Llama 3.1/llama31_groq_runner.py" --model "$(LLAMA31_MODEL)" --prompts_jsonl "$(RESULT_DIR)/prompts.jsonl" --out_jsonl "$(RESULT_DIR)/generations_llama31_groq.jsonl" --max_tokens 0 --limit $(RUN_LIMIT)

llama33: prompts
	@$(PY) "Medical LLM Model/Llama 3.3/llama33_groq_runner.py" --model "$(LLAMA33_MODEL)" --prompts_jsonl "$(RESULT_DIR)/prompts.jsonl" --out_jsonl "$(RESULT_DIR)/generations_llama33_groq.jsonl" --max_tokens 0 --limit $(RUN_LIMIT)

qwen3: prompts
	@$(PY) "Medical LLM Model/Qwen 3/qwen3_bedrock_runner.py" --model "$(QWEN3_MODEL)" --prompts_jsonl "$(RESULT_DIR)/prompts.jsonl" --out_jsonl "$(RESULT_DIR)/generations_qwen3_bedrock.jsonl" --max_tokens 0 --limit $(RUN_LIMIT)

# AI-SUGGESTION: OpenAI runners require OPENAI_API_KEY.
gpt41: prompts
	@$(PY) "Medical LLM Model/GPT-4.1/gpt41_api.py" --model "$(GPT41_MODEL)" --prompts_jsonl "$(RESULT_DIR)/prompts.jsonl" --out_jsonl "$(RESULT_DIR)/generations_gpt41.jsonl" --limit $(RUN_LIMIT)

gpt51: prompts
	@$(PY) "Medical LLM Model/GPT-5.1/gpt51_api.py" --model "$(GPT51_MODEL)" --prompts_jsonl "$(RESULT_DIR)/prompts.jsonl" --out_jsonl "$(RESULT_DIR)/generations_gpt51.jsonl" --limit $(RUN_LIMIT)

# AI-SUGGESTION: Anthropic runner requires ANTHROPIC_API_KEY.
claude45: prompts
	@$(PY) "Medical LLM Model/Claude 4.5/claude45_api.py" --model "$(CLAUDE45_MODEL)" --prompts_jsonl "$(RESULT_DIR)/prompts.jsonl" --out_jsonl "$(RESULT_DIR)/generations_claude_sonnet45.jsonl" --limit $(RUN_LIMIT)

# Run all runners (full run uses RUN_LIMIT=0 by default).
runners: llama31 llama33 qwen3 gpt41 gpt51 claude45

# Smoke test: run all runners with only the first 2 prompts.
test: RUN_LIMIT=2
test: runners

clean:
	@$(PY) -c "import shutil; shutil.rmtree(r'$(OUT_ROOT)', ignore_errors=True)"




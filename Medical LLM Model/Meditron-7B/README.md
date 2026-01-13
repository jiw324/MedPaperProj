# Meditron-7B Experiment — Text CLI + Small Eval

This folder provides a setup similar to your Meditron-70B / BioMistral / OpenBioLLM folders, but for **Meditron-7B**.

Default model id:
- `epfl-llm/meditron-7b`

## Install

```powershell
cd "C:\Users\wangj\Desktop\2025 Winter\MedPaperProj\Medical LLM Model\Meditron-7B"
python -m pip install -r requirements.txt
```

## (Optional) Check access without downloading weights

```powershell
$env:HF_TOKEN="hf_..."   # only if gated
python check_access.py --model epfl-llm/meditron-7b
```

## 1) Text-based interface (CLI)

```powershell
python run_meditron7b_cli.py --model epfl-llm/meditron-7b --device-map auto
```

Recommended on GPU (if supported):

```powershell
python run_meditron7b_cli.py --model epfl-llm/meditron-7b --device-map auto --quant 4bit
```

## 2) Small medical QA evaluation (subset-based)

PubMedQA:

```powershell
python evaluate_medical_qa.py --dataset pubmed_qa --max-samples 50
```

MedMCQA:

```powershell
python evaluate_medical_qa.py --dataset medmcqa --max-samples 100
```








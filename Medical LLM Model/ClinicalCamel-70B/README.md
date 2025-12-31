# Clinical Camel Experiment — Text CLI + Small Eval

Clinical Camel model (Hugging Face):
- [`wanglab/ClinicalCamel-70B`](https://huggingface.co/wanglab/ClinicalCamel-70B?utm_source=openai)

Clinical Camel paper (background):
- [`arXiv:2305.12031`](https://arxiv.org/abs/2305.12031?utm_source=openai)

## Install

```powershell
cd "C:\Users\wangj\Desktop\2025 Winter\MedPaperProj\Medical LLM Model\Clinical Camel"
python -m pip install -r requirements.txt
```

## (Optional) Check access without downloading 70B weights

```powershell
$env:HF_TOKEN="hf_..."   # only if gated/license restricted
python check_access.py --model wanglab/ClinicalCamel-70B
```

## 1) Text-based interface (CLI)

```powershell
python run_clinicalcamel_cli.py --model wanglab/ClinicalCamel-70B
```

Recommended on GPU:

```powershell
python run_clinicalcamel_cli.py --model wanglab/ClinicalCamel-70B --device-map auto --quant 4bit
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



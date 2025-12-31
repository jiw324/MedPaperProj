# PubMedGPT 2.7B (BioMedLM) — Text CLI + Small Eval

PubMedGPT 2.7B is commonly distributed as **BioMedLM**:
- Model: [`stanford-crfm/BioMedLM`](https://huggingface.co/stanford-crfm/BioMedLM)
- Project repo: `https://github.com/stanford-crfm/BioMedLM`

## Install

```powershell
cd "C:\Users\wangj\Desktop\2025 Winter\MedPaperProj\Medical LLM Model\PubMedGPT, 2.7B"
python -m pip install -r requirements.txt
```

## (Optional) Check access without downloading weights

```powershell
$env:HF_TOKEN="hf_..."   # only if gated/license restricted
python check_access.py --model stanford-crfm/BioMedLM
```

## 1) Text-based interface (CLI)

```powershell
python run_pubmedgpt_cli.py --model stanford-crfm/BioMedLM --device-map auto
```

GPU + quantization (if supported):

```powershell
python run_pubmedgpt_cli.py --model stanford-crfm/BioMedLM --device-map auto --quant 4bit
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






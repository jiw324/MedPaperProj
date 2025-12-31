# OpenBioLLM Experiment (Llama-3) — Text CLI + Small Eval

Model link (provided):
- [`aaditya/Llama3-OpenBioLLM-70B`](https://huggingface.co/aaditya/Llama3-OpenBioLLM-70B?utm_source=chatgpt.com)

## Install

```powershell
cd "C:\Users\wangj\Desktop\2025 Winter\MedPaperProj\Medical LLM Model\OpenBioLLM"
python -m pip install -r requirements.txt
```

## (Optional) Check access without downloading 70B weights

```powershell
$env:HF_TOKEN="hf_..."   # only if the model/license is gated
python check_access.py --model aaditya/Llama3-OpenBioLLM-70B
```

## 1) Text-based interface (uses Llama-3 chat template)

```powershell
python run_openbiollm_cli.py --model aaditya/Llama3-OpenBioLLM-70B
```

Notes:
- The CLI uses `tokenizer.apply_chat_template(...)` as recommended by the model card.
- 70B requires substantial GPU VRAM; consider quantization flags (`--quant 4bit`) if supported.

## 2) Small medical QA evaluation (subset-based)

PubMedQA:

```powershell
python evaluate_medical_qa_llama3.py --dataset pubmed_qa --max-samples 50
```

MedMCQA:

```powershell
python evaluate_medical_qa_llama3.py --dataset medmcqa --max-samples 100
```





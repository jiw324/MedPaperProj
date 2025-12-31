# BioMistral Experiment (Transformers)

This folder sets up a small experiment scaffold aligned with the **BioMistral** paper and model release.

- Paper: [`arXiv:2402.10373`](https://arxiv.org/abs/2402.10373?utm_source=chatgpt.com)
- Model: [`BioMistral/BioMistral-7B`](https://huggingface.co/BioMistral/BioMistral-7B?utm_source=chatgpt.com)

## Install

```powershell
cd "C:\Users\wangj\Desktop\2025 Winter\MedPaperProj\Medical LLM Model\BioMistral"
python -m pip install -r requirements.txt
```

## 1) Text-based interface (CLI)

```powershell
python run_biomistral_cli.py --model BioMistral/BioMistral-7B
```

## 2) Small medical QA evaluation (subset-based)

PubMedQA (labeled):

```powershell
python evaluate_medical_qa.py --dataset pubmed_qa --split validation --max-samples 50
```

MedMCQA:

```powershell
python evaluate_medical_qa.py --dataset medmcqa --split validation --max-samples 100
```

## Notes / constraints

- **Hardware**: 7B models still benefit a lot from an NVIDIA GPU. CPU inference works but is slow.
- **Model class**: the model card shows `AutoModel`, but for generation we intentionally use `AutoModelForCausalLM`.
- **Quantization**: you can try `--quant 8bit` / `--quant 4bit` if your CUDA + bitsandbytes setup supports it.



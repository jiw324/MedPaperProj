# Meditron (Transformers) — Text CLI

This folder contains a **text-based interface** to load and chat with Meditron models using Hugging Face Transformers.

### Install

From this folder:

```bash
python -m pip install -r requirements.txt
```

### Authentication (important for gated models)

Meditron-70B is commonly a **gated** model on Hugging Face, so you must:

- Request access on the model page (example): `https://huggingface.co/epfl-llm/meditron-70b`
- Create a **Read** token: `https://huggingface.co/settings/tokens`
- Set it before running:

PowerShell:

```powershell
$env:HF_TOKEN="hf_..."
```

CMD:

```bat
set HF_TOKEN=hf_...
```

### Run the text interface

```bash
python chat_cli.py --model epfl-llm/meditron-70b
```

Useful flags:

- `--model epfl-llm/meditron-7b`: smaller model (much easier to run than 70B)
- `--device-map auto`: let Transformers place weights across devices
- `--quant 8bit` / `--quant 4bit`: quantization (typically requires CUDA + bitsandbytes)

### Reference

Upstream project: [epfLLM/meditron](https://github.com/epfLLM/meditron)

# Meditron-70B Model Loader

This directory contains scripts for loading and using the Meditron-70B medical language model from Hugging Face.

## ⚠️ IMPORTANT: Authentication Required

**Meditron-70B is a gated model** and requires HuggingFace authentication.

### Step 1: Request Access

1. Create account at: https://huggingface.co/join
2. Visit model page: https://huggingface.co/epfl-llm/meditron-70b
3. Click **"Request Access"** button
4. Wait for approval (usually instant)

### Step 2: Authenticate

Generate an access token at https://huggingface.co/settings/tokens, then login:

**Option A - Using helper script:**
```bash
python login_huggingface.py
```

**Option B - Using CLI:**
```bash
python -m transformers.cli login
```

**Option C - Set environment variable:**
```bash
# Windows PowerShell
$env:HF_TOKEN="your_token_here"

# Linux/Mac
export HF_TOKEN="your_token_here"
```

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
# Or manually:
pip install transformers accelerate bitsandbytes torch huggingface-hub
```

### Basic Usage (After Authentication)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-70b", token=True)
model = AutoModelForCausalLM.from_pretrained("epfl-llm/meditron-70b", token=True)
```

### Advanced Usage

The `load_meditron.py` script provides additional features:

#### 1. Memory-Efficient Loading (Recommended for most systems)

**8-bit Quantization** (~35GB VRAM required):
```bash
python load_meditron.py --quantization 8bit
```

**4-bit Quantization** (~18GB VRAM required):
```bash
python load_meditron.py --quantization 4bit
```

#### 2. Device Selection

```bash
# Automatic device selection (default)
python load_meditron.py --device auto

# Force CPU (slow but works without GPU)
python load_meditron.py --device cpu

# Force CUDA/GPU
python load_meditron.py --device cuda
```

#### 3. Custom Inference

```bash
python load_meditron.py --prompt "What are the treatment options for hypertension?"
```

#### 4. Skip Inference Example

```bash
python load_meditron.py --no-example
```

## Model Information

- **Model:** Meditron-70B
- **Source:** EPFL LLM Team
- **HuggingFace:** [epfl-llm/meditron-70b](https://huggingface.co/epfl-llm/meditron-70b)
- **Parameters:** 70 billion
- **Memory Requirements:**
  - Full precision (FP16): ~140GB
  - 8-bit quantization: ~35GB
  - 4-bit quantization: ~18GB

## System Requirements

### Minimum Requirements (with 4-bit quantization)
- GPU: 24GB+ VRAM (e.g., RTX 3090, RTX 4090, A5000)
- RAM: 32GB+
- Disk Space: 140GB+

### Recommended Requirements (with 8-bit quantization)
- GPU: 40GB+ VRAM (e.g., A100, A6000)
- RAM: 64GB+
- Disk Space: 140GB+

### Full Precision Requirements
- GPU: 80GB+ VRAM (e.g., A100 80GB) or multi-GPU setup
- RAM: 256GB+
- Disk Space: 140GB+

## Python API Usage

```python
from load_meditron import MeditronLoader

# Initialize loader with 8-bit quantization
loader = MeditronLoader(quantization="8bit", device="auto")

# Load model and tokenizer
tokenizer, model = loader.load()

# Run inference
prompt = "What are the symptoms of type 2 diabetes?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    top_p=0.95
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Troubleshooting

### Authentication Error: "Access to model is restricted"

**Error message:**
```
Cannot access gated repo for url https://huggingface.co/epfl-llm/meditron-70b...
Access to model epfl-llm/meditron-70b is restricted.
```

**Solution:**
1. Request access at: https://huggingface.co/epfl-llm/meditron-70b
2. Generate token at: https://huggingface.co/settings/tokens
3. Run: `python login_huggingface.py`
4. Verify: `python login_huggingface.py --check`

### Out of Memory Error
- Use 4-bit quantization: `--quantization 4bit`
- Reduce batch size in your inference code
- Close other applications using GPU memory

### Slow Loading
- First load downloads ~140GB model (can take hours)
- Subsequent loads use cached model (much faster)
- Check disk space and internet connection

### CUDA Not Available
- Verify PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

## References

- [Meditron Paper](https://arxiv.org/abs/2311.16079)
- [HuggingFace Model Card](https://huggingface.co/epfl-llm/meditron-70b)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

## License

Please refer to the model's license on HuggingFace for usage terms and conditions.


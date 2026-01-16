## Llama 3.1 Setup (Groq Runner)

This folder contains `llama31_groq_runner.py`, which runs **Llama 3.1** remotely on **Groq** using the experiment prompts.

### Prerequisites
- Python 3.9+
- Groq API key in environment variable: `GROQ_API_KEY`

### Get a Groq API key (link)
- `https://console.groq.com/keys`

### Install

```bash
pip install groq
```

### Generate prompts (from repo root)

```bash
make prompts
```

This creates:
- `Research_setup/output/allergy_omission/results/prompts.jsonl`

### Smoke test (Windows PowerShell)

```powershell
$env:GROQ_API_KEY="YOUR_KEY"
$env:LLAMA31_GROQ_MODEL="llama-3.1-8b-instant"
python "Medical LLM Model/Llama 3.1/llama31_groq_runner.py" --limit 2 --max_tokens 0 --print_messages --verbose
```

### Outputs
Default output:
- `Research_setup/output/allergy_omission/results/generations_llama31_groq.jsonl`

Notes:
- The runner is intentionally **pure input → output** (it does not modify prompts and does not do task-specific scoring).
- If you need different “clean vs attack” wording, generate it in the experiment script (the text inside `prompts.jsonl`).



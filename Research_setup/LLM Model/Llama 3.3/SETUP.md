## Llama 3.3 Setup (Groq Runner)

This folder contains `llama33_groq_runner.py`, which runs **Llama 3.3 Instruct** remotely on **Groq** using the experiment prompts.

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

### Smoke test

```bash
export GROQ_API_KEY="YOUR_KEY"
export LLAMA33_GROQ_MODEL="llama-3.3-70b-versatile"
python "Medical LLM Model/Llama 3.3/llama33_groq_runner.py" --limit 2 --max_tokens 0 --print_messages --verbose
```

### Outputs
Default output:
- `Research_setup/output/allergy_omission/results/generations_llama33_groq.jsonl`

Notes:
- The runner is intentionally **pure input → output** (it does not modify prompts and does not do task-specific scoring).
- If you need different “clean vs attack” wording, generate it in the experiment script (the text inside `prompts.jsonl`).



## Qwen 3 Setup (Groq Runner)

This folder contains `qwen3_bedrock_runner.py` (now configured for Groq), which runs **Qwen 3** remotely on **Groq** using the experiment prompts.

### Get an API key (link)
- Groq API keys: `https://console.groq.com/keys`

### Prerequisites
- Python 3.9+
- Groq API key in environment variable: `GROQ_API_KEY`

### Install

```bash
pip install groq
```

### Generate prompts (from repo root)

```bash
make prompts
```

### Smoke test

```bash
export GROQ_API_KEY="YOUR_KEY"
export GROQ_MODEL="qwen/qwen3-32b"
# Use --max_tokens 0 to omit the limit (closest to "unlimited"; still capped by Groq/model).
python "Medical LLM Model/Qwen 3/qwen3_bedrock_runner.py" --limit 2 --max_tokens 0 --verbose
```

Notes:
- The runner is intentionally **pure input → output** (it does not modify prompts and does not do task-specific scoring).
- If you need different “clean vs attack” wording, generate it in the experiment script (the text inside `prompts.jsonl`).

### Outputs
Default output:
- `Research_setup/output/allergy_omission/results/generations_qwen3_bedrock.jsonl`



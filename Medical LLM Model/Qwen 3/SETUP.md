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
- By default, the runner appends: `Don't have a think part in your answer.` to the last user message for Qwen3.
- To disable that behavior: pass `--no_think_instruction`.
- By default, the runner removes literal `<think>` / `</think>` tags from `output_text` (but keeps the content). To keep tags: pass `--keep_think_tags`.

### Outputs
Default output:
- `Research_setup/output/allergy_omission/results/generations_qwen3_bedrock.jsonl`



# 🔍 View Source AI Experiments 

A local-first agent that evaluates GitHub issue/PR comments for sentiment using **OLMo 2 1B Instruct** — a fully open-source LLM trained on fully open data.

> This is really just a sandbox for experimenting and learning about how data informs decisions, what type of transparency is possible with open data an models -  AND to play with ways of visualizing that and providing additional prompts via the examples.json file.  I last experimented with adding prompts related to evalauting sentiment related to the 3.0 verison of the Contributor Covenant.

## Why This Stack?

| Component | Choice | Why It's Truly Open |
|-----------|--------|---------------------|
| **LLM** | [OLMo 2 1B Instruct](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct) | Weights, code, training recipes — all Apache 2.0 |
| **Training Data** | [Dolma](https://huggingface.co/datasets/allenai/dolma) | Fully documented, downloadable, auditable |
| **Organization** | Allen AI | Non-profit research lab committed to open science |
| **License** | Apache 2.0 | No usage restrictions, no "open-washing" |

Many models marketed as "open" (Llama, Mistral) don't disclose training data and have restrictive licenses. OLMo is one of the few where you can inspect *everything*.

## Quick Start

### Option A: Ollama (Recommended — fastest)

```bash
# 1. Install Ollama: https://ollama.com
# 2. Pull the model (~1GB)
ollama pull olmo2:1b

# 3. Install dependencies & run
pip install -r requirements.txt
python app.py
```

### Option B: HuggingFace Transformers (No Ollama needed)

```bash
# The model auto-downloads on first run (~2GB)
OLMO_BACKEND=transformers pip install -r requirements.txt
OLMO_BACKEND=transformers python app.py
```

Then open **http://localhost:7860** in your browser.

## Features

### Built-in Sentiment Categories

- **Toxicity / Hostility** — toxic, hostile, dismissive, neutral, respectful
- **Constructiveness** — constructive, unconstructive, mixed

### Custom Labels

Add your own categories by pasting JSON in the UI:

```json
{
  "urgency": {
    "name": "Urgency",
    "labels": ["critical", "important", "low-priority", "informational"],
    "description": "How urgent is the issue raised in this comment?"
  },
  "expertise": {
    "name": "Technical Depth",
    "labels": ["expert", "intermediate", "beginner", "non-technical"],
    "description": "The apparent technical expertise level of the commenter."
  }
}
```

### Input Formats

- Full URL: `https://github.com/owner/repo/issues/123`
- Full PR URL: `https://github.com/owner/repo/pull/456`
- Shorthand: `owner/repo#123`

### GitHub Token

Optional, but recommended for:
- Private repositories
- Avoiding GitHub's 60 req/hour unauthenticated rate limit
- Generate one at: https://github.com/settings/tokens (only `repo` scope needed for private repos, no scope needed for public)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLMO_BACKEND` | `auto` | Force `ollama` or `transformers` |
| `GITHUB_TOKEN` | — | Default GitHub token (can also set in UI) |

## Architecture

```
User Input (repo URL + custom labels)
       │
       ▼
GitHub API ──► Fetch issue/PR comments
       │
       ▼
Prompt Builder ──► Constructs classification prompt per comment
       │
       ▼
OLMo 2 1B Instruct ──► Local inference (Ollama or Transformers)
       │
       ▼
JSON Parser ──► Extracts structured labels
       │
       ▼
Gradio UI ──► Visual report + raw JSON export
```

## Limitations

- **1B model**: Good for classification but may struggle with nuanced sarcasm or multi-layered sentiment. If you need higher accuracy, consider OLMo 2 7B.
- **Context window**: Comments are truncated to 2000 chars for reliable inference.
- **Rate limits**: GitHub API allows 60 unauthenticated requests/hour. Use a token for more.

## License

This project is MIT licensed. The OLMo model and Dolma dataset are Apache 2.0.

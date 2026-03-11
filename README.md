# View Source for AI — Code Snippet Source and License

> *Code leaves fossils. This reads them.*

Fossil Record searches [Dolma](https://huggingface.co/datasets/allenai/dolma) — the fully open 3-trillion-token dataset used to train [OLMo](https://allenai.org/olmo) — to find impressions of your code in AI training data. Paste a snippet, see exactly where it appears, what license it carried there, and whether that's compatible with your project.

## Why "Fossil Record"?

When a model trains on code, that code leaves an impression in the weights — like an organism preserved in rock. Paleontologists read the fossil record from fragments; a single distinctive bone is enough to identify the species. Fossil Record works the same way: one function signature identifies the library, its license, and its training data history.

The gaps matter too. Deduplication collapses thousands of GitHub copies into a handful of unique documents. A low hit count doesn't mean the code is rare — it means the surviving fossils are few.

## What It Does

1. **Trace** — exact-phrase search of your code against the full Dolma corpus via [infini-gram](https://infini-gram.io)
2. **Inspect** — see source document context and detected license strings for each match
3. **Flag** — 🟢/🟡/🔴/⚪ compatibility between detected source licenses and your project's declared license
4. **Contribute** — report license gaps and data quality issues directly to Allen AI's open repos

## Quick Start

```bash
git clone https://github.com/emmairwin/view-source-ai
cd view-source-ai
pip install -r requirements.txt
python app.py
```

Open **http://localhost:7860** in your browser.

## Stack

| Component | Choice | Why |
|-----------|--------|-----|
| **Training corpus** | [Dolma v1.7](https://huggingface.co/datasets/allenai/dolma) | Fully open, documented, auditable |
| **Search** | [infini-gram](https://infini-gram.io) | Exact n-gram search over 3T tokens |
| **UI** | [Gradio](https://gradio.app) | Local web interface, no cloud dependency |
| **License** | Apache 2.0 | No restrictions |

## Limitations

- Matches are **exact phrases** — minified or reformatted code won't match its readable source form
- Dolma **deduplicates** aggressively — counts reflect unique copies, not real-world usage frequency
- The 800-char document window may not reach a file's license header if the match is mid-function
- This is forensic signal, not legal proof — a cluster of matches is evidence, not verdict

## License

Apache 2.0. The Dolma dataset is also Apache 2.0.


I am aware there are disagreements around openness and AI, with regard to safety, but think of this as a teaching tool and not a proposed solution, and you'll be fine.

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

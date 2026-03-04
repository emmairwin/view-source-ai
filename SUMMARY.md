# GitHub Comment Sentiment Agent — Project Summary

## What It Is

A local-first tool that analyzes GitHub issue/PR comments for toxicity, constructiveness, and custom sentiment labels you define. Built entirely on genuinely open-source components — no black boxes.

## Why It Matters

This isn't just another sentiment classifier. The core idea is **"View Source" for AI** — inspired by how early web browsers let you see and modify what made a page work. Every decision the model makes can be traced back to its training data, explained, and changed.

## The Stack (Strictly Open)

| Component | Choice | Why |
|-----------|--------|-----|
| **LLM** | OLMo 2 1B Instruct (Allen AI) | Weights, code, recipes — all Apache 2.0 |
| **Training Data** | Dolma (3T tokens) | Fully documented, downloadable, auditable |
| **Data Tracing** | OLMoTrace (Allen AI) | Traces model outputs back to exact training documents |
| **UI** | Gradio | Local web interface, no cloud dependency |
| **Inference** | Ollama or HuggingFace Transformers | Runs on CPU, no GPU required |

We specifically rejected Llama and Mistral — they have open weights but undisclosed training data and restrictive licenses. OLMo is one of the few models where you can inspect *everything*.

## Four Capabilities

### 1. Analyze (🔬)
Paste a GitHub issue/PR URL → the agent fetches all comments, classifies each one across your chosen categories, and generates a report with summary stats and per-comment breakdowns. Flagged comments (toxic, hostile, dismissive) get surfaced at the top.

### 2. View Source (🔍)
Every classification comes with:
- **Reasoning** — why the model chose that label
- **Evidence** — the exact phrases from the comment that triggered it
- **Training data provenance** — searches Dolma (via WIMBD/OLMoTrace) for similar text the model was trained on, so you can see what shaped the decision

### 3. Edit Source (✏️)
A local example bank (`examples.jsonl`) that you control. Add labeled examples, and they get injected as few-shot context into the model's prompt — directly steering classification behavior. Import/export as JSON. This is your "edit the source code" for the AI's behavior without retraining.

### 4. Contribute Upstream (🎫)
When you find something wrong, the app drafts a well-formatted GitHub issue targeting the right Allen AI repo:
- **allenai/OLMo** — model behavior / misclassification
- **allenai/dolma** — training data problems
- **allenai/open-instruct** — post-training / instruction tuning improvements
- **allenai/OLMoTrace** — data tracing feedback
- **allenai/OLMo-core** — training framework / architecture

Generates a pre-filled GitHub issue URL — you just click submit. Also supports email to olmo@allenai.org and local draft saving.

## Built-in Sentiment Categories

- **Toxicity / Hostility** — toxic, hostile, dismissive, neutral, respectful
- **Constructiveness** — constructive, unconstructive, mixed
- **Custom** — define any categories via JSON (e.g., urgency, technical depth, etc.)

## How to Run

```bash
# Option A: Ollama (recommended, fastest)
ollama pull olmo2:1b && ollama serve
pip install -r requirements.txt
python app.py

# Option B: HuggingFace Transformers (no Ollama needed, auto-downloads ~2GB)
OLMO_BACKEND=transformers pip install -r requirements.txt
OLMO_BACKEND=transformers python app.py
```

Opens at http://localhost:7860.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Main application — Gradio UI, GitHub API, LLM inference, all tabs |
| `data_provenance.py` | Training data search (Dolma/WIMBD), provenance reports, local example bank |
| `requirements.txt` | Python dependencies |
| `README.md` | Full documentation |
| `examples.jsonl` | Auto-created on first run — your editable example bank |

## Key Design Decisions

1. **Strictly open-source** — rejected "open-washing" models. Every component is auditable.
2. **Local-first** — runs on your machine, no data leaves your network.
3. **Explainable** — no label without reasoning and evidence.
4. **Editable** — you can change the model's behavior without retraining.
5. **Contributable** — feedback flows upstream to improve the model for everyone.

## How to Frame This

This is **not** "I built a sentiment classifier." It's a proof of concept for **"What if AI had View Source?"**

The early web gave everyone the ability to right-click, view source, and understand how things worked — then change them. That transparency was foundational to how the web grew. AI has gone the opposite direction: closed training data, unexplainable decisions, no way to contribute improvements. This project pushes back on that.

The interesting thing isn't the classification accuracy (a bigger closed model would score higher on benchmarks). The interesting things are:

- We deliberately chose a less powerful model because it's *genuinely* open — and can explain why
- Every decision traces back to documented training data you can actually inspect
- You can edit the model's behavior without retraining (via the example bank)
- When something is wrong, you can file a ticket to the actual humans who build the model and its data
- The whole loop — inspect, understand, override, contribute — works from one tool

No other AI stack currently offers this end-to-end. It's only possible because Allen AI open-sourced *everything*, not just the weights.

## What the Example Bank Actually Does (Technically)

The example bank uses **in-context learning / prompt engineering** — no model weights are changed. Labeled examples are injected into the prompt as few-shot context before each classification. This is the lightest intervention possible: zero compute, instant iteration, fully reversible.

If accuracy needs to go further, the natural next step is **LoRA/QLoRA fine-tuning** — training small adapter layers (~0.1-1% of parameters) on your curated examples using Allen AI's open-instruct framework. The example bank you've built becomes your fine-tuning dataset, so nothing is wasted. This requires a modest GPU and a few hours but produces real weight changes.

The spectrum: prompt engineering (our app) → LoRA adapters → full fine-tune → full retrain from scratch. We're at the far left by design — maximum transparency and flexibility.

## Natural Next Steps

- **LoRA fine-tuning** on the curated example bank for higher accuracy
- **Embedding-based search** over Dolma (replacing keyword matching) for better provenance
- **OLMoTrace API integration** for real-time verbatim training data attribution
- **GitHub Action version** to run analysis automatically on new PRs
- **OLMo 2 7B** for teams with a GPU who want better nuance detection

## Limitations

- 1B model — good for classification, may miss nuanced sarcasm. OLMo 2 7B/13B available if you have more VRAM.
- Training data attribution is approximate (semantic similarity), not causal. OLMoTrace provides verbatim matching for exact traces.
- GitHub API rate limit: 60 req/hour unauthenticated. Use a token for more.

## Links

- Model: https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct
- Training Data: https://huggingface.co/datasets/allenai/dolma
- OLMoTrace: https://playground.allenai.org
- Data Explorer (WIMBD): https://wimbd.apps.allenai.org
- Training Recipes: https://github.com/allenai/OLMo-core
- Contact: olmo@allenai.org

"""
Data Provenance & Override System
==================================
"View Source" for AI decisions.

Three capabilities:
1. TRACE   — Search Dolma (OLMo's training data) for text similar to what's being classified
2. INSPECT — Show what training data likely influenced the model's decision
3. OVERRIDE — Maintain a local example bank that steers model behavior (your "edit source")

This makes the AI auditable and editable at the data level.
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 1. DOLMA SEARCH — Look into the training data
# ---------------------------------------------------------------------------

# infini-gram is a live API for searching Dolma (OLMo's training corpus).
# It returns actual text snippets that matched the query in the training data.
# Docs: https://infini-gram.io  API: https://api.infini-gram.io

INFINIGRAM_API = "https://api.infini-gram.io/"
INFINIGRAM_INDEX = "v4_dolma-v1_7_llama"  # Dolma v1.7, the dataset OLMo 2 was trained on

# Common words that carry no signal on their own
_STOPWORDS = {
    "i", "me", "my", "you", "your", "we", "our", "they", "their", "it", "its",
    "a", "an", "the", "and", "or", "but", "so", "if", "is", "are", "was",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "to", "of", "in", "on", "at", "for", "with", "by", "from", "that",
    "this", "not", "no", "can", "will", "just", "about", "well", "even",
}


def _infinigram_count(query: str) -> int:
    """Return how many times a phrase appears in Dolma."""
    import requests
    try:
        r = requests.post(
            INFINIGRAM_API,
            json={"index": INFINIGRAM_INDEX, "query_type": "count", "query": query},
            timeout=10,
        )
        d = r.json()
        return d.get("count", d.get("cnt", 0))
    except Exception:
        return 0


def _infinigram_docs(query: str, n: int = 3) -> tuple[int, list[dict]]:
    """Return (total_count, list of doc snippets) for a phrase."""
    import requests
    try:
        r = requests.post(
            INFINIGRAM_API,
            json={"index": INFINIGRAM_INDEX, "query_type": "search_docs", "query": query, "maxnum": n},
            timeout=15,
        )
        d = r.json()
        docs = []
        for doc in d.get("documents", []):
            spans = doc.get("spans", [])
            text = "".join(s[0] for s in spans if s[0])
            try:
                meta = json.loads(doc.get("metadata", "{}"))
            except Exception:
                meta = {}
            inner = meta.get("metadata", {})
            subreddit = inner.get("metadata", {}).get("subreddit", "")
            source = f"reddit /r/{subreddit}" if subreddit else inner.get("source", "unknown")
            docs.append({"text": text[:400], "source": source, "url": inner.get("url", "")})
        return d.get("cnt", 0), docs
    except Exception:
        return 0, []


def _key_subphrases(text: str) -> list[str]:
    """
    Extract the most likely load-bearing sub-phrases from a comment:
    - All 2–3 word n-grams that contain at least one non-stopword
    - Plus individual non-stopwords > 4 chars
    Sorted by word length descending so longer phrases come first.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    phrases = []
    # 3-grams then 2-grams
    for n in (3, 2):
        for i in range(len(words) - n + 1):
            gram = words[i : i + n]
            if any(w not in _STOPWORDS and len(w) > 2 for w in gram):
                phrases.append(" ".join(gram))
    # Individual meaningful words
    for w in words:
        if w not in _STOPWORDS and len(w) > 4:
            phrases.append(w)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:12]  # cap to avoid too many API calls


def search_dolma_wimbd(query: str, max_results: int = 3) -> list[dict]:
    """
    Multi-layer training data search:
    1. Exact phrase count + sample docs
    2. Key sub-phrase counts (shows broader signal)
    Returns a list with one structured summary dict.
    """
    exact_count, exact_docs = _infinigram_docs(query, n=max_results)

    # Count key sub-phrases (limit to 5 most informative)
    sub_counts = []
    for phrase in _key_subphrases(query)[:8]:
        if phrase.strip().lower() == query.strip().lower():
            continue
        cnt = _infinigram_count(phrase)
        if cnt > 0:
            sub_counts.append((phrase, cnt))
    # Sort by count descending, keep top 5
    sub_counts.sort(key=lambda x: -x[1])
    sub_counts = sub_counts[:5]

    return [{
        "exact_phrase": query,
        "exact_count": exact_count,
        "exact_docs": exact_docs,
        "sub_phrase_counts": sub_counts,
        "dataset": "dolma",
    }]


def search_dolma_local(query: str, dolma_path: Optional[str] = None) -> list[dict]:
    """
    Search a local copy of Dolma (if downloaded).
    Users can download subsets from: https://huggingface.co/datasets/allenai/dolma

    This does simple text matching over JSONL files.
    For production use, you'd build an embedding index.
    """
    if not dolma_path or not os.path.exists(dolma_path):
        return []

    results = []
    query_lower = query.lower()

    for jsonl_file in Path(dolma_path).rglob("*.jsonl*"):
        try:
            import gzip

            opener = gzip.open if str(jsonl_file).endswith(".gz") else open
            with opener(jsonl_file, "rt", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if len(results) >= 5:
                        return results
                    try:
                        record = json.loads(line)
                        text = record.get("text", "")
                        if query_lower in text.lower():
                            results.append(
                                {
                                    "text": text[:500],
                                    "source": record.get("source", str(jsonl_file.name)),
                                    "url": record.get("url", ""),
                                    "dataset": "dolma-local",
                                }
                            )
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue

    return results


# ---------------------------------------------------------------------------
# 2. PROVENANCE REPORT — Explain what data shaped this decision
# ---------------------------------------------------------------------------


def build_provenance_report(
    comment_body: str,
    label_result: dict,
    dolma_matches: list[dict],
    override_matches: list[dict],
) -> dict:
    """
    Build a "view source" provenance report for a classification decision.

    Returns a structured report showing:
    - The decision made
    - Training data that likely influenced it
    - Any local overrides that were active
    - Links to inspect/modify the data
    """
    report = {
        "decision": label_result,
        "provenance": {
            "model": {
                "name": "OLMo 2 1B Instruct",
                "org": "Allen AI",
                "license": "Apache 2.0",
                "training_data": "Dolma (3T tokens, fully open)",
                "model_card": "https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct",
                "data_card": "https://huggingface.co/datasets/allenai/dolma",
                "data_explorer": "https://huggingface.co/spaces/liujch1998/infini-gram",
            },
            "training_data_matches": dolma_matches,
            "local_overrides_applied": override_matches,
            "how_to_modify": {
                "add_examples": "Add entries to your local example bank (examples.jsonl) to steer classification",
                "remove_bias": "Mark training data patterns you disagree with as counter-examples",
                "retrain": "For deeper changes, fine-tune OLMo on your curated data using Allen AI's training recipes",
                "recipes_url": "https://github.com/allenai/OLMo",
            },
        },
    }
    return report


def format_provenance_markdown(report: dict) -> str:
    """Render provenance report as readable Markdown."""
    lines = ["## 🔍 View Source: Data Provenance Report\n"]

    # Model info
    model = report["provenance"]["model"]
    lines.append(f"**Model:** [{model['name']}]({model['model_card']})")
    lines.append(f"**Training Data:** [{model['training_data']}]({model['data_card']})")
    lines.append(f"**Explore Training Data:** [{model['data_explorer']}]({model['data_explorer']})")
    lines.append(f"**License:** {model['license']}\n")

    # Decision
    decision = report["decision"]
    for cat_key, val in decision.items():
        if isinstance(val, dict):
            lines.append(f"### Decision: {cat_key} → `{val.get('label', '?')}`")
            if val.get("reasoning"):
                lines.append(f"*{val['reasoning']}*")
            if val.get("evidence"):
                ev = val["evidence"]
                if isinstance(ev, list):
                    lines.append(f"Evidence from comment: {', '.join(f'`{e}`' for e in ev)}")
            lines.append("")

    # Training data matches — new multi-layer structure
    matches = report["provenance"]["training_data_matches"]
    if matches and isinstance(matches[0], dict) and "exact_phrase" in matches[0]:
        m = matches[0]
        exact_count = m.get("exact_count", 0)
        sub_counts = m.get("sub_phrase_counts", [])
        exact_docs = m.get("exact_docs", [])

        lines.append("### 📚 Why did the model label this? (Training Data)\n")
        lines.append(
            "*The model's label comes from patterns it absorbed during training. "
            "Here are three layers of evidence from Dolma — OLMo's 3-trillion-token training dataset:*\n"
        )

        # Layer 1: exact phrase
        lines.append(f"**Layer 1 — Exact phrase** `\"{m['exact_phrase']}\"`: "
                     f"appeared **{exact_count:,} time(s)** in all of Dolma.")
        if exact_count < 100:
            lines.append(
                f"> ⚠️ Very rare in training data ({exact_count} occurrences). "
                "The label is not driven by this exact phrasing — it comes from the sub-phrases below.\n"
            )
        else:
            lines.append("")

        # Layer 2: key sub-phrase counts
        if sub_counts:
            lines.append("**Layer 2 — Key phrases the model has seen many times:**\n")
            for phrase, cnt in sub_counts:
                bar = "█" * min(20, max(1, int(20 * cnt / max(c for _, c in sub_counts))))
                lines.append(f"- `\"{phrase}\"` — **{cnt:,}** occurrences `{bar}`")
            lines.append(
                "\n*The more times a phrase appears in hostile/aggressive contexts in training data, "
                "the stronger the model's association with that label.*\n"
            )

        # Layer 3: instruction fine-tuning note
        lines.append(
            "**Layer 3 — Instruction fine-tuning (Tulu2):**\n"
            "> OLMo 2 was fine-tuned on [Tulu2](https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture), "
            "a dataset of human-written examples that explicitly teaches the model what 'hostile', "
            "'constructive', etc. mean. This fine-tuning layer is often *more* responsible for the "
            "final label than the raw pretraining counts above — it's where the definition lives.\n"
        )

        # Sample docs
        if exact_docs:
            lines.append("**Sample passages from Dolma containing the exact phrase:**\n")
            for i, doc in enumerate(exact_docs, 1):
                text_preview = doc["text"][:300].replace("\n", " ")
                lines.append(f"**{i}. From `{doc['source']}`**")
                lines.append(f"> {text_preview}{'...' if len(doc['text']) > 300 else ''}")
                if doc.get("url"):
                    lines.append(f"[Original source]({doc['url']})")
                lines.append("")

    elif matches:
        # Legacy flat format fallback
        lines.append("### 📚 What did the AI learn from?\n")
        for i, m in enumerate(matches, 1):
            source = m.get("source", "unknown")
            text_preview = m.get("text", "")[:300].replace("\n", " ")
            lines.append(f"**{i}. From: `{source}`**")
            lines.append(f"> {text_preview}")
            lines.append("")
    else:
        lines.append("### 📚 What did the AI learn from?\n")
        lines.append(
            "*We couldn't automatically fetch matching training texts, but you can explore them yourself:*\n"
        )
        lines.append(
            '- 🔎 <a href="https://huggingface.co/spaces/liujch1998/infini-gram" '
            'title="infini-gram — a live search engine over Dolma, the 3-trillion-token dataset OLMo was trained on. Paste the comment text to find the actual passages the model learned this type of language from.">'
            '**Search the texts the AI was trained on**</a> — paste the comment text to find matching training examples <em>(infini-gram / Dolma)</em>'
        )
        lines.append(
            '- 📥 <a href="https://huggingface.co/datasets/allenai/dolma" '
            'title="The full Dolma dataset on Hugging Face — OLMo&#39;s open training corpus. Download subsets for local search.">'
            '**Download the training dataset**</a> — full Dolma corpus for local search\n'
        )

    # Local overrides
    overrides = report["provenance"]["local_overrides_applied"]
    if overrides:
        lines.append("### ✏️ Local Overrides Applied\n")
        lines.append(
            "*These examples from your local bank were injected as context "
            "to steer the model's decision.*\n"
        )
        for o in overrides:
            lines.append(f"- **{o.get('label', '?')}**: \"{o['text'][:200]}\"")
        lines.append("")

    # How to modify
    how = report["provenance"]["how_to_modify"]
    lines.append("### 🔧 How to Change This\n")
    lines.append(f"- **Add examples:** {how['add_examples']}")
    lines.append(f"- **Counter biases:** {how['remove_bias']}")
    lines.append(f"- **Deep changes:** {how['retrain']}")
    lines.append(f"- **Training recipes:** [{how['recipes_url']}]({how['recipes_url']})")

    return "\n".join(lines)


def build_provenance_chart(provenance_reports: list[dict]):
    """
    Build a matplotlib figure with two panels:
      Left  — horizontal bar chart: how many times key phrases appear in Dolma
      Right — donut chart: which sources (reddit, common-crawl, etc.) those
              training examples came from

    Returns a matplotlib Figure, or None if there is no data to show.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from collections import Counter
    except ImportError:
        return None

    # Collect sub-phrase counts and source distribution across all reports
    all_sub_counts: dict[str, int] = {}
    source_counter: Counter = Counter()

    for pr in provenance_reports:
        matches = pr["report"]["provenance"]["training_data_matches"]
        if not matches or not isinstance(matches[0], dict):
            continue
        m = matches[0]
        # Sub-phrase counts
        for phrase, cnt in m.get("sub_phrase_counts", []):
            if phrase not in all_sub_counts or all_sub_counts[phrase] < cnt:
                all_sub_counts[phrase] = cnt
        # Exact phrase too
        ep = m.get("exact_phrase", "")
        ec = m.get("exact_count", 0)
        if ep and ec:
            all_sub_counts[ep] = ec
        # Source distribution from sample docs
        for doc in m.get("exact_docs", []):
            src = doc.get("source", "unknown")
            # Normalise — collapse subreddit detail to top-level source
            if src.startswith("reddit"):
                src = "reddit"
            source_counter[src] += 1

    if not all_sub_counts:
        return None

    # Sort phrases by count ascending (so largest bar is at top)
    sorted_phrases = sorted(all_sub_counts.items(), key=lambda x: x[1])
    phrases = [p for p, _ in sorted_phrases]
    counts  = [c for _, c in sorted_phrases]

    fig, axes = plt.subplots(
        1, 2 if source_counter else 1,
        figsize=(11, max(3, len(phrases) * 0.55 + 1.5)),
        facecolor="#0f0f0f",
    )
    ax_bar = axes[0] if source_counter else axes
    ax_pie = axes[1] if source_counter else None

    # ── Left: horizontal bar chart ──────────────────────────────────────────
    colors = ["#4a90d9" if c < 10_000 else "#e8734a" if c < 1_000_000 else "#d94a4a"
              for c in counts]
    bars = ax_bar.barh(phrases, counts, color=colors, height=0.6)
    ax_bar.set_facecolor("#1a1a1a")
    ax_bar.set_xlabel("Occurrences in Dolma training data", color="#cccccc", fontsize=9)
    ax_bar.set_title("How often key phrases\nappeared in training data",
                     color="#ffffff", fontsize=10, pad=8)
    ax_bar.tick_params(colors="#cccccc", labelsize=8)
    ax_bar.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K" if x >= 1000 else str(int(x))
    ))
    ax_bar.spines[:].set_color("#333333")
    for bar, cnt in zip(bars, counts):
        label = f"{cnt/1e6:.1f}M" if cnt >= 1e6 else f"{cnt/1e3:.0f}K" if cnt >= 1000 else str(cnt)
        ax_bar.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    label, va="center", ha="left", color="#cccccc", fontsize=8)
    ax_bar.set_xlim(right=max(counts) * 1.18)

    # Colour legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4a90d9", label="Rare (< 10K)"),
        Patch(facecolor="#e8734a", label="Common (10K–1M)"),
        Patch(facecolor="#d94a4a", label="Very common (> 1M)"),
    ]
    ax_bar.legend(handles=legend_elements, loc="lower right",
                  fontsize=7, facecolor="#222222", labelcolor="#cccccc",
                  edgecolor="#444444")

    # ── Right: source donut chart ────────────────────────────────────────────
    if ax_pie is not None and source_counter:
        src_labels = list(source_counter.keys())
        src_values = [source_counter[s] for s in src_labels]
        palette = ["#4a90d9", "#e8734a", "#5cb85c", "#d94a4a", "#9b59b6",
                   "#f39c12", "#1abc9c"]
        ax_pie.pie(
            src_values,
            labels=src_labels,
            autopct="%1.0f%%",
            colors=palette[:len(src_labels)],
            wedgeprops={"width": 0.5, "edgecolor": "#0f0f0f", "linewidth": 1.5},
            textprops={"color": "#cccccc", "fontsize": 8},
            pctdistance=0.75,
        )
        ax_pie.set_facecolor("#1a1a1a")
        ax_pie.set_title("Where those training\nexamples came from",
                         color="#ffffff", fontsize=10, pad=8)

    fig.patch.set_facecolor("#0f0f0f")
    fig.tight_layout(pad=1.5)
    return fig


# ---------------------------------------------------------------------------
# 3. LOCAL EXAMPLE BANK — Your "edit source" layer
# ---------------------------------------------------------------------------


class ExampleBank:
    """
    A local, editable collection of labeled examples that get injected
    into the model's prompt as few-shot context.

    This is your "edit source" — by adding, removing, or changing examples,
    you directly steer the model's classification behavior.

    Stored as a simple JSONL file so it's human-readable and version-controllable.
    """

    def __init__(self, path: str = "examples.jsonl"):
        self.path = path
        self.examples: list[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.examples.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

    def save(self):
        with open(self.path, "w") as f:
            for ex in self.examples:
                f.write(json.dumps(ex) + "\n")

    def add(self, text: str, labels: dict, source: str = "user", notes: str = ""):
        """
        Add an example to the bank.

        Args:
            text: The comment text
            labels: Dict mapping category keys to labels, e.g. {"toxicity": "hostile"}
            source: Where this example came from ("user", "dolma", "github", etc.)
            notes: Why you added this — your reasoning
        """
        entry = {
            "text": text,
            "labels": labels,
            "source": source,
            "notes": notes,
        }
        self.examples.append(entry)
        self.save()
        return entry

    def remove(self, index: int):
        """Remove an example by index."""
        if 0 <= index < len(self.examples):
            removed = self.examples.pop(index)
            self.save()
            return removed
        return None

    def update(self, index: int, labels: dict = None, notes: str = None):
        """Update labels or notes for an existing example."""
        if 0 <= index < len(self.examples):
            if labels:
                self.examples[index]["labels"] = labels
            if notes is not None:
                self.examples[index]["notes"] = notes
            self.save()
            return self.examples[index]
        return None

    def find_relevant(self, comment: str, max_results: int = 3) -> list[dict]:
        """
        Find examples from the bank that are relevant to a given comment.
        Uses simple keyword overlap. For production, use embeddings.
        """
        comment_words = set(re.findall(r"\w+", comment.lower()))
        scored = []
        for i, ex in enumerate(self.examples):
            ex_words = set(re.findall(r"\w+", ex["text"].lower()))
            overlap = len(comment_words & ex_words)
            if overlap > 0:
                scored.append((overlap, i, ex))
        scored.sort(reverse=True)
        return [ex for _, _, ex in scored[:max_results]]

    def get_few_shot_prompt(self, comment: str, config_categories: dict) -> str:
        """
        Build a few-shot prompt section from relevant examples.
        This gets injected into the model's prompt to steer its behavior.
        """
        relevant = self.find_relevant(comment)
        if not relevant:
            return ""

        lines = [
            "\nHere are reference examples from a curated example bank. "
            "Use these to calibrate your classifications:\n"
        ]
        for ex in relevant:
            labels_str = json.dumps(ex["labels"])
            text_preview = ex["text"][:300]
            lines.append(f'Comment: """{text_preview}"""')
            lines.append(f"Classification: {labels_str}")
            if ex.get("notes"):
                lines.append(f"Note: {ex['notes']}")
            lines.append("")

        return "\n".join(lines)

    def to_display_list(self) -> list[dict]:
        """Return examples formatted for UI display."""
        return [
            {
                "index": i,
                "text_preview": ex["text"][:100] + ("..." if len(ex["text"]) > 100 else ""),
                "labels": ex["labels"],
                "source": ex.get("source", "unknown"),
                "notes": ex.get("notes", ""),
            }
            for i, ex in enumerate(self.examples)
        ]

    def export_json(self) -> str:
        return json.dumps(self.examples, indent=2)

    def import_json(self, data: str):
        """Import examples from JSON string."""
        imported = json.loads(data)
        if isinstance(imported, list):
            self.examples.extend(imported)
            self.save()
            return len(imported)
        return 0


# ---------------------------------------------------------------------------
# Default seed examples to get started
# ---------------------------------------------------------------------------

SEED_EXAMPLES = [
    {
        "text": "This is terrible code. Did you even test this? I can't believe this was merged.",
        "labels": {"toxicity": "hostile", "constructiveness": "unconstructive"},
        "source": "seed",
        "notes": "Hostile tone with no actionable feedback.",
    },
    {
        "text": "I think there's a bug on line 42 — the null check should happen before the array access. Here's a fix: ...",
        "labels": {"toxicity": "respectful", "constructiveness": "constructive"},
        "source": "seed",
        "notes": "Identifies specific issue with proposed solution.",
    },
    {
        "text": "This approach won't scale. Have you considered using a hashmap instead? See O(n) vs O(1) lookup discussion here: ...",
        "labels": {"toxicity": "neutral", "constructiveness": "constructive"},
        "source": "seed",
        "notes": "Direct but professional criticism with alternative.",
    },
    {
        "text": "Why do we even have this feature? Nobody asked for it. Complete waste of everyone's time.",
        "labels": {"toxicity": "dismissive", "constructiveness": "unconstructive"},
        "source": "seed",
        "notes": "Dismissive without offering alternatives or context.",
    },
    {
        "text": "+1, works for me. Thanks for the fix!",
        "labels": {"toxicity": "respectful", "constructiveness": "mixed"},
        "source": "seed",
        "notes": "Positive but minimal actionable content.",
    },
]


def initialize_example_bank(path: str = "examples.jsonl") -> ExampleBank:
    """Create an example bank with seed examples if it doesn't exist."""
    bank = ExampleBank(path)
    if not bank.examples:
        for seed in SEED_EXAMPLES:
            bank.examples.append(seed)
        bank.save()
    return bank

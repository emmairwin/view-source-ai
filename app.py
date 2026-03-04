"""
GitHub Comment Sentiment Agent
==============================
Uses OLMo 2 1B Instruct (Allen AI) — a fully open-source LLM
trained on the Dolma dataset (open, documented training data).

License: Apache 2.0
Model: https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct
Training Data: https://huggingface.co/datasets/allenai/dolma
"""

import json
import os
import re
import gradio as gr
import requests
from dataclasses import dataclass, field
from typing import Optional

# Pre-import transformers pipeline at module level — avoids thread-safety issues
# with lazy imports in transformers 5.x when called from Gradio worker threads.
try:
    from transformers import pipeline as _hf_pipeline_fn
except ImportError:
    _hf_pipeline_fn = None

from data_provenance import (
    ExampleBank,
    build_provenance_chart,
    build_provenance_report,
    format_provenance_markdown,
    initialize_example_bank,
    search_dolma_wimbd,
    search_dolma_local,
)


# ---------------------------------------------------------------------------
# Upstream Contribution Targets
# ---------------------------------------------------------------------------

UPSTREAM_TARGETS = {
    "model_behavior": {
        "name": "Model Behavior (Misclassification / Bias)",
        "repo": "allenai/OLMo",
        "description": "The model consistently misclassifies a type of comment",
        "new_issue_url": "https://github.com/allenai/OLMo/issues/new",
        "email": "olmo@allenai.org",
    },
    "training_data": {
        "name": "Training Data (Dolma)",
        "repo": "allenai/dolma",
        "description": "Problematic or missing data in the training corpus",
        "new_issue_url": "https://github.com/allenai/dolma/issues/new",
        "email": "olmo@allenai.org",
    },
    "post_training": {
        "name": "Post-Training / Instruction Tuning (Open Instruct / Tülu)",
        "repo": "allenai/open-instruct",
        "description": "Improve instruction-following or classification via better training examples",
        "new_issue_url": "https://github.com/allenai/open-instruct/issues/new",
        "email": "olmo@allenai.org",
    },
    "olmotrace": {
        "name": "OLMoTrace (Data Tracing Tool)",
        "repo": "allenai/OLMoTrace",
        "description": "Feedback on the training data tracing/attribution system",
        "new_issue_url": "https://github.com/allenai/OLMoTrace/issues/new",
        "email": "olmo@allenai.org",
    },
    "training_framework": {
        "name": "Training Framework (OLMo-core)",
        "repo": "allenai/OLMo-core",
        "description": "Training code, recipes, architecture suggestions",
        "new_issue_url": "https://github.com/allenai/OLMo-core/issues/new",
        "email": "olmo@allenai.org",
    },
}


def generate_issue_body(
    target_key: str,
    issue_type: str,
    title: str,
    description: str,
    example_comment: str,
    expected_label: str,
    actual_label: str,
    evidence_notes: str,
) -> tuple[str, str, str]:
    """
    Generate a well-formatted GitHub issue body for upstream contribution.
    Returns (formatted_issue_markdown, github_url_with_prefill, raw_text).
    """
    target = UPSTREAM_TARGETS.get(target_key, UPSTREAM_TARGETS["model_behavior"])

    # Build the issue body
    body_parts = []

    if issue_type == "Misclassification Report":
        body_parts.append(f"## Misclassification Report\n")
        body_parts.append(f"**Reported by:** GitHub Sentiment Agent user")
        body_parts.append(f"**Model:** OLMo 2 1B Instruct")
        body_parts.append(f"**Task:** GitHub comment sentiment classification\n")
        body_parts.append(f"### Description\n{description}\n")
        if example_comment:
            body_parts.append(f"### Example Comment\n```\n{example_comment[:1000]}\n```\n")
        if expected_label or actual_label:
            body_parts.append(f"### Classification")
            body_parts.append(f"- **Expected label:** {expected_label or 'N/A'}")
            body_parts.append(f"- **Actual label:** {actual_label or 'N/A'}\n")
        if evidence_notes:
            body_parts.append(f"### Additional Context\n{evidence_notes}\n")

    elif issue_type == "Training Data Suggestion":
        body_parts.append(f"## Training Data Suggestion\n")
        body_parts.append(f"**Reported by:** GitHub Sentiment Agent user")
        body_parts.append(f"**Relevant dataset:** Dolma / Tülu post-training\n")
        body_parts.append(f"### Description\n{description}\n")
        if example_comment:
            body_parts.append(f"### Example Data\n```\n{example_comment[:1000]}\n```\n")
        if evidence_notes:
            body_parts.append(f"### Why This Matters\n{evidence_notes}\n")

    elif issue_type == "Feature Request":
        body_parts.append(f"## Feature Request\n")
        body_parts.append(f"**Reported by:** GitHub Sentiment Agent user\n")
        body_parts.append(f"### Description\n{description}\n")
        if evidence_notes:
            body_parts.append(f"### Use Case\n{evidence_notes}\n")

    else:  # General Feedback
        body_parts.append(f"## Feedback\n")
        body_parts.append(f"**Reported by:** GitHub Sentiment Agent user")
        body_parts.append(f"**Model:** OLMo 2 1B Instruct\n")
        body_parts.append(f"### Description\n{description}\n")
        if example_comment:
            body_parts.append(f"### Example\n```\n{example_comment[:1000]}\n```\n")
        if evidence_notes:
            body_parts.append(f"### Additional Context\n{evidence_notes}\n")

    body_parts.append("---")
    body_parts.append(
        "*This issue was drafted using the [GitHub Sentiment Agent]"
        "(https://github.com/allenai/OLMo) — "
        "a tool for auditable AI-powered comment analysis built on OLMo.*"
    )

    full_body = "\n".join(body_parts)

    # Build the GitHub new issue URL with pre-filled content
    import urllib.parse
    params = urllib.parse.urlencode({
        "title": title or f"[Sentiment Agent] {issue_type}",
        "body": full_body,
    })
    github_url = f"{target['new_issue_url']}?{params}"

    # Build a nice preview
    preview_md = f"""### 📋 Issue Preview

**Target repo:** [`{target['repo']}`]({target['new_issue_url']})
**Type:** {issue_type}
**Title:** {title}

---

{full_body}

---

### 🚀 Ready to Submit?

**Option 1 — Open on GitHub:**
[Click here to open this issue on GitHub]({github_url})
*(This will pre-fill the title and body — you just click "Submit")*

**Option 2 — Email the team:**
Send to `{target['email']}` with the content above.

**Option 3 — Copy and paste:**
Copy the issue body above and paste it into [{target['repo']}]({target['new_issue_url']}).
"""

    return preview_md, github_url, full_body


def save_issue_draft(preview_md: str, github_url: str, full_body: str) -> str:
    """Save the draft to a local file for reference."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    draft_path = f"issue_drafts/draft_{timestamp}.md"
    os.makedirs("issue_drafts", exist_ok=True)
    with open(draft_path, "w") as f:
        f.write(full_body)
    return f"✅ Draft saved to `{draft_path}`"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_LABELS = {
    "toxicity": {
        "name": "Toxicity / Hostility",
        "labels": ["toxic", "hostile", "dismissive", "neutral", "respectful"],
        "description": "Detects toxic, hostile, or dismissive language in comments.",
    },
    "constructiveness": {
        "name": "Constructiveness",
        "labels": ["constructive", "unconstructive", "mixed"],
        "description": "Whether the comment provides actionable, helpful feedback.",
    },
}


@dataclass
class SentimentConfig:
    """Holds user-defined sentiment categories and labels."""
    categories: dict = field(default_factory=lambda: dict(DEFAULT_LABELS))

    def add_category(self, key: str, name: str, labels: list[str], description: str = ""):
        self.categories[key] = {
            "name": name,
            "labels": labels,
            "description": description,
        }

    def remove_category(self, key: str):
        self.categories.pop(key, None)

    def get_prompt_section(self) -> str:
        lines = []
        for key, cat in self.categories.items():
            label_str = ", ".join(f'"{l}"' for l in cat["labels"])
            lines.append(
                f'- **{cat["name"]}** (key: "{key}"): '
                f'Choose exactly one of [{label_str}]. {cat.get("description", "")}'
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# GitHub API
# ---------------------------------------------------------------------------

def fetch_github_comments(repo: str, issue_number: int, token: Optional[str] = None) -> list[dict]:
    """Fetch comments from a GitHub issue or PR."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Try issue comments
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    comments = resp.json()

    # Also try PR review comments
    pr_url = f"https://api.github.com/repos/{repo}/pulls/{issue_number}/comments"
    try:
        pr_resp = requests.get(pr_url, headers=headers, timeout=30)
        if pr_resp.status_code == 200:
            comments.extend(pr_resp.json())
    except Exception:
        pass

    return [
        {
            "id": c["id"],
            "user": c["user"]["login"],
            "body": c["body"],
            "created_at": c["created_at"],
            "url": c.get("html_url", ""),
        }
        for c in comments
        if c.get("body", "").strip()
    ]


def parse_repo_input(raw: str) -> tuple[str, int]:
    """
    Parse various GitHub URL formats or 'owner/repo#123' shorthand.
    Returns (repo, issue_number).
    """
    raw = raw.strip()

    # Full URL: https://github.com/owner/repo/issues/123  (protocol optional)
    url_match = re.match(
        r"(?:https?://)?github\.com/([^/]+/[^/]+)/(?:issues|pull)/(\d+)", raw
    )
    if url_match:
        return url_match.group(1), int(url_match.group(2))

    # Shorthand: owner/repo#123
    short_match = re.match(r"([^#]+)#(\d+)", raw)
    if short_match:
        return short_match.group(1).strip(), int(short_match.group(2))

    # Looks like a repo URL but missing an issue/PR number
    repo_only = re.match(r"(?:https?://)?github\.com/([^/]+/[^/\s]+)/?$", raw)
    if repo_only:
        raise ValueError(
            f"Looks like a repo URL — please include a specific issue or PR number. "
            f"Example: {raw.rstrip('/')}/issues/1 or {repo_only.group(1)}#1"
        )

    raise ValueError(
        "Could not parse input. Use a GitHub issue/PR URL "
        "(e.g. github.com/owner/repo/issues/42) or 'owner/repo#42' shorthand."
    )


# ---------------------------------------------------------------------------
# LLM Inference (OLMo via llama.cpp / Ollama / HF Transformers)
# ---------------------------------------------------------------------------

def build_prompt(comment_body: str, config: SentimentConfig, example_bank=None) -> list[dict]:
    """
    Build a chat-template prompt for OLMo Instruct.
    Returns a list of message dicts [{"role": ..., "content": ...}].
    We prime the assistant response with '{' so the model only has to complete the JSON.
    """
    # Build the exact JSON skeleton the model should fill in
    skeleton_parts = []
    for key, cat in config.categories.items():
        labels_str = ", ".join(f'"{l}"' for l in cat["labels"])
        skeleton_parts.append(
            f'  "{key}": {{"label": <one of [{labels_str}]>, "reasoning": <why>, "evidence": [<phrases>]}}'
        )
    skeleton = "{\n" + ",\n".join(skeleton_parts) + "\n}"

    # Few-shot examples from the bank
    few_shot = ""
    if example_bank:
        raw = example_bank.get_few_shot_prompt(comment_body, config.categories)
        if raw.strip():
            few_shot = f"\n\nReference examples to calibrate your labels:\n{raw.strip()}"

    user_content = (
        f"Classify the following GitHub comment. "
        f"Reply ONLY with a JSON object matching this exact structure — no prose, no markdown fences:{few_shot}\n\n"
        f"Structure:\n{skeleton}\n\n"
        f'Comment to classify:\n"""{comment_body[:1500]}"""'
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a precise GitHub comment sentiment classifier. "
                "You always respond with valid JSON only. No explanation outside the JSON."
            ),
        },
        {"role": "user", "content": user_content},
    ]


class OlmoBackend:
    """
    Connects to OLMo 2 1B Instruct via one of these backends (auto-detected):
    1. Ollama (local server on port 11434)
    2. HuggingFace transformers (direct loading)

    You can override with OLMO_BACKEND=ollama or OLMO_BACKEND=transformers.
    """

    MODEL_NAME_OLLAMA = os.environ.get("OLMO_MODEL", "olmo2:1b")
    MODEL_NAME_HF = "allenai/OLMo-2-0425-1B-Instruct"

    def __init__(self):
        import threading
        self.backend = os.environ.get("OLMO_BACKEND", "auto")
        self._hf_pipeline = None
        self._load_lock = threading.Lock()  # prevent double-load race between prewarm & first click

    def _try_ollama(self, messages: list) -> Optional[str]:
        """Try Ollama chat API."""
        try:
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.MODEL_NAME_OLLAMA,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 512},
                },
                timeout=120,
            )
            if resp.status_code == 200:
                return resp.json().get("message", {}).get("content", "")
        except requests.ConnectionError:
            return None
        return None

    def _try_transformers(self, messages: list) -> str:
        """Load model via HuggingFace transformers and run chat inference."""
        with self._load_lock:
            if self._hf_pipeline is None:
                if _hf_pipeline_fn is None:
                    raise RuntimeError("transformers package is not installed")
                print(f"Loading {self.MODEL_NAME_HF} via transformers (first run may download ~2GB)...")
                self._hf_pipeline = _hf_pipeline_fn(
                    "text-generation",
                    model=self.MODEL_NAME_HF,
                    device_map="auto",
                    dtype="auto",
                )
                # Clear stored generation_config constraints that conflict with our kwargs
                cfg = self._hf_pipeline.model.generation_config
                cfg.max_length = None
                cfg.max_new_tokens = None
        # The pipeline accepts a list of message dicts and applies the chat template automatically
        result = self._hf_pipeline(
            messages,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            return_full_text=False,
        )
        return result[0]["generated_text"]

    def generate(self, messages: list) -> str:
        """Run inference. messages is a list of {role, content} dicts."""
        if self.backend in ("auto", "ollama"):
            result = self._try_ollama(messages)
            if result is not None:
                return result
            if self.backend == "ollama":
                raise RuntimeError(
                    "Ollama not reachable. Start it with: ollama serve && ollama pull olmo2:1b"
                )

        if self.backend in ("auto", "transformers"):
            return self._try_transformers(messages)

        raise RuntimeError(f"Unknown backend: {self.backend}")


# ---------------------------------------------------------------------------
# Module-level singletons — model loads once per process, not per click
# ---------------------------------------------------------------------------

_llm: Optional["OlmoBackend"] = None
_example_bank: Optional["ExampleBank"] = None


def get_llm() -> "OlmoBackend":
    global _llm
    if _llm is None:
        _llm = OlmoBackend()
    return _llm


def get_example_bank() -> "ExampleBank":
    global _example_bank
    if _example_bank is None:
        _example_bank = initialize_example_bank()
    return _example_bank


def prewarm_model():
    """Load the model into memory at startup so the first Analyze click is instant."""
    import threading
    def _load():
        print("Pre-warming OLMo model (this downloads ~2GB on first run, then stays loaded)...")
        try:
            llm = get_llm()
            llm.generate([{"role": "user", "content": "warmup"}])
            print("OLMo model ready.")
        except Exception as e:
            print(f"Pre-warm failed (will retry on first Analyze click): {e}")
    t = threading.Thread(target=_load, daemon=True)
    t.start()


def parse_llm_response(raw: str, config: SentimentConfig) -> dict:
    """Extract classification labels from LLM response, handling multiple output formats."""
    unknown = {
        key: {"label": "unknown", "reasoning": "", "evidence": []}
        for key in config.categories
    }

    if not raw or not raw.strip():
        return unknown

    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", raw).strip()

    # --- Attempt 1: find outermost JSON object (handles flat or nested values) ---
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            result = {}
            matched_any = False
            for key in config.categories:
                if key in parsed:
                    val = parsed[key]
                    matched_any = True
                    if isinstance(val, dict):
                        result[key] = {
                            "label": str(val.get("label", "unknown")).lower().strip(),
                            "reasoning": val.get("reasoning", ""),
                            "evidence": val.get("evidence", []),
                        }
                    else:
                        result[key] = {"label": str(val).lower().strip(), "reasoning": "", "evidence": []}
                else:
                    result[key] = {"label": "unknown", "reasoning": "", "evidence": []}
            if matched_any:
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # --- Attempt 2: model emitted one {"label": ...} block per category in order ---
    # Find all top-level JSON objects in the response
    objects = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start:i + 1])
                    objects.append(obj)
                except (json.JSONDecodeError, ValueError):
                    pass
                start = None

    if objects:
        result = {}
        cat_keys = list(config.categories.keys())
        for idx, key in enumerate(cat_keys):
            obj = objects[idx] if idx < len(objects) else {}
            label = (
                obj.get("label")
                or obj.get(key, {}).get("label") if isinstance(obj.get(key), dict) else None
                or "unknown"
            )
            result[key] = {
                "label": str(label).lower().strip(),
                "reasoning": obj.get("reasoning", ""),
                "evidence": obj.get("evidence", []),
            }
        return result

    # --- Attempt 3: plain text labels  ("toxicity: hostile") ---
    result = {}
    all_labels = set()
    for cat in config.categories.values():
        all_labels.update(cat["labels"])
    for key, cat in config.categories.items():
        for label in cat["labels"]:
            if re.search(rf"\b{re.escape(label)}\b", raw, re.IGNORECASE):
                result[key] = {"label": label, "reasoning": "(extracted from plain text)", "evidence": []}
                break
        if key not in result:
            result[key] = {"label": "unknown", "reasoning": "", "evidence": []}
    return result


# ---------------------------------------------------------------------------
# Main Analysis Pipeline
# ---------------------------------------------------------------------------

def analyze_comments(
    repo_input: str,
    github_token: str,
    custom_categories_json: str,
    dolma_path: str,
):
    """Main entry point: fetch comments, classify each, stream results via yield."""

    def _status(msg: str):
        """Yield a status-only update, leaving outputs blank until done."""
        return f"⏳ {msg}", "", "", "", None

    # Parse config
    config = SentimentConfig()
    if custom_categories_json.strip():
        try:
            custom = json.loads(custom_categories_json)
            for key, cat in custom.items():
                config.add_category(
                    key=key,
                    name=cat.get("name", key),
                    labels=cat["labels"],
                    description=cat.get("description", ""),
                )
        except (json.JSONDecodeError, KeyError) as e:
            yield f"❌ Invalid custom categories JSON: {e}", "", "", "", None
            return

    # Parse repo
    try:
        repo, issue_num = parse_repo_input(repo_input)
    except ValueError as e:
        yield f"❌ {e}", "", "", "", None
        return

    yield _status(f"Fetching comments from {repo}#{issue_num}...")

    # Fetch comments
    try:
        comments = fetch_github_comments(repo, issue_num, github_token or None)
    except requests.HTTPError as e:
        yield f"❌ GitHub API error: {e}", "", "", "", None
        return

    if not comments:
        yield "⚠️ No comments found on this issue/PR.", "", "", "", None
        return

    yield _status(f"Fetched {len(comments)} comment(s). Loading OLMo 2 1B Instruct...")

    # Use module-level singletons — model stays loaded between clicks
    llm = get_llm()
    example_bank = get_example_bank()

    # Classify each comment
    results = []
    provenance_reports = []
    for i, comment in enumerate(comments):
        yield _status(f"Analyzing comment {i+1} of {len(comments)} (@{comment['user']})... this may take ~30s per comment on CPU")
        prompt = build_prompt(comment["body"], config, example_bank)
        try:
            raw_response = llm.generate(prompt)
            labels = parse_llm_response(raw_response, config)
        except Exception as e:
            labels = {
                key: {"label": "error", "reasoning": str(e), "evidence": []}
                for key in config.categories
            }

        results.append({**comment, "labels": labels})

        # Search training data for provenance
        override_matches = example_bank.find_relevant(comment["body"])
        dolma_matches = search_dolma_wimbd(comment["body"][:200])
        if not dolma_matches and dolma_path.strip():
            dolma_matches = search_dolma_local(comment["body"][:200], dolma_path.strip())

        prov = build_provenance_report(comment["body"], labels, dolma_matches, override_matches)
        provenance_reports.append({"comment_user": comment["user"], "report": prov})

    # Format outputs
    markdown = format_results_markdown(results, config, repo, issue_num)
    json_output = json.dumps(results, indent=2, default=str)

    # Build combined provenance markdown
    prov_md_parts = ["# 🔍 Data Provenance: View Source\n"]
    prov_md_parts.append(
        "*For each comment, here's what data shaped the model's decision "
        "and how you can change it.*\n---\n"
    )
    for pr in provenance_reports:
        prov_md_parts.append(f"## Comment by @{pr['comment_user']}\n")
        prov_md_parts.append(format_provenance_markdown(pr["report"]))
        prov_md_parts.append("\n---\n")
    provenance_md = "\n".join(prov_md_parts)

    provenance_chart = build_provenance_chart(provenance_reports)
    yield f"✅ Done! Analyzed {len(results)} comment(s).", markdown, json_output, provenance_md, provenance_chart


def format_results_markdown(
    results: list[dict], config: SentimentConfig, repo: str, issue_num: int
) -> str:
    """Format results as a readable Markdown report with reasoning."""
    lines = [
        f"# Sentiment Analysis: {repo}#{issue_num}",
        f"**Model:** OLMo 2 1B Instruct (Allen AI) — fully open-source, trained on Dolma",
        f"**Comments analyzed:** {len(results)}",
        f"**Categories:** {', '.join(cat['name'] for cat in config.categories.values())}",
        "",
        "---",
        "",
    ]

    # Define which labels are "flagged" (negative/concerning)
    flagged_labels = {
        "toxic", "hostile", "dismissive", "unconstructive", "critical",
        "hateful", "aggressive", "abusive", "negative", "harmful",
    }

    # Summary stats
    lines.append("## Summary")
    for key, cat in config.categories.items():
        lines.append(f"\n### {cat['name']}")
        label_counts = {}
        for r in results:
            label = r["labels"].get(key, {}).get("label", "unknown")
            label_counts[label] = label_counts.get(label, 0) + 1
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            pct = count / len(results) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            flag = " ⚠️" if label in flagged_labels else ""
            lines.append(f"  {bar} **{label}**: {count} ({pct:.0f}%){flag}")

    # Flagged comments section (toxic/hostile/etc get special treatment)
    flagged_comments = []
    for r in results:
        for key, val in r["labels"].items():
            if val.get("label", "") in flagged_labels:
                flagged_comments.append((r, key, val))

    if flagged_comments:
        lines.append("\n---\n## ⚠️ Flagged Comments\n")
        lines.append("*These comments were classified with concerning labels. "
                      "The model explains its reasoning and highlights the specific "
                      "language that triggered the flag.*\n")
        for r, cat_key, val in flagged_comments:
            cat_name = config.categories[cat_key]["name"]
            body_preview = r["body"][:300].replace("\n", " ")
            if len(r["body"]) > 300:
                body_preview += "..."

            lines.append(f"### @{r['user']} — {r['created_at'][:10]}")
            lines.append(f"> {body_preview}\n")
            lines.append(f"**{cat_name}:** 🔴 `{val['label']}`\n")

            if val.get("reasoning"):
                lines.append(f"**Why this was flagged:** {val['reasoning']}\n")

            if val.get("evidence"):
                evidence_list = val["evidence"]
                if isinstance(evidence_list, list) and evidence_list:
                    highlighted = ", ".join(f'`"{e}"`' for e in evidence_list)
                    lines.append(f"**Triggering language:** {highlighted}\n")

            lines.append('> 📊 **See the "View Source (Provenance)" tab** for a full breakdown of what training data shaped this label.\n')
            if r.get("url"):
                lines.append(f"[View on GitHub]({r['url']})")
            lines.append("")

    # Per-comment detail
    lines.append("\n---\n## All Comments\n")
    for r in results:
        body_preview = r["body"][:200].replace("\n", " ")
        if len(r["body"]) > 200:
            body_preview += "..."
        lines.append(f"### @{r['user']} — {r['created_at'][:10]}")
        lines.append(f"> {body_preview}\n")

        for k, val in r["labels"].items():
            cat_name = config.categories[k]["name"]
            label = val.get("label", "unknown")
            is_flagged = label in flagged_labels
            icon = "🔴" if is_flagged else "🟢"
            lines.append(f"  {icon} **{cat_name}:** `{label}`")

            # Show reasoning for all comments, not just flagged
            if val.get("reasoning"):
                lines.append(f"    *Reasoning: {val['reasoning']}*")
            if val.get("evidence") and isinstance(val["evidence"], list) and val["evidence"]:
                highlighted = ", ".join(f'"{e}"' for e in val["evidence"])
                lines.append(f"    *Evidence: {highlighted}*")

        if r.get("url"):
            lines.append(f"\n[View on GitHub]({r['url']})")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLE_CUSTOM_JSON = json.dumps(
    {
        "urgency": {
            "name": "Urgency",
            "labels": ["critical", "important", "low-priority", "informational"],
            "description": "How urgent is the issue raised in this comment?",
        }
    },
    indent=2,
)

INFO_MD = """
## About This Agent

**Model:** [OLMo 2 1B Instruct](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct) by Allen AI
**Training Data:** [Dolma](https://huggingface.co/datasets/allenai/dolma) — fully open and documented
**License:** Apache 2.0

### The "View Source" Philosophy

Traditional software has View Source — you can see exactly what makes it work and change it.
This agent brings that to AI:

1. **TRACE** — When the model makes a decision, see what training data likely influenced it
2. **INSPECT** — Every classification comes with reasoning and evidence
3. **OVERRIDE** — Edit the local example bank to steer the model's behavior
4. **CONTRIBUTE** — Draft and file issues directly to Allen AI's open repos to improve the model for everyone

### Contribute Upstream

Unlike closed models where feedback goes into a void, OLMo's open nature means you can:
- File misclassification reports on [allenai/OLMo](https://github.com/allenai/OLMo)
- Suggest training data improvements on [allenai/dolma](https://github.com/allenai/dolma)
- Contribute post-training examples on [allenai/open-instruct](https://github.com/allenai/open-instruct)
- Use [OLMoTrace](https://playground.allenai.org) to find the training data behind a decision
- Email the team at olmo@allenai.org

### Backends (auto-detected)
1. **Ollama** (recommended): `ollama pull olmo2:1b && ollama serve`
2. **HuggingFace Transformers**: Auto-downloads on first run (~2GB)

Set `OLMO_BACKEND=ollama` or `OLMO_BACKEND=transformers` to force one.
"""


# ---------------------------------------------------------------------------
# Example Bank UI Helpers
# ---------------------------------------------------------------------------

_bank = initialize_example_bank()


def get_bank_display():
    items = _bank.to_display_list()
    if not items:
        return "No examples in bank yet."
    lines = []
    for item in items:
        labels = ", ".join(f"{k}: **{v}**" for k, v in item["labels"].items())
        lines.append(
            f"**[{item['index']}]** ({item['source']}) {labels}\n"
            f"> {item['text_preview']}\n"
            f"*{item['notes']}*\n"
        )
    return "\n---\n".join(lines)


def add_example(text, labels_json, source, notes):
    try:
        labels = json.loads(labels_json)
    except json.JSONDecodeError:
        return "❌ Invalid labels JSON", get_bank_display()
    _bank.add(text, labels, source, notes)
    return "✅ Example added", get_bank_display()


def remove_example(index):
    try:
        idx = int(index)
    except (ValueError, TypeError):
        return "❌ Enter a valid index number", get_bank_display()
    result = _bank.remove(idx)
    if result:
        return f"✅ Removed example {idx}", get_bank_display()
    return "❌ Index not found", get_bank_display()


def export_bank():
    return _bank.export_json()


def import_bank(data):
    try:
        count = _bank.import_json(data)
        return f"✅ Imported {count} examples", get_bank_display()
    except Exception as e:
        return f"❌ Import failed: {e}", get_bank_display()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(
        title="GitHub Sentiment Agent — OLMo 2 (View Source Edition)",
    ) as app:
        gr.Markdown("# 🔍 GitHub Comment Sentiment Agent")
        gr.Markdown(
            "*Powered by OLMo 2 1B Instruct — fully open model, fully open data, fully auditable*"
        )

        with gr.Tabs():
            # ---- TAB 1: ANALYZE ----
            with gr.Tab("🔬 Analyze"):
                with gr.Row():
                    with gr.Column(scale=2):
                        repo_input = gr.Textbox(
                            label="GitHub Issue / PR",
                            placeholder="https://github.com/owner/repo/issues/123  or  owner/repo#123",
                            info="Paste a full URL or use shorthand notation",
                        )
                        github_token = gr.Textbox(
                            label="GitHub Token (optional)",
                            placeholder="ghp_...",
                            type="password",
                            info="Needed for private repos or to avoid rate limits",
                        )
                        dolma_path = gr.Textbox(
                            label="Local Dolma Path (optional)",
                            placeholder="/path/to/dolma/subset",
                            info="If you downloaded a Dolma subset, point here for local training data search",
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("*Add your own sentiment categories. Leave empty to use defaults.*")
                        custom_json = gr.Code(
                            label="Custom Categories (optional JSON)",
                            language="json",
                            value="",
                            lines=8,
                        )
                        with gr.Accordion("Example custom category", open=False):
                            gr.Code(value=EXAMPLE_CUSTOM_JSON, language="json", interactive=False)

                analyze_btn = gr.Button("🚀 Analyze Comments", variant="primary", size="lg")
                status_box = gr.Markdown(value="", label="Status")

                with gr.Tabs():
                    with gr.Tab("📊 Report"):
                        report_output = gr.Markdown(label="Report")
                    with gr.Tab("🔍 View Source (Provenance)"):
                        provenance_chart_output = gr.Plot(
                            label="Training Data Signal",
                            visible=True,
                        )
                        provenance_output = gr.Markdown(label="Data Provenance")
                    with gr.Tab("📋 Raw JSON"):
                        json_output = gr.Code(label="JSON", language="json")

                analyze_btn.click(
                    fn=analyze_comments,
                    inputs=[repo_input, github_token, custom_json, dolma_path],
                    outputs=[status_box, report_output, json_output, provenance_output, provenance_chart_output],
                )

            # ---- TAB 2: EDIT SOURCE (Example Bank) ----
            with gr.Tab("✏️ Edit Source"):
                gr.Markdown("## Local Example Bank\n")
                gr.Markdown(
                    "This is your **edit source** layer. Examples here are injected into "
                    "the model's prompt as few-shot context, directly steering how it classifies. "
                    "Add examples to teach it your definitions. Remove ones that mislead it."
                )

                bank_display = gr.Markdown(value=get_bank_display, label="Current Examples")

                gr.Markdown("### Add Example")
                with gr.Row():
                    with gr.Column():
                        new_text = gr.Textbox(
                            label="Comment Text",
                            placeholder="Paste a GitHub comment here...",
                            lines=3,
                        )
                        new_labels = gr.Code(
                            label="Labels (JSON)",
                            language="json",
                            value='{"toxicity": "hostile", "constructiveness": "unconstructive"}',
                            lines=3,
                        )
                    with gr.Column():
                        new_source = gr.Textbox(
                            label="Source",
                            value="user",
                            info="Where this came from (user, github, dolma, etc.)",
                        )
                        new_notes = gr.Textbox(
                            label="Notes",
                            placeholder="Why you're adding this example...",
                            lines=2,
                        )

                add_btn = gr.Button("➕ Add Example", variant="primary")
                add_status = gr.Markdown("")

                add_btn.click(
                    fn=add_example,
                    inputs=[new_text, new_labels, new_source, new_notes],
                    outputs=[add_status, bank_display],
                )

                gr.Markdown("### Remove Example")
                with gr.Row():
                    remove_idx = gr.Number(label="Index to Remove", precision=0)
                    remove_btn = gr.Button("🗑️ Remove", variant="stop")
                remove_status = gr.Markdown("")

                remove_btn.click(
                    fn=remove_example,
                    inputs=[remove_idx],
                    outputs=[remove_status, bank_display],
                )

                gr.Markdown("### Import / Export")
                with gr.Row():
                    with gr.Column():
                        export_btn = gr.Button("📤 Export Bank as JSON")
                        export_output = gr.Code(label="Exported JSON", language="json")
                        export_btn.click(fn=export_bank, outputs=[export_output])
                    with gr.Column():
                        import_input = gr.Code(
                            label="Paste JSON to Import", language="json", lines=5
                        )
                        import_btn = gr.Button("📥 Import Examples")
                        import_status = gr.Markdown("")
                        import_btn.click(
                            fn=import_bank,
                            inputs=[import_input],
                            outputs=[import_status, bank_display],
                        )

            # ---- TAB 3: CONTRIBUTE UPSTREAM ----
            with gr.Tab("🎫 Contribute Upstream"):
                gr.Markdown("## Improve OLMo — Open a Ticket\n")
                gr.Markdown(
                    "Found a misclassification? Think the training data is missing something? "
                    "Because OLMo is fully open, you can file issues directly with Allen AI's "
                    "repositories. This tab drafts a well-formatted issue for you.\n\n"
                    "**Your feedback goes directly to the people who build the model and its data.**"
                )

                gr.Markdown("### Where to Send It")
                for key, target in UPSTREAM_TARGETS.items():
                    gr.Markdown(
                        f"- **[{target['repo']}]({target['new_issue_url']})** — "
                        f"{target['description']}"
                    )

                gr.Markdown("---\n### Draft Your Issue")

                with gr.Row():
                    with gr.Column():
                        ticket_target = gr.Dropdown(
                            label="Target Repository",
                            choices=[
                                (v["name"], k) for k, v in UPSTREAM_TARGETS.items()
                            ],
                            value="model_behavior",
                        )
                        ticket_type = gr.Dropdown(
                            label="Issue Type",
                            choices=[
                                "Misclassification Report",
                                "Training Data Suggestion",
                                "Feature Request",
                                "General Feedback",
                            ],
                            value="Misclassification Report",
                        )
                        ticket_title = gr.Textbox(
                            label="Issue Title",
                            placeholder="e.g., Blunt code review misclassified as toxic",
                        )
                    with gr.Column():
                        ticket_description = gr.Textbox(
                            label="Description",
                            placeholder="Describe the problem or suggestion in detail...",
                            lines=4,
                        )
                        ticket_example = gr.Textbox(
                            label="Example Comment (optional)",
                            placeholder="Paste the GitHub comment that was misclassified...",
                            lines=3,
                        )

                with gr.Row():
                    with gr.Column():
                        ticket_expected = gr.Textbox(
                            label="Expected Label (optional)",
                            placeholder="e.g., neutral",
                        )
                    with gr.Column():
                        ticket_actual = gr.Textbox(
                            label="Actual Label (optional)",
                            placeholder="e.g., hostile",
                        )

                ticket_evidence = gr.Textbox(
                    label="Additional Context / Evidence (optional)",
                    placeholder=(
                        "e.g., OLMoTrace showed similar training data from Reddit that "
                        "uses this phrasing in a non-hostile context..."
                    ),
                    lines=3,
                )

                draft_btn = gr.Button("📝 Draft Issue", variant="primary", size="lg")

                ticket_preview = gr.Markdown(label="Issue Preview")
                ticket_url = gr.Textbox(label="GitHub URL (pre-filled)", visible=False)
                ticket_body = gr.Textbox(label="Raw Body", visible=False)

                with gr.Row():
                    save_btn = gr.Button("💾 Save Draft Locally")
                    save_status = gr.Markdown("")

                draft_btn.click(
                    fn=generate_issue_body,
                    inputs=[
                        ticket_target, ticket_type, ticket_title,
                        ticket_description, ticket_example,
                        ticket_expected, ticket_actual, ticket_evidence,
                    ],
                    outputs=[ticket_preview, ticket_url, ticket_body],
                )

                save_btn.click(
                    fn=save_issue_draft,
                    inputs=[ticket_preview, ticket_url, ticket_body],
                    outputs=[save_status],
                )

            # ---- TAB 4: CONFIGURE ----
            with gr.Tab("⚙️ Configure Labels"):
                gr.Markdown("## Default Categories")
                gr.Markdown(
                    "These are applied unless you override with custom JSON on the Analyze tab."
                )
                for key, cat in DEFAULT_LABELS.items():
                    gr.Markdown(f"### {cat['name']}")
                    gr.Markdown(f"Labels: `{'`, `'.join(cat['labels'])}`")
                    gr.Markdown(f"_{cat['description']}_")

                gr.Markdown("---")
                gr.Markdown("## Adding Custom Categories")
                gr.Markdown(
                    "Paste JSON into the Analyze tab. Each category needs a `name`, "
                    "`labels` (list of strings), and optional `description`."
                )
                gr.Code(value=EXAMPLE_CUSTOM_JSON, language="json", interactive=False)

            # ---- TAB 5: ABOUT ----
            with gr.Tab("ℹ️ About"):
                gr.Markdown(INFO_MD)

    return app


if __name__ == "__main__":
    prewarm_model()  # Start loading OLMo into memory immediately at startup
    app = build_ui()
    app.queue()  # Required for long-running inference — without this Gradio drops requests
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"))

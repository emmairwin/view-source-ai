"""
Fossil Record
=============
Code leaves fossils. This reads them.

Paste a code snippet and see:
- Which lines appear verbatim in Dolma (OLMo's 3-trillion-token training corpus)
- How many times each line appears
- What licenses the matched source documents carry
- Whether those licenses are compatible with your project's license

Training Data: https://huggingface.co/datasets/allenai/dolma
infini-gram API: https://api.infini-gram.io
License: Apache 2.0
"""

import json
import os
import re
import gradio as gr
import requests
from typing import Optional

from data_provenance import (
    ExampleBank,
    build_provenance_chart,
    initialize_example_bank,
    search_snippet_provenance,
    format_snippet_provenance_markdown,
    assess_license_compatibility,
    build_dolma_issue_body,
    extract_snippet_phrases,
    debug_single_query,
    INFINIGRAM_API,
    INFINIGRAM_INDEX,
)


# ---------------------------------------------------------------------------
# Upstream Contribution Targets
# ---------------------------------------------------------------------------

UPSTREAM_TARGETS = {
    "training_data": {
        "name": "Training Data Quality (Dolma)",
        "repo": "allenai/dolma",
        "description": "Report problematic, incorrectly licensed, or missing data in the Dolma corpus",
        "new_issue_url": "https://github.com/allenai/dolma/issues/new",
        "email": "olmo@allenai.org",
    },
    "model_behavior": {
        "name": "Model Behavior (OLMo)",
        "repo": "allenai/OLMo",
        "description": "Report unexpected model outputs that may trace back to problematic training data",
        "new_issue_url": "https://github.com/allenai/OLMo/issues/new",
        "email": "olmo@allenai.org",
    },
    "olmotrace": {
        "name": "OLMoTrace (Data Tracing Tool)",
        "repo": "allenai/OLMoTrace",
        "description": "Feedback on the training data tracing and attribution system",
        "new_issue_url": "https://github.com/allenai/OLMoTrace/issues/new",
        "email": "olmo@allenai.org",
    },
}


def generate_issue_body(
    target_key: str,
    issue_type: str,
    title: str,
    description: str,
    snippet: str,
    detected_license: str,
    declared_license: str,
    evidence_notes: str,
) -> tuple[str, str, str]:
    """
    Generate a well-formatted GitHub issue body for upstream contribution.
    Returns (formatted_issue_markdown, github_url_with_prefill, raw_text).
    """
    target = UPSTREAM_TARGETS.get(target_key, UPSTREAM_TARGETS["training_data"])
    body_parts = []

    if issue_type == "License Concern":
        body_parts.append("## License Concern Report\n")
        body_parts.append("**Reported by:** Code Provenance Checker user")
        body_parts.append("**Tool:** Fossil Record — Code leaves fossils. This reads them.\n")
        body_parts.append(f"### Description\n{description}\n")
        if snippet:
            body_parts.append(f"### Code Snippet\n```\n{snippet[:1000]}\n```\n")
        if detected_license or declared_license:
            body_parts.append("### License Information")
            body_parts.append(f"- **Your declared license:** {declared_license or 'N/A'}")
            body_parts.append(f"- **Detected in Dolma source documents:** {detected_license or 'N/A'}\n")
        if evidence_notes:
            body_parts.append(f"### Additional Context\n{evidence_notes}\n")

    elif issue_type == "Missing Attribution":
        body_parts.append("## Missing Attribution Report\n")
        body_parts.append("**Reported by:** Code Provenance Checker user\n")
        body_parts.append(f"### Description\n{description}\n")
        if snippet:
            body_parts.append(f"### Code Snippet\n```\n{snippet[:1000]}\n```\n")
        if evidence_notes:
            body_parts.append(f"### Evidence\n{evidence_notes}\n")

    elif issue_type == "Data Quality Issue":
        body_parts.append("## Data Quality Issue\n")
        body_parts.append("**Reported by:** Code Provenance Checker user\n")
        body_parts.append(f"### Description\n{description}\n")
        if snippet:
            body_parts.append(f"### Example\n```\n{snippet[:1000]}\n```\n")
        if evidence_notes:
            body_parts.append(f"### Why This Matters\n{evidence_notes}\n")

    else:  # General Feedback
        body_parts.append("## Feedback\n")
        body_parts.append("**Reported by:** Code Provenance Checker user\n")
        body_parts.append(f"### Description\n{description}\n")
        if evidence_notes:
            body_parts.append(f"### Additional Context\n{evidence_notes}\n")

    body_parts.append("---")
    body_parts.append(
        "*Drafted via [Fossil Record](https://github.com/emmairwin/view-source-ai) — "
        "Code leaves fossils. This reads them.*"
    )
    full_body = "\n".join(body_parts)

    import urllib.parse
    params = urllib.parse.urlencode({
        "title": title or f"[Provenance] {issue_type}",
        "body": full_body,
    })
    github_url = f"{target['new_issue_url']}?{params}"

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
*(Pre-fills title and body — just click "Submit")*

**Option 2 — Email the team:**
Send to `{target['email']}` with the content above.
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
# Example Bank — module-level singleton
# ---------------------------------------------------------------------------

_bank = initialize_example_bank()


def get_bank_display():
    items = _bank.to_display_list()
    if not items:
        return "No patterns in bank yet."
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
    return "✅ Pattern added", get_bank_display()


def remove_example(index):
    try:
        idx = int(index)
    except (ValueError, TypeError):
        return "❌ Enter a valid index number", get_bank_display()
    result = _bank.remove(idx)
    if result:
        return f"✅ Removed pattern {idx}", get_bank_display()
    return "❌ Index not found", get_bank_display()


def export_bank():
    return _bank.export_json()


def import_bank(data):
    try:
        count = _bank.import_json(data)
        return f"✅ Imported {count} patterns", get_bank_display()
    except Exception as e:
        return f"❌ Import failed: {e}", get_bank_display()


# ---------------------------------------------------------------------------
# Snippet Provenance Pipeline
# ---------------------------------------------------------------------------

INFO_MD = """
## About Fossil Record

*Code leaves fossils. This reads them.*

When an AI model trains on code, that code leaves an impression — like a fossil in rock.
Fossil Record searches [Dolma](https://huggingface.co/datasets/allenai/dolma), the fully open
3-trillion-token dataset used to train OLMo, to find those impressions: exact phrases,
the documents they came from, and the licenses those documents carried.

**Dataset:** [Dolma v1.7](https://huggingface.co/datasets/allenai/dolma) — OLMo's training corpus (fully open and documented)
**Search API:** [infini-gram](https://infini-gram.io) — exact-phrase search over the full corpus
**License:** Apache 2.0

### What It Does

1. **TRACE** — Search your code snippet against Dolma to find exact-phrase matches in OLMo's training data
2. **INSPECT** — For each match, see source context and any license strings detected in the document
3. **FLAG** — Get per-line 🟢/🟡/🔴/⚪ compatibility flags comparing source licenses to your project's license
4. **CONTRIBUTE** — Report license gaps and data quality issues directly to Allen AI's open repos

### The Fossil Analogy

Paleontologists don't have every specimen — they read the record from fragments.
Fossil Record works the same way: a single distinctive function signature is enough
to identify the library, its license, and whether it belongs in your project.
The gaps in the record (deduplication, missing license headers, minified bundles)
are part of the story too.

### Flag Meanings

| Flag | When you see it | Severity |
|------|----------------|----------|
| 🟢 | Found in Dolma — source license matches (or is compatible with) your declared license | OK |
| 🟡 | Found in Dolma — permissive/permissive mismatch, or no license info available | Low |
| 🔴 | Found in Dolma — permissive/copyleft conflict requiring legal review | High |
| ⚪ | Not found in Dolma training data | — |

**Permissive vs. Copyleft:**
- 🟡 *MIT declared, Apache-2.0 in source* — both permissive, different attribution requirements
- 🔴 *MIT declared, GPL-3.0 in source* — copyleft is "viral"; combining may force you to adopt GPL terms

### Test Cases

**🔴 React + GPL-3.0** — paste React's `createElement` signature, set license to GPL-3.0.
React is MIT-licensed, so the source license (MIT detected in Dolma) is permissive and yours
is copyleft → Dolma source is fine for your GPL project, so actually 🟢.

Try the reverse: declare MIT and paste GPL-licensed code → 🔴 (copyleft source, permissive declaration).

**🟢 MIT header** — paste `Permission is hereby granted…`, set license to MIT.
Should get ~900K+ hits, MIT source, MIT declared → 🟢.

**🟡 MIT declared, Apache-2.0 code** — paste Apache-licensed code, set license to MIT.
Both permissive but different → 🟡 to flag the attribution-clause difference.

### Contribute Upstream

Because Dolma is fully open, you can report issues directly:
- Data quality / license concerns: [allenai/dolma](https://github.com/allenai/dolma/issues)
- Model behavior issues: [allenai/OLMo](https://github.com/allenai/OLMo/issues)
- Trace model outputs to training data: [OLMoTrace](https://playground.allenai.org)
- Email: olmo@allenai.org
"""


# ---------------------------------------------------------------------------
# License choices — grouped with visual separators for the dropdown
# ---------------------------------------------------------------------------

# Tuples of (display_label, value).  Items whose value starts with "──" are
# group-header separators; the normalize helper below converts them to "".
LICENSE_CHOICES: list[tuple[str, str]] = [
    ("── Permissive ──",  "── Permissive ──"),
    ("  MIT",             "MIT"),
    ("  Apache-2.0",      "Apache-2.0"),
    ("  BSD-2-Clause",    "BSD-2-Clause"),
    ("  BSD-3-Clause",    "BSD-3-Clause"),
    ("  ISC",             "ISC"),
    ("── Copyleft ──",    "── Copyleft ──"),
    ("  GPL-2.0",         "GPL-2.0"),
    ("  GPL-3.0",         "GPL-3.0"),
    ("  AGPL-3.0",        "AGPL-3.0"),
    ("  LGPL-2.1",        "LGPL-2.1"),
    ("  MPL-2.0",         "MPL-2.0"),
    ("── Other ──",       "── Other ──"),
    ("  Proprietary",     "Proprietary"),
    ("  Unlicensed",      "Unlicensed"),
]

_LICENSE_VALUES = [v for _, v in LICENSE_CHOICES]


def _normalize_user_license(raw: str) -> str:
    """Convert a dropdown value to a clean license string for compatibility checks.
    Group-header separators and "I don't know" map to empty string (= unknown)."""
    if not raw:
        return ""
    s = raw.strip()
    if s.startswith("──"):
        return ""
    return s


EXAMPLE_LABELS_JSON = json.dumps(
    {"provenance": "known-pattern", "library": "my-library", "license": "MIT"},
    indent=2,
)


def analyze_snippet(snippet: str, user_license: str = ""):
    """
    Check a code snippet against Dolma training data.
    Extracts 3-5 distinctive lines, counts each in Dolma, fetches source docs,
    detects licenses, and flags compatibility against the user's declared license.
    Yields: (status, report_md, json_out, provenance_md, chart, flag_md)
    """

    def _status(msg: str):
        return f"⏳ {msg}", "", "", "", None, ""

    if not snippet or not snippet.strip():
        yield "⚠️ Please paste a code snippet to analyze.", "", "", "", None, ""
        return

    non_blank = [l for l in snippet.splitlines() if l.strip()]
    if len(non_blank) > 50:
        yield "❌ Snippet is too long (max 50 non-blank lines). Please trim it.", "", "", "", None, ""
        return

    effective_license = _normalize_user_license(user_license)
    if not effective_license:
        yield "⚠️ Please select your project's license from the dropdown before checking.", "", "", "", None, ""
        return

    yield _status("Extracting distinctive lines from snippet…")

    phrases = extract_snippet_phrases(snippet)
    if not phrases:
        yield (
            "⚠️ Could not extract distinctive phrases "
            "(snippet may be too short or consist only of imports/single-word lines).",
            "", "", "", None, "",
        )
        return

    phrase_preview = "\n".join(f"- `{p}`" for p in phrases)
    yield _status(f"Searching {len(phrases)} line(s) in Dolma:\n\n{phrase_preview}")

    provenance = search_snippet_provenance(snippet, bank=_bank)
    report_md = format_snippet_provenance_markdown(provenance, effective_license)
    json_out = json.dumps(provenance, indent=2, default=str)

    assessment = assess_license_compatibility(
        effective_license, provenance.get("all_detected_licenses", [])
    )
    flag_icons = {"green": "🟢", "yellow": "🟡", "red": "🔴", "white": "⚪"}
    flag_icon = flag_icons.get(assessment["flag"], "⚪")
    flag_md = f"## {flag_icon} {assessment['explanation']}"

    fake_reports = [
        {
            "comment_user": "snippet",
            "report": {
                "provenance": {
                    "training_data_matches": [{
                        "exact_phrase": item["phrase"],
                        "exact_count": item["count"],
                        "exact_docs": item["docs"],
                        "sub_phrase_counts": [],
                    }]
                }
            },
        }
        for item in provenance["phrases"]
    ]
    chart = build_provenance_chart(fake_reports)

    hit_count = sum(1 for item in provenance["phrases"] if item["count"] > 0)
    yield (
        f"✅ Done — {hit_count} of {len(phrases)} line(s) found in Dolma.",
        report_md,
        json_out,
        report_md,
        chart,
        flag_md,
    )


def draft_dolma_issue(snippet: str, user_license: str, flag_md: str) -> tuple[str, str]:
    """Build a pre-filled GitHub issue for allenai/dolma."""
    if not snippet or not snippet.strip():
        return "⚠️ No snippet to report.", ""
    effective_license = _normalize_user_license(user_license)
    provenance = search_snippet_provenance(snippet, bank=_bank)
    assessment = assess_license_compatibility(
        effective_license, provenance.get("all_detected_licenses", [])
    )
    preview, url = build_dolma_issue_body(snippet, effective_license, assessment, provenance)
    return preview, url


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

MIT_TEST_PHRASE = "Permission is hereby granted, free of charge, to any person obtaining a copy"


def run_debug_query(query: str) -> tuple[str, str]:
    """
    Call infini-gram directly with `query` and return:
      - A human-readable summary (Markdown)
      - The raw JSON responses formatted for display
    """
    if not query or not query.strip():
        return "⚠️ Enter a query.", ""

    print(f"[DEBUG TAB] running debug_single_query for: {query!r}", flush=True)
    result = debug_single_query(query.strip())

    # ── Summary (Markdown) ──────────────────────────────────────────────
    count = result.get("count_parsed") or result.get("docs_parsed_count") or 0
    count_str = f"{count:,}" if isinstance(count, int) else str(count)
    error = result.get("error", "")

    lines = []
    lines.append(f"## Query: `{result['query']}`\n")
    lines.append(f"**API:** `{result['api_url']}`  **Index:** `{result['index']}`\n")

    if error:
        lines.append(f"### ❌ Error\n```\n{error}\n```\n")
    else:
        lines.append(f"### Count result: **{count_str}** occurrences in Dolma\n")
        if count == 0:
            lines.append(
                "> ⚠️ Zero hits. Either the query has no exact matches in Dolma, "
                "or there's an API/network issue. Check the raw response below."
            )
        else:
            lines.append(f"> ✅ Found {count_str} exact occurrences.")

    lines.append("\n---\n### Sub-phrase counts (4–6 word windows + long words)\n")
    sub = result.get("sub_phrase_counts", [])
    if sub:
        for item in sub[:20]:
            c = item.get("count", -1)
            c_str = f"{c:,}" if isinstance(c, int) and c >= 0 else "(error)"
            icon = "✅" if c > 0 else "⚪"
            lines.append(f"- {icon} `{item['phrase']}` → **{c_str}**")
    else:
        lines.append("*No sub-phrases generated.*")

    docs = result.get("docs_parsed_documents") or []
    lines.append(f"\n---\n### Sample documents returned: {len(docs)}\n")
    for i, doc in enumerate(docs, 1):
        lines.append(f"**Doc {i}:**")
        preview = doc.get("full_text_preview", "").replace("\n", " ")
        lines.append(f"```\n{preview[:400]}\n```")
        lines.append(f"*Metadata raw (first 200):* `{doc.get('metadata_raw','')[:200]}`\n")

    summary = "\n".join(lines)

    # ── Raw JSON ─────────────────────────────────────────────────────────
    raw_out = json.dumps(result, indent=2, default=str)

    return summary, raw_out


def preview_extraction(snippet: str) -> str:
    """Show exactly which lines extract_snippet_phrases would pick and why."""
    if not snippet or not snippet.strip():
        return "⚠️ Paste a snippet first."

    import re as _re

    lines_info = []
    for line in snippet.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        reasons = []
        skipped = False

        if _re.match(r'^(?:import\s+\S|from\s+\S+\s+import)\b', stripped):
            skipped = True
            reasons.append("SKIP: import statement")
        elif len(stripped) < 15:
            skipped = True
            reasons.append(f"SKIP: too short ({len(stripped)} chars, min 15)")
        else:
            score = 0.0
            if _re.search(
                r'SPDX-License-Identifier|permission\s+is\s+hereby|copyright|\ball\s+rights\s+reserved\b|licensed\s+under',
                stripped, _re.IGNORECASE
            ):
                score += 10
                reasons.append("+10 license/copyright keyword")
            if _re.match(r'#|//|\*+\s|/\*', stripped):
                score += 4
                reasons.append("+4 comment line")
            if _re.search(r'\b(raise|error|exception|warn|assert|fail)\b', stripped, _re.IGNORECASE):
                score += 3
                reasons.append("+3 error/raise")
            if _re.match(r'(?:def |class |function |public |private |static |async\s+def |fn )\w', stripped):
                score += 3
                reasons.append("+3 function/class def")
            if _re.search(r'["\'](?:[^"\']{10,})["\']', stripped):
                score += 2
                reasons.append("+2 quoted string")
            length_bonus = min(len(stripped) / 25.0, 4.0)
            score += length_bonus
            reasons.append(f"+{length_bonus:.1f} length ({len(stripped)} chars)")
            reasons.append(f"= score {score:.1f}")

        phrase = (stripped[:100].rsplit(' ', 1)[0] if len(stripped) > 100 else stripped).rstrip(',;:').strip()
        lines_info.append((skipped, score if not skipped else -1, stripped, phrase, reasons))

    # Sort same as the real function
    eligible = [(s, ln, p, r) for sk, s, ln, p, r in lines_info if not sk]
    eligible.sort(key=lambda x: (-x[0], -len(x[1])))
    chosen_phrases = set()
    chosen = []
    for score, line, phrase, _ in eligible:
        if phrase not in chosen_phrases:
            chosen_phrases.add(phrase)
            chosen.append(phrase)
        if len(chosen) >= 5:
            break

    out = ["## Phrase Extraction Preview\n"]
    out.append("### Lines evaluated\n")
    for skipped, score, line, phrase, reasons in lines_info:
        marker = "⏭️ skip" if skipped else ("✅ CHOSEN" if phrase in chosen_phrases else "  scored")
        reason_str = " | ".join(reasons)
        out.append(f"- **[{marker}]** `{line[:80]}` — {reason_str}")

    out.append(f"\n### Final phrases sent to API ({len(chosen)} total)\n")
    for i, p in enumerate(chosen, 1):
        out.append(f"{i}. `{p}`")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(
        title="View Source for AI — Code Snippet Source and License",
    ) as app:
        gr.Markdown("# View Source for AI — Code Snippet Source and License")
        gr.Markdown(
            "*Code leaves fossils in AI training data. Paste a snippet — see if it appears in "
            "[Dolma](https://huggingface.co/datasets/allenai/dolma), "
            "OLMo's 3-trillion-token training corpus, what license it carried there, "
            "and whether that's compatible with your project.*"
        )

        with gr.Tabs():
            # ----------------------------------------------------------------
            # TAB 1: CHECK SNIPPET
            # ----------------------------------------------------------------
            with gr.Tab("🔬 Check Snippet"):
                with gr.Row():
                    with gr.Column(scale=3):
                        snippet_input = gr.Textbox(
                            label="code snippet",
                            placeholder=(
                                "Paste your code snippet here…\n\n"
                                "Test 🟢 — MIT header, declare MIT:\n"
                                "Permission is hereby granted, free of charge, to any person obtaining a copy\n\n"
                                "Test 🟡 — Apache source, declare MIT (permissive/permissive mismatch):\n"
                                "Licensed under the Apache License, Version 2.0\n\n"
                                "Test 🔴 — copyleft source, declare MIT:\n"
                                "This program is free software: you can redistribute it and/or modify it\n"
                                "under the terms of the GNU General Public License"
                            ),
                            lines=14,
                            max_lines=50,
                        )
                    with gr.Column(scale=1):
                        user_license_input = gr.Dropdown(
                            label="Your project's license",
                            choices=LICENSE_CHOICES,
                            value=None,
                            info=(
                                "The license under which **you** will use or distribute this code. "
                                "Used to check whether the source license(s) found in Dolma "
                                "are compatible with your project."
                            ),
                        )
                        gr.Markdown(
                            "**Flag meanings:**\n\n"
                            "🟢 Compatible — same or allowed license\n\n"
                            "🟡 Check attribution — permissive mismatch\n\n"
                            "🔴 Conflict — copyleft source vs permissive project\n\n"
                            "⚪ No license info (select yours above to compare)"
                        )

                check_btn = gr.Button("🔎 Check Provenance", variant="primary", size="lg")
                status_box = gr.Markdown(value="")
                flag_output = gr.Markdown(value="")

                with gr.Tabs():
                    with gr.Tab("📊 Results"):
                        report_output = gr.Markdown(label="Per-Line Provenance Results")
                    with gr.Tab("📈 Dolma Hit Chart"):
                        provenance_chart_output = gr.Plot(label="Dolma Occurrence Counts")
                    with gr.Tab("🔍 Full Provenance"):
                        provenance_output = gr.Markdown(label="Full Data Provenance")
                    with gr.Tab("📋 Raw JSON"):
                        json_output = gr.Code(label="JSON", language="json")

                check_btn.click(
                    fn=analyze_snippet,
                    inputs=[snippet_input, user_license_input],
                    outputs=[
                        status_box, report_output, json_output,
                        provenance_output, provenance_chart_output, flag_output,
                    ],
                )

                gr.Markdown("---")
                with gr.Accordion("🎫 Draft issue to allenai/dolma (for 🟡/🔴 flags)", open=False):
                    gr.Markdown(
                        "Found a license mismatch or missing attribution? "
                        "Draft a data-quality issue to "
                        "[allenai/dolma](https://github.com/allenai/dolma/issues) "
                        "with your snippet, Dolma hit counts, and source context pre-filled."
                    )
                    draft_issue_btn = gr.Button("🎫 Build Issue Draft", variant="secondary")
                    issue_preview = gr.Markdown()
                    issue_url_box = gr.Textbox(
                        label="GitHub URL (pre-filled — open in browser to submit)",
                        interactive=False,
                    )
                    draft_issue_btn.click(
                        fn=draft_dolma_issue,
                        inputs=[snippet_input, user_license_input, flag_output],
                        outputs=[issue_preview, issue_url_box],
                    )

            # ----------------------------------------------------------------
            # TAB 2: EDIT SOURCE (Known Pattern Bank)
            # ----------------------------------------------------------------
            with gr.Tab("✏️ Edit Source"):
                gr.Markdown("## Known Pattern Bank\n")
                gr.Markdown(
                    "A local database of known code patterns with confirmed libraries "
                    "and licenses. When a Dolma search returns a match, the tool "
                    "cross-references it here to annotate well-known signatures.\n\n"
                    "Seed entries cover: MIT/Apache/GPL/BSD headers, "
                    "React `createElement`, Lodash exports, Flask `wsgi_app`, "
                    "Meta/JetBrains copyright markers."
                )

                bank_display = gr.Markdown(value=get_bank_display)

                gr.Markdown("### Add Pattern")
                with gr.Row():
                    with gr.Column():
                        new_text = gr.Textbox(
                            label="Code Line / Signature",
                            placeholder="Paste the distinctive line or function signature…",
                            lines=3,
                        )
                        new_labels = gr.Code(
                            label="Labels (JSON)",
                            language="json",
                            value=EXAMPLE_LABELS_JSON,
                            lines=4,
                        )
                    with gr.Column():
                        new_source = gr.Textbox(
                            label="Source",
                            value="user",
                            info="Where this came from (user, npm, pypi, github, etc.)",
                        )
                        new_notes = gr.Textbox(
                            label="Notes",
                            placeholder="Library name, version, license, why you're adding this…",
                            lines=3,
                        )

                add_btn = gr.Button("➕ Add Pattern", variant="primary")
                add_status = gr.Markdown("")
                add_btn.click(
                    fn=add_example,
                    inputs=[new_text, new_labels, new_source, new_notes],
                    outputs=[add_status, bank_display],
                )

                gr.Markdown("### Remove Pattern")
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
                        export_btn = gr.Button("📤 Export as JSON")
                        export_output = gr.Code(label="Exported JSON", language="json")
                        export_btn.click(fn=export_bank, outputs=[export_output])
                    with gr.Column():
                        import_input = gr.Code(
                            label="Paste JSON to Import", language="json", lines=5
                        )
                        import_btn = gr.Button("📥 Import")
                        import_status = gr.Markdown("")
                        import_btn.click(
                            fn=import_bank,
                            inputs=[import_input],
                            outputs=[import_status, bank_display],
                        )

            # ----------------------------------------------------------------
            # TAB 3: CONTRIBUTE UPSTREAM
            # ----------------------------------------------------------------
            with gr.Tab("🎫 Contribute Upstream"):
                gr.Markdown("## Report Issues to Allen AI\n")
                gr.Markdown(
                    "Found a license problem, missing attribution, or data quality "
                    "issue in Dolma? Because Dolma is fully open, you can file issues "
                    "directly with the team.\n\n"
                    "**Your feedback goes directly to the people who build and "
                    "maintain the training data.**"
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
                            choices=[(v["name"], k) for k, v in UPSTREAM_TARGETS.items()],
                            value="training_data",
                        )
                        ticket_type = gr.Dropdown(
                            label="Issue Type",
                            choices=[
                                "License Concern",
                                "Missing Attribution",
                                "Data Quality Issue",
                                "General Feedback",
                            ],
                            value="License Concern",
                        )
                        ticket_title = gr.Textbox(
                            label="Issue Title",
                            placeholder="e.g., GPL-licensed code found in Dolma without attribution",
                        )
                    with gr.Column():
                        ticket_description = gr.Textbox(
                            label="Description",
                            placeholder="Describe the problem or concern in detail…",
                            lines=4,
                        )
                        ticket_snippet = gr.Textbox(
                            label="Code Snippet (optional)",
                            placeholder="Paste the code snippet that triggered this report…",
                            lines=3,
                        )

                with gr.Row():
                    with gr.Column():
                        ticket_detected = gr.Textbox(
                            label="Detected License in Dolma Source (optional)",
                            placeholder="e.g., GPL-3.0",
                        )
                    with gr.Column():
                        ticket_declared = gr.Textbox(
                            label="Your Declared License (optional)",
                            placeholder="e.g., MIT",
                        )

                ticket_evidence = gr.Textbox(
                    label="Additional Context / Evidence (optional)",
                    placeholder=(
                        "Dolma hit counts, infini-gram search links, "
                        "source document URLs, why this is a problem…"
                    ),
                    lines=3,
                )

                draft_btn = gr.Button("📝 Draft Issue", variant="primary", size="lg")
                ticket_preview = gr.Markdown()
                ticket_url = gr.Textbox(label="GitHub URL (pre-filled)", visible=False)
                ticket_body = gr.Textbox(label="Raw Body", visible=False)

                with gr.Row():
                    save_btn = gr.Button("💾 Save Draft Locally")
                    save_status = gr.Markdown("")

                draft_btn.click(
                    fn=generate_issue_body,
                    inputs=[
                        ticket_target, ticket_type, ticket_title,
                        ticket_description, ticket_snippet,
                        ticket_detected, ticket_declared, ticket_evidence,
                    ],
                    outputs=[ticket_preview, ticket_url, ticket_body],
                )
                save_btn.click(
                    fn=save_issue_draft,
                    inputs=[ticket_preview, ticket_url, ticket_body],
                    outputs=[save_status],
                )

            # ----------------------------------------------------------------
            # TAB 4: API DEBUG
            # ----------------------------------------------------------------
            with gr.Tab("🔬 API Debug"):
                gr.Markdown("## infini-gram API Debugger\n")
                gr.Markdown(
                    f"Directly test any query against the infini-gram API "
                    f"(`{INFINIGRAM_INDEX}` index at `{INFINIGRAM_API}`). "
                    f"Shows the exact request, the full raw HTTP response, "
                    f"parsed counts, sample documents, and sub-phrase signal.\n\n"
                    f"**Console** (where you launched the app) shows all API calls in real time."
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        debug_query_input = gr.Textbox(
                            label="Query string (sent verbatim to infini-gram)",
                            value=MIT_TEST_PHRASE,
                            lines=2,
                        )
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "**Step 1 — baseline test:**\n\nRun the pre-filled MIT phrase. "
                            "If count = 0, the API call itself is broken.\n\n"
                            "**Step 2 — snippet test:**\n\nCopy one extracted line "
                            "from the Check Snippet tab and test it here.\n\n"
                            "**Step 3 — shorten it:**\n\nPick 4-6 words from a "
                            "distinctive part and try those."
                        )

                debug_run_btn = gr.Button("▶ Run Query", variant="primary")

                with gr.Tabs():
                    with gr.Tab("📊 Summary"):
                        debug_summary = gr.Markdown()
                    with gr.Tab("📋 Raw JSON"):
                        debug_raw = gr.Code(label="Full API response JSON", language="json")

                debug_run_btn.click(
                    fn=run_debug_query,
                    inputs=[debug_query_input],
                    outputs=[debug_summary, debug_raw],
                )

                gr.Markdown("---\n### Phrase Extraction Preview\n")
                gr.Markdown(
                    "Shows exactly which lines get picked from your snippet "
                    "and why — scores, skip reasons, and the final phrase sent to the API."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        debug_snippet_input = gr.Textbox(
                            label="Paste your snippet here",
                            lines=8,
                            placeholder="function createElement(type, config, children) {\n  var propName;\n  var props = {};\n  var key = null;\n  var ref = null;",
                        )
                    with gr.Column(scale=1):
                        debug_extract_btn = gr.Button("🔍 Preview Extraction", variant="secondary")

                debug_extraction_output = gr.Markdown()
                debug_extract_btn.click(
                    fn=preview_extraction,
                    inputs=[debug_snippet_input],
                    outputs=[debug_extraction_output],
                )

            # ----------------------------------------------------------------
            # TAB 5: ABOUT
            # ----------------------------------------------------------------
            with gr.Tab("ℹ️ About"):
                gr.Markdown(INFO_MD)

    return app


if __name__ == "__main__":
    app = build_ui()
    app.queue()  # Required for generator functions (streaming status updates)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
    )

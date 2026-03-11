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
    payload = {"index": INFINIGRAM_INDEX, "query_type": "count", "query": query}
    print(f"[infini-gram COUNT] query={query!r} → POST {INFINIGRAM_API}", flush=True)
    try:
        r = requests.post(INFINIGRAM_API, json=payload, timeout=10)
        raw = r.text
        print(f"[infini-gram COUNT] status={r.status_code} response={raw[:300]}", flush=True)
        d = r.json()
        count = d.get("count", d.get("cnt", 0))
        print(f"[infini-gram COUNT] parsed count={count}", flush=True)
        return count
    except Exception as exc:
        print(f"[infini-gram COUNT] EXCEPTION: {exc}", flush=True)
        return 0


def _infinigram_docs(query: str, n: int = 3) -> tuple[int, list[dict]]:
    """Return (total_count, list of doc snippets) for a phrase."""
    import requests
    payload = {"index": INFINIGRAM_INDEX, "query_type": "search_docs", "query": query, "maxnum": n}
    print(f"[infini-gram DOCS] query={query!r} → POST {INFINIGRAM_API}", flush=True)
    try:
        r = requests.post(INFINIGRAM_API, json=payload, timeout=15)
        raw = r.text
        print(f"[infini-gram DOCS] status={r.status_code} response_len={len(raw)} preview={raw[:500]}", flush=True)
        d = r.json()
        count = d.get("cnt", d.get("count", 0))
        print(f"[infini-gram DOCS] parsed count={count} num_docs={len(d.get('documents', []))}", flush=True)
        docs = []
        for doc in d.get("documents", []):
            spans = doc.get("spans", [])
            # spans is a list of [text_segment, is_match_bool] pairs
            text = "".join(s[0] for s in spans if isinstance(s, (list, tuple)) and s)
            try:
                meta = json.loads(doc.get("metadata", "{}"))
            except Exception:
                meta = {}
            inner = meta.get("metadata", {})
            url = inner.get("url", "") or inner.get("metadata", {}).get("url", "")
                # Keep up to 800 chars so nearby license comments aren't cut off
            docs.append({"text": text[:800], "source": "Dolma training data", "url": url})
        return count, docs
    except Exception as exc:
        print(f"[infini-gram DOCS] EXCEPTION: {exc}", flush=True)
        return 0, []


def debug_single_query(query: str) -> dict:
    """
    Run a single query against infini-gram and return everything:
    - The exact request sent
    - The raw HTTP status and response body
    - Parsed count
    - Sample documents with their raw spans
    - Any error
    Also tries progressively shorter sub-phrases (4..6-word windows)
    so you can see which parts have signal.
    """
    import requests

    result = {
        "query": query,
        "api_url": INFINIGRAM_API,
        "index": INFINIGRAM_INDEX,
        "count_request": None,
        "count_response_status": None,
        "count_response_raw": None,
        "count_parsed": None,
        "docs_request": None,
        "docs_response_status": None,
        "docs_response_raw": None,
        "docs_parsed_count": None,
        "docs_parsed_documents": None,
        "sub_phrase_counts": [],
        "error": None,
    }

    # ── 1. COUNT ──────────────────────────────────────────────────────────
    count_payload = {"index": INFINIGRAM_INDEX, "query_type": "count", "query": query}
    result["count_request"] = count_payload
    try:
        r = requests.post(INFINIGRAM_API, json=count_payload, timeout=15)
        result["count_response_status"] = r.status_code
        result["count_response_raw"] = r.text[:2000]
        d = r.json()
        result["count_parsed"] = d.get("count", d.get("cnt", d))
    except Exception as exc:
        result["error"] = f"COUNT call failed: {exc}"
        return result

    # ── 2. SEARCH_DOCS ────────────────────────────────────────────────────
    docs_payload = {"index": INFINIGRAM_INDEX, "query_type": "search_docs", "query": query, "maxnum": 3}
    result["docs_request"] = docs_payload
    try:
        r = requests.post(INFINIGRAM_API, json=docs_payload, timeout=20)
        result["docs_response_status"] = r.status_code
        result["docs_response_raw"] = r.text[:4000]
        d = r.json()
        result["docs_parsed_count"] = d.get("cnt", d.get("count", 0))
        result["docs_parsed_documents"] = [
            {
                "spans": doc.get("spans", [])[:5],  # first 5 span entries
                "full_text_preview": "".join(
                    s[0] for s in doc.get("spans", [])
                    if isinstance(s, (list, tuple)) and s
                )[:500],
                "metadata_raw": doc.get("metadata", "")[:300],
            }
            for doc in d.get("documents", [])
        ]
    except Exception as exc:
        result["error"] = f"DOCS call failed: {exc}"
        return result

    # ── 3. SUB-PHRASE COUNTS (4-6 word windows) ───────────────────────────
    words = query.split()
    sub_phrases_to_try = set()
    for window in (4, 5, 6):
        for i in range(len(words) - window + 1):
            sub_phrases_to_try.add(" ".join(words[i:i + window]))
    # Also try individual words longer than 5 chars
    for w in words:
        if len(w) > 5:
            sub_phrases_to_try.add(w)

    for sp in sorted(sub_phrases_to_try):
        if sp == query:
            continue
        try:
            sr = requests.post(
                INFINIGRAM_API,
                json={"index": INFINIGRAM_INDEX, "query_type": "count", "query": sp},
                timeout=10,
            )
            sd = sr.json()
            cnt = sd.get("count", sd.get("cnt", 0))
            result["sub_phrase_counts"].append({"phrase": sp, "count": cnt, "raw": sr.text[:200]})
        except Exception as exc:
            result["sub_phrase_counts"].append({"phrase": sp, "count": -1, "error": str(exc)})

    result["sub_phrase_counts"].sort(key=lambda x: -(x.get("count") or 0))
    return result


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
# 1b. SNIPPET PROVENANCE — Phrase extraction, Dolma search, license detection
# ---------------------------------------------------------------------------

_LICENSE_PATTERNS = [
    (r'\bMIT\b(?:\s+License)?', 'MIT'),
    (r'\bApache(?:-2\.0)?\b', 'Apache'),
    (r'\bGPL\b|\bGNU\s+General\s+Public\s+License\b', 'GPL'),
    (r'\bBSD\b(?:\s+License)?', 'BSD'),
    (r'permission\s+is\s+hereby\s+granted', 'MIT (permission grant)'),
    (r'\ball\s+rights\s+reserved\b', 'All Rights Reserved'),
    (r'\bpublic\s+domain\b', 'Public Domain'),
    (r'\bCC\s*BY\b', 'Creative Commons'),
    (r'\bISC\b(?:\s+License)?', 'ISC'),
    (r'\bMPL\b', 'Mozilla Public License'),
    (r'SPDX-License-Identifier\s*:', 'SPDX-License-Identifier'),
    (r'[Cc]opyright\s+(?:\(c\)|©|\(C\))', 'Copyright'),
    (r'[Ll]icensed\s+under\s+the', 'Licensed under'),
]


def _detect_licenses(text: str) -> list[str]:
    """Return a deduplicated list of license names found in a block of text."""
    found: list[str] = []
    for pattern, label in _LICENSE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE) and label not in found:
            found.append(label)
    return found


# ---------------------------------------------------------------------------
# License compatibility matrix
# "green"  — same or clearly compatible family
# "yellow" — possibly compatible but worth checking (e.g. permissive vs copyleft)
# "red"    — known incompatibility / proprietary restriction
# "white"  — no match / can't determine
# ---------------------------------------------------------------------------

# Canonical family groupings (lower-case keys)
_PERMISSIVE  = {"mit", "apache", "bsd", "isc", "public domain", "creative commons", "unlicensed"}
_COPYLEFT    = {"gpl", "agpl", "lgpl", "mpl"}
_RESTRICTIVE = {"all rights reserved", "proprietary"}

# Normalized label → canonical family
def _lic_family(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("mit", "permission grant")):
        return "mit"
    if "apache" in n:
        return "apache"
    if "agpl" in n or "affero" in n:
        return "agpl"
    if "lgpl" in n or "lesser" in n:
        return "lgpl"
    if any(k in n for k in ("gpl", "gnu general")):
        return "gpl"
    if "mpl" in n or "mozilla" in n:
        return "mpl"
    if "bsd" in n:
        return "bsd"
    if "isc" in n:
        return "isc"
    if "all rights reserved" in n:
        return "all rights reserved"
    if "proprietary" in n:
        return "proprietary"
    if "public domain" in n or "unlicensed" in n:
        return "unlicensed"
    if "creative commons" in n or "cc by" in n:
        return "creative commons"
    return n.strip()


def assess_license_compatibility(
    user_license: str,
    detected_licenses: list[str],
) -> dict:
    """
    Compare the user's declared snippet license against licenses found in
    Dolma source documents.

    Returns:
        {
          "flag": "green" | "yellow" | "red" | "white",
          "explanation": str,          # one-line human summary
          "user_license": str,
          "detected": list[str],       # deduplicated across all phrases
        }
    """
    user = user_license.strip() if user_license else ""
    if not detected_licenses:
        if not user:
            return {"flag": "white", "explanation": "No license information found in Dolma source documents.", "user_license": user, "detected": []}
        return {"flag": "white", "explanation": "No license strings detected in the matching Dolma documents — compatibility cannot be determined.", "user_license": user, "detected": []}

    # Deduplicate preserving order
    seen: set[str] = set()
    unique_detected: list[str] = []
    for lic in detected_licenses:
        if lic not in seen:
            seen.add(lic)
            unique_detected.append(lic)

    if not user:
        names = ", ".join(f"`{l}`" for l in unique_detected)
        return {
            "flag": "white",
            "explanation": f"Licenses detected in training sources ({names}), but you haven't selected your project's license — select one from the dropdown to get a compatibility flag.",
            "user_license": "",
            "detected": unique_detected,
        }

    uf = _lic_family(user)
    detected_families = [_lic_family(d) for d in unique_detected]

    # All-rights-reserved in source → red regardless of user license
    if "all rights reserved" in detected_families:
        return {
            "flag": "red",
            "explanation": (
                f"⛔ A source document contains **All Rights Reserved** — "
                f"this phrasing in Dolma may originate from proprietary code that conflicts with your `{user}` license."
            ),
            "user_license": user,
            "detected": unique_detected,
        }

    # Permissive user + copyleft source → 🔴 (copyleft is viral; GPL taints permissive code)
    if uf in _PERMISSIVE and any(f in _COPYLEFT for f in detected_families):
        conflicting = [d for d, f in zip(unique_detected, detected_families) if f in _COPYLEFT]
        return {
            "flag": "red",
            "explanation": (
                f"⛔ Permissive/copyleft conflict — source document(s) carry copyleft license(s) "
                f"({', '.join(conflicting)}) which cannot be combined with your permissive `{user}` "
                f"license without adopting the copyleft terms."
            ),
            "user_license": user,
            "detected": unique_detected,
        }

    # Copyleft user + permissive source → 🟢 (copyleft can incorporate permissive code)
    if uf in _COPYLEFT and all(f in _PERMISSIVE for f in detected_families):
        return {
            "flag": "green",
            "explanation": (
                f"✅ Source licenses ({', '.join(unique_detected)}) are permissive — "
                f"compatible with your copyleft `{user}` license."
            ),
            "user_license": user,
            "detected": unique_detected,
        }

    # Exact same family → 🟢
    if uf in detected_families:
        return {
            "flag": "green",
            "explanation": (
                f"✅ Detected license(s) in source documents ({', '.join(unique_detected)}) "
                f"match your declared `{user}` license — compatible."
            ),
            "user_license": user,
            "detected": unique_detected,
        }

    # Both permissive but different (e.g. MIT declared, Apache detected) → 🟡
    # Lower severity than a copyleft conflict; worth noting but not a blocker.
    if uf in _PERMISSIVE and all(f in _PERMISSIVE for f in detected_families):
        names = ", ".join(unique_detected)
        return {
            "flag": "yellow",
            "explanation": (
                f"⚠️ Permissive/permissive mismatch — source uses ({names}), "
                f"you declared `{user}`. Both are permissive but check attribution "
                f"requirements differ (e.g. Apache-2.0 requires NOTICE file)."
            ),
            "user_license": user,
            "detected": unique_detected,
        }

    # Both copyleft but different versions/flavours → 🟡
    if uf in _COPYLEFT and all(f in _COPYLEFT for f in detected_families):
        names = ", ".join(unique_detected)
        return {
            "flag": "yellow",
            "explanation": (
                f"⚠️ Copyleft/copyleft mismatch — source uses ({names}), "
                f"you declared `{user}`. Combining different copyleft licenses "
                f"may require legal review."
            ),
            "user_license": user,
            "detected": unique_detected,
        }

    # Mixed or unknown → yellow
    names = ", ".join(unique_detected)
    return {
        "flag": "yellow",
        "explanation": (
            f"⚠️ Source document license(s) ({names}) differ from your declared `{user}` license — "
            f"review compatibility before redistributing."
        ),
        "user_license": user,
        "detected": unique_detected,
    }


def build_dolma_issue_body(
    snippet: str,
    user_license: str,
    assessment: dict,
    provenance: dict,
) -> tuple[str, str]:
    """
    Build a pre-filled GitHub issue body for allenai/dolma reporting a
    potential license concern.

    Returns (issue_markdown_preview, github_new_issue_url).
    """
    import urllib.parse

    flag = assessment.get("flag", "white")
    explanation = assessment.get("explanation", "")
    detected = assessment.get("detected", [])

    # Collect phrase hit details
    phrase_rows: list[str] = []
    sample_passages: list[str] = []
    for item in provenance.get("phrases", []):
        if item["count"] > 0:
            phrase_rows.append(
                f'| `{item["phrase"][:80]}` | {item["count"]:,} | {", ".join(item["licenses"]) or "none detected"} |'
            )
            for i, doc in enumerate(item["docs"][:2], 1):
                preview = doc["text"][:300].replace("\n", " ")
                url_line = f"\n  Source: {doc['url']}" if doc.get("url") else ""
                sample_passages.append(f"**Passage {i} for phrase `{item['phrase'][:60]}`:**\n```\n{preview}\n```{url_line}")

    severity_label = {"green": "info", "yellow": "potential concern", "red": "license conflict"}.get(flag, "unknown")

    body_lines = [
        f"## License Concern Report — {severity_label.title()}",
        "",
        f"**Reported by:** View Source AI (snippet provenance tool)",
        f"**Flag:** {flag.upper()} — {explanation}",
        f"**User-declared license:** `{user_license or 'not specified'}`",
        f"**Detected license(s) in Dolma sources:** {', '.join(detected) if detected else 'none'}",
        "",
        "### Snippet",
        f"```\n{snippet[:1000]}\n```",
        "",
    ]

    if phrase_rows:
        body_lines += [
            "### Phrase Match Summary",
            "",
            "| Phrase | Dolma count | Licenses in source |",
            "|--------|-------------|-------------------|",
            *phrase_rows,
            "",
        ]

    if sample_passages:
        body_lines += ["### Sample Source Passages", ""]
        body_lines += sample_passages
        body_lines.append("")

    body_lines += [
        "---",
        "*Drafted via [View Source AI](https://github.com/allenai/dolma) — "
        "a tool for auditing training data provenance.*",
    ]

    full_body = "\n".join(body_lines)
    title = f"[License] {severity_label.title()}: {', '.join(detected[:2]) or 'unknown'} in training source"

    params = urllib.parse.urlencode({"title": title, "body": full_body, "labels": "data-quality"})
    github_url = f"https://github.com/allenai/dolma/issues/new?{params}"

    preview_md = f"""### 📋 Issue Preview — allenai/dolma

**Severity:** {flag.upper()} — {severity_label}
**Title:** {title}

---

{full_body}

---

### 🚀 Submit

[Open this issue on GitHub (pre-filled)]({github_url})
"""
    return preview_md, github_url


def extract_snippet_phrases(snippet: str, n: int = 5) -> list[str]:
    """
    Extract 3–5 distinctive phrases from a code snippet for Dolma searching.

    Priority (highest first):
      1. License / copyright / permission lines
      2. Comment lines (# // /* *)
      3. Error / exception / raise lines
      4. Function / class signatures with specific names
      5. Lines with quoted strings (>= 10 chars inside quotes)
      6. Other long lines

    Skips blank lines and pure import statements.
    Each returned phrase is capped at 100 characters.
    """
    scored: list[tuple[float, int, str]] = []
    for line in snippet.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Skip pure import lines
        if re.match(r'^(?:import\s+\S|from\s+\S+\s+import)\b', stripped):
            continue
        # Skip very short lines (lone brackets, keywords, etc.)
        if len(stripped) < 15:
            continue

        score = 0.0
        if re.search(
            r'SPDX-License-Identifier|permission\s+is\s+hereby|copyright|\ball\s+rights\s+reserved\b|licensed\s+under',
            stripped, re.IGNORECASE
        ):
            score += 10
        if re.match(r'#|//|\*+\s|/\*', stripped):
            score += 4
        if re.search(r'\b(raise|error|exception|warn|assert|fail)\b', stripped, re.IGNORECASE):
            score += 3
        if re.match(r'(?:def |class |function |public |private |static |async\s+def |fn )\w', stripped):
            score += 3
        if re.search(r'["\'](?:[^"\']{10,})["\']', stripped):
            score += 2
        score += min(len(stripped) / 25.0, 4.0)

        scored.append((score, len(stripped), stripped))

    scored.sort(key=lambda x: (-x[0], -x[1]))

    seen: set[str] = set()
    result: list[str] = []
    for _, _, line in scored:
        phrase = (line[:100].rsplit(' ', 1)[0] if len(line) > 100 else line).rstrip(',;:').strip()
        if phrase and phrase not in seen:
            seen.add(phrase)
            result.append(phrase)
        if len(result) >= n:
            break
    return result


def _joined_snippet_phrase(snippet: str, max_len: int = 180) -> str:
    """
    Return the first `max_len` characters of the snippet with lines joined by
    a single space and leading whitespace stripped from each line.

    This normalises indented, multi-line code into the single-line form that
    often appears in minified JS/CSS bundles stored in Dolma.  Those bundles
    usually include the license comment on the very same line (or just before),
    so matching this joined form is the most reliable way to find the license.
    """
    parts = []
    for line in snippet.splitlines():
        s = line.strip()
        if s:
            parts.append(s)
    joined = " ".join(parts)
    return joined[:max_len]


def search_snippet_provenance(snippet: str, bank=None) -> dict:
    """
    Full pipeline for a code snippet:
      1. Build a "whole-snippet" probe (lines joined, first 180 chars) and search it
      2. Extract 3–5 distinctive individual phrases and search each
      3. For phrases with count > 0, fetch sample documents
      4. Detect license strings in those documents
      5. Supplement with bank-derived license when no Dolma license found
      6. Cross-reference against known pattern bank (optional)

    Returns:
        {
          "phrases": [{"phrase", "count", "docs", "licenses", "bank_match"}, ...],
          "all_detected_licenses": [str, ...],
        }
    """
    # Build phrase list: whole-snippet probe first, then individual extracted lines
    joined_probe = _joined_snippet_phrase(snippet)
    extracted = extract_snippet_phrases(snippet, n=5)

    # Deduplicate: if the joined probe is nearly identical to the first extracted
    # phrase (i.e. the snippet is already a single line), skip it to save an API call.
    phrases: list[str] = []
    seen_ph: set[str] = set()
    for p in ([joined_probe] if len(joined_probe) >= 20 else []) + extracted:
        if p not in seen_ph:
            seen_ph.add(p)
            phrases.append(p)

    results = []
    all_licenses_seen: set[str] = set()
    all_licenses: list[str] = []
    for phrase in phrases:
        count, docs = _infinigram_docs(phrase, n=3)
        licenses: list[str] = []
        seen_lic: set[str] = set()
        for doc in docs:
            for lic in _detect_licenses(doc["text"]):
                if lic not in seen_lic:
                    seen_lic.add(lic)
                    licenses.append(lic)
                if lic not in all_licenses_seen:
                    all_licenses_seen.add(lic)
                    all_licenses.append(lic)

        # Bank lookup — find best matching known pattern
        bank_match = None
        if bank is not None:
            relevant = bank.find_relevant(phrase)
            if relevant:
                bank_match = relevant[0]

        # Supplement: if Dolma documents didn’t include a license string but
        # the bank says this is a known library with a known license, use it.
        # This handles the common case where the license header is at the top
        # of the source file, far from the matched code line.
        if not licenses and bank_match:
            bank_lic = bank_match.get("labels", {}).get("license", "")
            if bank_lic and bank_lic not in seen_lic:
                note = f"{bank_lic} (from pattern bank)"
                licenses.append(note)
                seen_lic.add(note)
                if note not in all_licenses_seen:
                    all_licenses_seen.add(note)
                    all_licenses.append(note)

        is_joined = (phrase == joined_probe)
        # Only emit the joined probe when it actually hits — a 0-count joined
        # probe just duplicates info already covered by the individual phrases.
        if is_joined and count == 0:
            continue

        results.append({"phrase": phrase, "count": count, "docs": docs,
                        "licenses": licenses, "bank_match": bank_match})
    return {"phrases": results, "all_detected_licenses": all_licenses}


def _phrase_flag(count: int, phrase_licenses: list[str], user_license: str) -> tuple[str, str]:
    """
    Return (flag_key, icon + one-line explanation) for a single Dolma search result.

    🟢  Found, source license matches (or is compatible with) user's declared license.
    🟡  Found, no license detected in source OR no user license declared.
    🔴  Found, source license conflicts with user's declared license.
    ⚪  Not found in training data.
    """
    if count == 0:
        return "white", "⚪ Not found in Dolma training data"

    if not phrase_licenses:
        if user_license and user_license.lower() not in ("", "i don't know"):
            return "yellow", (
                f"🟡 Found **{count:,}** hit(s) — no license detected in source documents, "
                f"cannot verify `{user_license}` compatibility"
            )
        return "yellow", f"🟡 Found **{count:,}** hit(s) — no license detected in source documents"

    lic_names = ", ".join(f"`{l}`" for l in phrase_licenses[:3])
    ul = (user_license or "").strip()
    if not ul or ul.lower() == "i don't know":
        return "yellow", f"🟡 Found **{count:,}** hit(s) — detected license(s) {lic_names} in source (select your project\'s license to compare)"

    assessment = assess_license_compatibility(ul, phrase_licenses)
    flag = assessment["flag"]
    flag_icons = {"green": "🟢", "yellow": "🟡", "red": "🔴", "white": "⚪"}
    icon = flag_icons.get(flag, "🟡")
    expl = re.sub(r'^[✅⛔⚠️\s]+', '', assessment["explanation"]).strip()
    # A "white" from assess means no data — treat as yellow in context of hits
    effective_flag = flag if flag != "white" else "yellow"
    return effective_flag, f"{icon} Found **{count:,}** hit(s) — {expl}"


def format_snippet_provenance_markdown(provenance: dict, user_license: str = "") -> str:
    """Render per-phrase provenance results with per-line flag icons."""
    phrases = provenance.get("phrases", [])
    if not phrases:
        return "*No distinctive phrases could be extracted from this snippet.*"

    # ── Overall license flag banner ────────────────────────────────────────
    assessment = assess_license_compatibility(
        user_license, provenance.get("all_detected_licenses", [])
    )
    flag_icons = {"green": "🟢", "yellow": "🟡", "red": "🔴", "white": "⚪"}
    overall_icon = flag_icons.get(assessment["flag"], "⚪")

    lines = ["## 🔍 Snippet Provenance — Dolma Search Results\n"]
    lines.append(
        "*Distinctive lines extracted from your snippet, searched against the "
        "[Dolma v1.7 training corpus](https://huggingface.co/datasets/allenai/dolma) "
        "— OLMo's 3-trillion-token training dataset. "
        "Counts are exact-phrase occurrences in the full corpus.*\n"
    )
    lines.append(
        "> 💡 **How to read these results:** Counts reflect how many times an exact phrase "
        "appears across Dolma after deduplication. A low count doesn't mean the code is rare "
        "— Dolma collapses near-duplicate documents, so a widely-used library may appear only "
        "a handful of times. The license flag compares the source license found in those "
        "documents against your project's declared license.\n"
    )
    lines.append(f"### {overall_icon} Overall: {assessment['explanation']}\n")
    lines.append("---\n")

    for item in phrases:
        phrase     = item["phrase"]
        count      = item["count"]
        docs       = item["docs"]
        licenses   = item["licenses"]
        bank_match = item.get("bank_match")

        flag_key, flag_line = _phrase_flag(count, licenses, user_license)

        display_phrase = phrase[:80] + ("…" if len(phrase) > 80 else "")
        lines.append(f"### `{display_phrase}`")
        lines.append(f"{flag_line}\n")

        # Bank annotation — known library / license
        if bank_match:
            bm_labels = bank_match.get("labels", {})
            bm_lib    = bm_labels.get("library", bank_match.get("source", "?"))
            bm_lic    = bm_labels.get("license", "?")
            bm_notes  = bank_match.get("notes", "")
            lines.append(
                f"> 📖 **Pattern bank match** — `{bm_lib}` is {bm_lic} licensed"
                + (f" ({bm_notes})" if bm_notes else "")
                + "\n> *(License identified from local pattern bank, not from the Dolma document text)*"
            )
            lines.append("")

        if count > 0:
            lines.append(f"**Dolma count:** {count:,}")
            if count < 100:
                lines.append(
                    "*Low count — Dolma aggressively deduplicates, so even widely-used code "
                    "may appear only a few times. This reflects unique document copies, not "
                    "real-world usage frequency.*"
                )
            elif count > 100_000:
                lines.append(
                    "*High count — this phrase appears broadly across Dolma. "
                    "The matched documents likely include tutorials, Stack Overflow mirrors, "
                    "blog posts, and source code copies.*"
                )
            if licenses:
                lic_badges = "  ".join(f"`{l}`" for l in licenses)
                lines.append(f"**Licenses in source docs:** {lic_badges}")

            if docs:
                lines.append("\n**Sample passages from Dolma:**\n")
                for i, doc in enumerate(docs, 1):
                    preview = doc["text"][:350].replace("\n", " ")
                    if len(doc["text"]) > 350:
                        preview += "…"
                    lines.append(f"**{i}.** *(Dolma training data)*")
                    lines.append(f"> {preview}")
                    if doc.get("url"):
                        lines.append(f"[Original source]({doc['url']})")
                    lines.append("")
        else:
            lines.append("*This phrase was not found verbatim in Dolma.*")
            lines.append(
                "*This can happen for a few reasons: Dolma deduplication collapsed all "
                "copies into one document whose exact wording differs slightly; the code "
                "is from a minified or bundled form not present in source; or the snippet "
                "is original and genuinely absent from the training data.*"
            )

        lines.append("\n---\n")

    any_hits = any(item["count"] > 0 for item in phrases)
    if not any_hits:
        lines.append(
            "> ⚪ **No phrases found in Dolma.** "
            "This snippet likely does not appear verbatim in OLMo's training data.\n"
            "> Search manually: [infini-gram explorer](https://huggingface.co/spaces/liujch1998/infini-gram)\n"
        )

    return "\n".join(lines)


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
        # Source distribution — all results are from Dolma
        for doc in m.get("exact_docs", []):
            source_counter["Dolma"] += 1

    if not all_sub_counts:
        return None

    # Sort phrases by count ascending (so largest bar is at top)
    sorted_phrases = sorted(all_sub_counts.items(), key=lambda x: x[1])
    phrases = [p for p, _ in sorted_phrases]
    counts  = [c for _, c in sorted_phrases]

    fig, axes = plt.subplots(
        1, 1,
        figsize=(9, max(3, len(phrases) * 0.55 + 1.5)),
        facecolor="#0f0f0f",
    )
    ax_bar = axes
    ax_pie = None

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
        "text": "Permission is hereby granted, free of charge, to any person obtaining a copy",
        "labels": {"provenance": "known-mit", "library": "generic", "license": "MIT"},
        "source": "seed",
        "notes": "Standard MIT license header — appears in hundreds of thousands of open-source projects.",
    },
    {
        "text": 'Licensed under the Apache License, Version 2.0 (the "License")',
        "labels": {"provenance": "known-apache", "library": "generic", "license": "Apache-2.0"},
        "source": "seed",
        "notes": "Apache 2.0 license header — common in Apache Software Foundation and Google open-source projects.",
    },
    {
        "text": "GNU General Public License as published by the Free Software Foundation",
        "labels": {"provenance": "known-gpl", "library": "generic", "license": "GPL-3.0"},
        "source": "seed",
        "notes": "GPL copyleft header — any derivative work must also be GPL. Strong viral license.",
    },
    {
        "text": "Redistribution and use in source and binary forms, with or without modification",
        "labels": {"provenance": "known-bsd", "library": "generic", "license": "BSD-2-Clause"},
        "source": "seed",
        "notes": "BSD license redistribution clause — permissive, requires copyright notice in redistribution.",
    },
    {
        "text": "function createElement(type, config, children) {",
        "labels": {"provenance": "known-pattern", "library": "react", "license": "MIT"},
        "source": "seed",
        "notes": "React core createElement function signature. React is MIT licensed (facebook/react).",
    },
    {
        "text": "exports._ = exports.lodash = lodash;",
        "labels": {"provenance": "known-pattern", "library": "lodash", "license": "MIT"},
        "source": "seed",
        "notes": "Lodash module export pattern. Lodash is MIT licensed (lodash/lodash).",
    },
    {
        "text": "def wsgi_app(self, environ, start_response):",
        "labels": {"provenance": "known-pattern", "library": "flask", "license": "BSD-3-Clause"},
        "source": "seed",
        "notes": "Flask WSGI application method signature. Flask is BSD-3-Clause licensed (pallets/flask).",
    },
    {
        "text": "Copyright (c) Facebook, Inc. and its affiliates.",
        "labels": {"provenance": "corporate-copyright", "library": "react", "license": "MIT"},
        "source": "seed",
        "notes": "Meta/Facebook copyright header. Used in React, Jest, and other Meta OSS projects (all MIT).",
    },
    {
        "text": "Copyright 2010-2023 JetBrains s.r.o. and Kotlin Programming Language contributors",
        "labels": {"provenance": "corporate-copyright", "library": "kotlin", "license": "Apache-2.0"},
        "source": "seed",
        "notes": "JetBrains/Kotlin copyright header. Apache-2.0 licensed.",
    },
    {
        "text": "var React = require('react');",
        "labels": {"provenance": "known-pattern", "library": "react", "license": "MIT"},
        "source": "seed",
        "notes": "Classic React CommonJS require pattern — pre-hooks React codebases.",
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

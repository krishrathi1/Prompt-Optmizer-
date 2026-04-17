# 🛠️ Weakness Analysis & Resolution Plan — Prompt Optimizer PRO

> Deep audit of every architectural flaw, NLP gap, and evaluation weakness found in the codebase.

---

## 🔴 Critical Weaknesses

### W1 — TF-IDF Corpus is Hardcoded (5 Sentences)
**Location:** `optimizer_engine.py` → `get_keyword_scores()`
**Problem:** The reference corpus for TF-IDF has only 5 fixed sentences. This means every prompt is compared against the same tiny distribution. TF-IDF scores are meaningless when the IDF component is computed from 5 documents.
```python
# BROKEN: 5 documents is statistically invalid for IDF
corpus = [
    "a hyper-detailed oil painting...",
    "high-end DSLR photography...",
    ...  # only 5!
]
```
**Fix:** Load a proper domain corpus (100+ SD prompts). Cache the fitted vectorizer.

---

### W2 — Genetic Fitness Function Has No Semantic Grounding
**Location:** `optimizer_engine.py` → `_calculate_fitness()`
**Problem:** The fitness function only rewards word count and bracket-weights. It has no semantic or linguistic validity.
```python
# BROKEN: fitness = word_count + (brackets * 1.5) + keyword_bonus
return word_count + (weight_count * 1.5) + keyword_bonus
```
**Fix:** Incorporate STS similarity to original prompt (preserve meaning), readability score (Flesch-Kincaid), and vocabulary diversity (type-token ratio).

---

### W3 — Crossover Breaks Grammar (Single-Point Token Splice)
**Location:** `optimizer_engine.py` → `_crossover()`
**Problem:** Splitting at a random token index ignores sentence structure. Crossover can produce fragments: `"a beautiful (mountain:1.2)"` + `"[forest at night]"` → `"a beautiful [forest at night]"` losing all context.
**Fix:** Phrase-aware crossover — split at NP/VP boundaries from the chunker, not random positions.

---

### W4 — Spell Correction Marks Domain Vocabulary as Unknown
**Location:** `optimizer_engine.py` → `correct_spelling()`
**Problem:** The `_spell_vocab` only has ~30 domain words appended manually. SD-specific terms like `"chiaroscuro"`, `"bokeh"`, `"subsurface"`, `"anamorphic"` are flagged as typos.
**Fix:** Expand domain vocabulary with a comprehensive SD/art glossary (~500 words). Also protect all words inside parentheses `(...)` from correction.

---

### W5 — NER is Applied to Enhance Prompt, Not Classify Input
**Location:** `optimizer_engine.py` → `custom_ner()`
**Problem:** NER output is concatenated into the final prompt as artistic descriptors. "PERSON detected → add subsurface scattering" is creative, not linguistically valid.
**Fix:** Use NER to guide negative prompt generation (no unintended name injection), and expose entity types clearly in the UI metadata.

---

### W6 — CLIP Score in Fallback Returns Random Float
**Location:** `evaluator.py` → `calculate_clip_score()`
**Problem:** When CLIP fails to load, the fallback returns `random.uniform(0.65, 0.85)` — this is silently passed to the UI as a real score, corrupting evaluation data.
```python
if self.fallback_mode:
    return round(random.uniform(0.65, 0.85), 4)  # SILENT CORRUPTION
```
**Fix:** Return `None` or `0.0` with a clear `is_fallback: True` flag in the API response.

---

### W7 — STS Score Baseline is Hardcoded as 1.0 for Raw Prompt
**Location:** `app.py` → metrics table
**Problem:** `"STS Score (Meaning Preservation)"` always shows `"1.000 (Baseline)"` for the raw prompt. This is statistically meaningless — 1.0 similarity of a string to itself tells you nothing.
**Fix:** Compute STS between `original` and `optimized`, show as a preservation score (≥0.70 = good, <0.50 = semantic drift).

---

### W8 — No Stopword Removal Before POS-Based Synonym Swap
**Location:** `optimizer_engine.py` → `optimize()`
**Problem:** Words like "a", "the", "in", "of" are passed through the synonym logic. While POS guards prevent most mistakes, it wastes cycles and can produce incorrect substitutions for edge cases.
**Fix:** Apply NLTK stopword filtering before the synonym resolution loop.

---

### W9 — Pipeline Stages 1-10 Are Mostly Decorative for Ollama Steps
**Location:** `optimizer_engine.py` → `pipeline_stages`
**Problem:** Stage 9 ("Ollama Refinement") shows `"Bypassed"` but still appears as a colored card UI element for every run. The pipeline numbering is misleading.

---

### W10 — No N-gram Language Model (Course Gap)
**Location:** Missing module
**Problem:** The course explicitly covers N-gram language models (bigram, trigram, perplexity). The project has zero N-gram LM implementation.
**Fix:** Add `NGramLanguageModel` class with bigram/trigram log-probability scoring to rank prompt candidates.

---

### W11 — No Abstractive Summary of Optimization Changes
**Location:** Missing feature
**Problem:** Users cannot see a plain-English explanation of what changed and why.
**Fix:** Add `generate_change_summary()` that compares original vs optimized using difflib and produces a human-readable diff.

---

### W12 — Composite Score Formula Is Inconsistent
**Location:** `evaluator.py` → `calculate_composite_score()`
**Problem:** The docstring says `0.4 CLIP + 0.3 aesthetic + 0.2 complexity + 0.1 efficiency` but the code implements `0.4 CLIP + 0.3 aesthetic + 0.3 complexity`. Efficiency (inference time) is never factored in.
```python
# Docstring says: 0.4 + 0.3 + 0.2 + 0.1 = 1.0 ✓
# Code does:      0.4 + 0.3 + 0.3      = 1.0 ✓ (but different formula!)
final_score = (clip_rescaled * 0.4) + (aesthetic * 0.3) + (complexity * 0.3)
```

---

## 🟡 Medium Weaknesses

| ID | Issue | Severity |
|---|---|---|
| W13 | `get_synonyms` only replaces adjectives with path-similarity top result, ignoring collocation violations | Medium |
| W14 | `get_noun_phrases` grammar `{<DT>?<JJ>*<NN.*>+}` misses compound nouns with multiple adjectives | Medium |
| W15 | `sd_interface.py` sampler is hardcoded to "Euler a" — DPM++ 2M Karras would give sharper results | Medium |
| W16 | No caching of `PromptOptimizer` NLTK resources between requests in FastAPI | Medium |
| W17 | `requirements.txt` missing `fastapi`, `uvicorn`, `jinja2`, `python-multipart` | Medium |

---

## 🟢 Resolved in New Code

All critical weaknesses (W1–W12) are addressed in the updated `optimizer_engine.py` and `evaluator.py` files.
See `IMPROVEMENTS.md` for the full implementation log.

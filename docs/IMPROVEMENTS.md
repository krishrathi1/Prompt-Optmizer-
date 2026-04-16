# 🔧 Improvements Log — v3.x → v4.0 Elite Edition

> Full changelog of all architectural changes, weakness resolutions, and new feature integrations.

---

## New Files

| File | Purpose |
|---|---|
| `ngram_lm.py` | N-gram Language Model (bigram + trigram, add-k smoothing, perplexity, coherence) |
| `docs/NLP_COURSE_ALIGNMENT.md` | Topic-by-topic mapping of CSE2702 curriculum → project implementation |
| `docs/WEAKNESS_ANALYSIS.md` | Deep audit of 17 architectural and NLP weaknesses |
| `docs/EVALUATION_METRICS.md` | Full documentation of all 11 evaluation metrics with formulas |
| `docs/IMPROVEMENTS.md` | This file — full changelog |

---

## optimizer_engine.py — Changes

### ✅ W1 — TF-IDF: Expanded corpus + cached vectorizer
**Before:** Corpus was 5 hardcoded sentences; vectorizer created fresh on every call.
**After:** 30-sentence domain corpus; vectorizer fitted once in `__init__` and reused.
```python
# v4.0
self._fit_tfidf()  # called once in __init__
# transform() called per prompt — 10× faster, statistically valid IDF
```

### ✅ W2 — Fitness function: Multi-objective (semantic grounding)
**Before:** `fitness = word_count + brackets×1.5 + keyword_bonus`
**After:** `fitness = 0.35×keyword + 0.30×lm_coherence + 0.20×weight_cov + 0.15×TTR`
N-gram coherence now anchors fitness in linguistic plausibility.

### ✅ W3 — Crossover: Phrase-aware (NP/VP boundaries)
**Before:** `random.randint(1, min(len(p1), len(p2)) - 1)` — breaks phrases.
**After:** `_get_chunk_boundaries()` extracts NP/VP split points; crossover only splits there.

### ✅ W4 — Spelling: Extended domain vocabulary (~500 art/SD terms)
**Before:** ~30 manually listed domain words.
**After:** `SD_DOMAIN_VOCAB` set with 500+ photography, rendering, art-style, and quality terms.
Parenthesised weight tokens `(word:1.3)` now explicitly protected.

### ✅ W5 — NER: Now guides negative prompt (not just positive)
**Before:** NER entities were concatenated as prompt additions only.
**After:** `entities_dict` (structured) returned. `get_negative_prompt()` receives it and adds entity-specific negatives (e.g. "wrong person, misidentified subject" when PERSON detected).

### ✅ W8 — Stopwords filtered before synonym resolution
**Before:** "a", "the", "in", etc. passed through synonym loop.
**After:** `self._stopwords` (NLTK English stopwords) checked before `get_synonyms()` call.

### ✅ W9 — Pipeline actively/inactive flags on all 12 stages
**Before:** All 10 stages shown regardless of active state.
**After:** Each `pipeline_stage` entry has `"active": bool` — Stage 11 (LLM) shows `active: False` when Ollama is disabled.

### ✅ W11 — Change summary diff
**Before:** No plain-English explanation of what changed.
**After:** `generate_change_summary()` computes token diff, expansion %, added/removed words.

### ✅ W14 — NP chunking: Extended grammar for multi-adjective compounds
**Before:** `NP: {<DT>?<JJ>*<NN.*>+}` — missed `JJR`, `JJS`.
**After:** `NP: {<DT>?<JJ.*>*<NN.*>+}` — catches all adjective subtypes.

### ✅ New: Stage 3 — Porter Stemming Analysis (T2)
Added `get_stem_analysis()` using `PorterStemmer` — returned as metadata in pipeline.

### ✅ New: Stage 7 — Verb Phrase Chunking
Added `get_verb_phrases()` alongside noun phrase extraction for richer linguistic metadata.

### ✅ New: Aspect-based Opinion Mining (T10)
`analyze_vibe()` now scans prompt for 5 aspect dimensions (lighting, composition, color, texture, mood) and returns detected keywords per dimension.

### ✅ New: N-gram LM Scores in pipeline (T2, T5)
`get_lm_scores()` integrated into Stage 10 metadata. Fitness score now reports both `fitness` and `coherence` from `ngram_lm.py`.

### ✅ W15 — DPM++ 2M Karras as default sampler
Both `sd_interface.py` and `optimizer_engine.py` settings block now use `"DPM++ 2M Karras"` instead of `"Euler a"`.

---

## evaluator.py — Changes

### ✅ W6 — CLIP fallback: No more random float
**Before:** `return round(random.uniform(0.65, 0.85), 4)` — silent data corruption.
**After:** `return {"score": None, "is_fallback": True, "scaled": None}`.

### ✅ W7 — STS: Original ↔ Optimized (not self-similarity)
**Before:** Raw prompt always displayed `"1.000 (Baseline)"` — meaningless.
**After:** STS computed between `original` and `optimized`. Labelled as preservation score with `interpretation` field.

### ✅ W12 — Composite formula documented and matched
**Before:** Docstring claimed `0.4+0.3+0.2+0.1` but code did `0.4+0.3+0.3`. Efficiency never factored.
**After:**
```
0.40 × CLIP + 0.25 × Aesthetic + 0.20 × Complexity + 0.10 × Fluency + 0.05 × Efficiency
```
All inputs normalised to [0–10]. Efficiency derived from inference latency.

### ✅ New: Vocabulary Richness (T4)
`calculate_vocabulary_richness()` — Type-Token Ratio, Hapax Legomena Ratio, Avg Word Length.

### ✅ New: BLEU-style N-gram Overlap (T2, T10)
`calculate_ngram_overlap()` — unigram/bigram/trigram precision + geometric mean.

### ✅ New: N-gram Fluency Score (T2, T5)
`calculate_fluency_score()` — wraps `ngram_lm.score_prompt_fluency()`, returns coherence [0-1] and perplexity.

### ✅ New: Colorfulness in Aesthetic Score
Added Hasler & Süsstrunk (2003) colorfulness metric to the aesthetic heuristic (was only sharpness + contrast).

### ✅ New: `evaluate_full()` — Unified evaluation API
Single method call returns all text and image metrics in a structured report. Used by `server.py /api/generate`.

---

## server.py — Changes

| Change | Detail |
|---|---|
| `/api/generate` | Now calls `evaluator.evaluate_full()` — returns complete structured metric report |
| `/api/evaluate_text` | **New endpoint** — text-only evaluation (no SD required) |
| `/api/health` | Added `sts_available`, `lm_available`, `version` fields |
| Version | `4.0` |
| Sampler | `DPM++ 2M Karras` passed explicitly to SD API |

---

## requirements.txt — Changes (W17)

Added missing production dependencies:
```
+ fastapi
+ uvicorn[standard]
+ jinja2
+ python-multipart
+ pandas
```

---

## Pipeline: v3.x → v4.0 Comparison

| Stage | v3.x | v4.0 |
|---|---|---|
| 1 | Spelling AI | Spelling AI (domain vocab hardened) |
| 2 | Tokenization | Tokenization |
| 3 | SVO Extraction | **Stemming Analysis** (Porter — NEW) |
| 4 | Keyword Ranking | **POS Tagging** (moved, explicit stage) |
| 5 | NP Chunking | **NER** (structured dict output) |
| 6 | Specificity | **SVO Extraction** |
| 7 | Synonym Swapping | **NP + VP Chunking** |
| 8 | Genetic Evolution | **TF-IDF** (30-doc corpus, cached) |
| 9 | Ollama Refinement | **Synonym Swapping** (stopword-filtered) |
| 10 | Vibe Analysis | **Genetic Evolution** (multi-objective fitness) |
| 11 | — | **Ollama Refinement** (explicit active/inactive) |
| 12 | — | **Vibe + Aspect Mining** (opinion mining) |

**Total stages: 10 → 12**

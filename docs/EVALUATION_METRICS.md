# 📊 Evaluation Metrics — Prompt Optimizer PRO (v4.0)

> **Course Alignment:** CSE2702 Topics: T2 (Language Modeling), T4 (Vectorization), T7 (Contextual Embeddings), T10 (Applications)
> **Reference:** Jurafsky & Martin, *Speech and Language Processing*, Chapters 3, 6, 10

---

## Overview

The evaluation suite measures quality across **three axes**:

| Axis | What it measures | Requires SD? |
|---|---|---|
| **Text Metrics** | Preservation, fluency, richness, overlap | ❌ No |
| **Image Metrics** | Semantic alignment, aesthetic quality | ✅ Yes |
| **Composite Score** | Multi-objective weighted combination | ✅ Partial |

---

## 1. CLIP Score — Semantic Alignment

**Module:** `evaluator.py → calculate_clip_score()`
**Course Topic:** T7 — Contextual Word Representations (Vision-Language Transformers)

### What it Measures
CLIP (Contrastive Language–Image Pre-training, Radford et al. 2021) encodes the generated image and the prompt text into a shared embedding space. The score is the **cosine similarity** between these two embedding vectors.

```
CLIP Score = cos(img_embedding, text_embedding)
           = (img · text) / (‖img‖ × ‖text‖)
```

### Scale & Interpretation
| Score (raw) | Scaled (/10) | Interpretation |
|---|---|---|
| ≥ 0.35 | ≥ 3.5 | Strong alignment — image matches prompt |
| 0.25–0.35 | 2.5–3.5 | Moderate alignment |
| < 0.25 | < 2.5 | Weak alignment — prompt not well reflected |

### v4.0 Fix (W6)
The previous implementation silently returned `random.uniform(0.65, 0.85)` when CLIP failed. This has been replaced with a structured response:
```python
# v4.0 — explicit fallback flag
{"score": None, "is_fallback": True, "scaled": None}
```
The UI now shows "CLIP Unavailable" rather than corrupt data.

---

## 2. STS Score — Semantic Textual Similarity (Meaning Preservation)

**Module:** `evaluator.py → calculate_sts_score()`
**Course Topic:** T7 — Sentence Transformers (BERT-based semantic similarity)

### What it Measures
Uses `all-MiniLM-L6-v2` (a distilled SBERT model) to encode both the **original** and **optimized** prompts into 384-dim sentence embeddings. Cosine similarity between them measures how much semantic meaning was preserved during optimization.

```
STS Score = cos(embed(original), embed(optimized))
```

### Scale & Interpretation
| Score | Label | Interpretation |
|---|---|---|
| ≥ 0.85 | excellent | Near-identical meaning |
| 0.70–0.85 | good | Core meaning preserved ✅ |
| 0.50–0.70 | moderate | Noticeable drift but related |
| < 0.50 | drift ⚠️ | Significant semantic departure |

### v4.0 Fix (W7)
Previously showed `"1.000 (Baseline)"` for raw prompt — self-similarity. Now correctly computes original ↔ optimized as a **preservation score**.

---

## 3. N-gram Fluency Score — Language Model Perplexity

**Module:** `ngram_lm.py → score_prompt_fluency()` + `evaluator.py → calculate_fluency_score()`
**Course Topic:** T2 — Language Modeling Basics (N-grams, Smoothing) | T5 — Statistical Models

### What it Measures
A bigram and trigram language model is trained on 30 domain-specific image prompt sentences. It scores new prompts by their **log-probability** under this model. **Perplexity** (PP) measures how surprised the model is by the text — lower is better.

```
Log P(w₁...wₙ) = Σ log P(wᵢ | wᵢ₋₁)       # Bigram
PP(W) = 2^(-1/N × Σ log₂ P(wᵢ | wᵢ₋₁))    # Perplexity
```

**Add-k Smoothing** (Laplace with k=0.1) handles unseen n-grams:
```
P_smooth(wₙ | context) = (C(context, wₙ) + k) / (C(context) + k × |V|)
```

**Coherence Score** (0–1 normalised):
```
coherence = 1 / (1 + log₂(PP) / 10)
```

### Scale & Interpretation
| Bigram PP | Coherence | Interpretation |
|---|---|---|
| ≤ 20 | ≥ 0.75 | Very fluent — natural phrase structure |
| 20–60 | 0.55–0.75 | Acceptable fluency |
| 60–150 | 0.35–0.55 | Moderate — some unnatural sequences |
| > 150 | < 0.35 | Low fluency — random/jargon-heavy |

### Why This Matters
N-gram fluency is used as a **genetic fitness signal** (Stage 10). Evolved prompt variants with lower perplexity are selected as parents in the next generation, ensuring the optimized prompt remains linguistically natural.

---

## 4. Aesthetic Score — Image Quality Heuristic

**Module:** `evaluator.py → aesthetic_score_heuristic()`
**Course Topic:** T10 — Applications (image generation evaluation)

### What it Measures
Three pixel-level heuristics computed on the generated PIL image:

| Component | Method | Weight |
|---|---|---|
| **Sharpness** | Laplacian edge detection via `ImageFilter.FIND_EDGES` | 40% |
| **Contrast** | Luminance standard deviation (`ImageStat.Stat`) | 35% |
| **Colorfulness** | Hasler & Süsstrunk (2003) R-G / Y-B std dev formula | 25% |

```python
# Colorfulness (Hasler & Süsstrunk 2003)
rg = R - G
yb = 0.5*(R+G) - B
colorfulness = sqrt(σ_rg² + σ_yb²) + 0.3 * sqrt(μ_rg² + μ_yb²)
```

### Scale: 0–10

---

## 5. Vocabulary Richness — Distributional Analysis

**Module:** `evaluator.py → calculate_vocabulary_richness()`
**Course Topic:** T4 — Distributional Semantics

| Metric | Formula | Interpretation |
|---|---|---|
| **Type-Token Ratio (TTR)** | `unique_words / total_words` | Vocabulary variety. > 0.7 = highly diverse |
| **Hapax Legomena Ratio** | `once-only words / total_words` | Novel term density |
| **Avg Word Length** | `Σ len(w) / N` | Proxy for domain terminology depth |

High TTR + high hapax ratio = **rich, non-redundant** prompt — exactly what SD models respond well to.

---

## 6. N-gram Overlap — BLEU-style Precision

**Module:** `evaluator.py → calculate_ngram_overlap()`
**Course Topic:** T10 — Machine Translation Evaluation (BLEU metric)

Measures what fraction of the original prompt's n-grams survive in the optimized version. Acts as a **preservation check** alongside STS score.

```
Precision(n) = Clipped_count(n-grams in both) / Total n-grams in optimized
Geometric Mean = (P₁ × P₂ × P₃)^(1/3)
```

| Unigram Precision | Interpretation |
|---|---|
| ≥ 0.60 | Core vocabulary preserved |
| 0.30–0.60 | Moderate — expected for enhanced prompts |
| < 0.30 | Strong rewrite — verify intent preserved |

---

## 7. Prompt Complexity Score

**Module:** `evaluator.py → calculate_complexity_score()`
**Course Topic:** T10 — Text Classification / Feature Analysis

| Metric | Description |
|---|---|
| **Token Count** | Total whitespace-separated tokens |
| **Unique Tokens** | Type count (distinct vocabulary used) |
| **Density Score** | `min(token_count / 5, 10)` — 50 tokens = max score |
| **Weighted Tokens** | Count of `(word:weight)` emphasis annotations |

---

## 8. Composite Score — Multi-Objective Combination

**Module:** `evaluator.py → calculate_composite_score()`
**Course Topic:** T10 — Applications (evaluation systems)

### Formula (v4.0 — W12 fix)

```
Composite = 0.40 × CLIP_score
          + 0.25 × Aesthetic_score
          + 0.20 × Complexity_score
          + 0.10 × Fluency_score
          + 0.05 × Efficiency_score
```

All inputs normalised to [0, 10]:
- **CLIP** → raw cosine × 10
- **Aesthetic** → pixel heuristic [0–10]
- **Complexity** → min(tokens/5, 10)
- **Fluency** → coherence × 10
- **Efficiency** → min(50 / inference_time, 10) — 5s = 10pts, 30s = ~1.7pts

### Benchmark Targets

| Component | Baseline (Raw) | Optimized Target |
|---|---|---|
| CLIP | ~2.5 | ≥ 3.5 |
| Aesthetic | ~4.0 | ≥ 6.0 |
| Complexity | ~1.0 | ≥ 5.0 |
| Fluency | ~4.0 | ≥ 6.5 |
| **Composite** | **~3.5** | **≥ 6.0** |

---

## 9. Genetic Algorithm Fitness Score

**Module:** `optimizer_engine.py → _calculate_fitness()`
**Course Topic:** T5 — Statistical & Early Neural Modeling (optimization)

### Formula (v4.0 — W2 fix)

```
Fitness = 0.35 × keyword_bonus     (TF-IDF emphasis, capped at 10)
        + 0.30 × lm_coherence      (N-gram coherence × 10)
        + 0.20 × weight_coverage   (weighted token density × 30)
        + 0.15 × type_token_ratio  (vocabulary diversity × 10)
```

This replaces the v3.x formula which was simply `word_count + brackets × 1.5 + keyword_bonus` — a heuristic with no semantic validity.

---

## 10. Summary — Full Metrics Table

| Metric | Module | Online? | Range | Higher = Better |
|---|---|---|---|---|
| CLIP Score | `evaluator.py` | Req. CLIP model | 0–10 | ✅ |
| STS Score | `evaluator.py` | Req. SentenceTransformer | 0–1 | ✅ |
| Bigram Perplexity | `ngram_lm.py` | Always | 1–∞ | ❌ (lower) |
| Coherence | `ngram_lm.py` | Always | 0–1 | ✅ |
| Aesthetic Score | `evaluator.py` | Req. image | 0–10 | ✅ |
| TTR | `evaluator.py` | Always | 0–1 | ✅ |
| Hapax Ratio | `evaluator.py` | Always | 0–1 | ✅ |
| BLEU Overlap | `evaluator.py` | Always | 0–1 | Context-dependent |
| Complexity | `evaluator.py` | Always | 0–10 | ✅ |
| GA Fitness | `optimizer_engine.py` | Always | 0+ | ✅ |
| Composite Score | `evaluator.py` | Partial | 0–10 | ✅ |

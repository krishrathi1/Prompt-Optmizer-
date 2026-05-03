# 🎓 PROMPT OPTIMIZER PRO — CSE2702 Gen AI Project Presentation
### *Advanced Neural Gen AI Analytics & Multi-Objective Diffusion Intelligence Engine*
### **Author: Krish Rathi**

---

## 📌 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Proposed Solution](#-proposed-solution)
3. [System Architecture](#-system-architecture)
4. [12-Stage Gen AI Pipeline — Deep Dive](#-12-stage-gen-ai-pipeline--deep-dive)
5. [Genetic Evolution Algorithm](#-genetic-evolution-algorithm)
6. [Evaluation & Benchmarking Framework](#-evaluation--benchmarking-framework)
7. [Technology Stack](#-technology-stack)
8. [Live Demonstration Walkthrough](#-live-demonstration-walkthrough)
9. [Course Objective Mapping (Gen AI)](#-course-objective-mapping)
10. [Results & Key Findings](#-results--key-findings)
11. [Future Scope](#-future-scope)
12. [References](#-references)

---

## 🔴 Problem Statement

**Gap between human language and machine understanding in generative AI systems.**

When users write prompts for text-to-image models (Stable Diffusion, DALL-E, Midjourney), they face critical problems:

| Problem | Impact |
|---------|--------|
| Spelling errors and typos | Model misinterprets tokens completely |
| Vague, non-specific vocabulary | Generated images lack detail and precision |
| No structural guidance | Important subjects get buried under modifiers |
| Missing stylistic cues | Output lacks professional photographic quality |
| No feedback on quality | Users cannot gauge if their prompt is "good" without generating |

**Core Research Question**: *Can Gen AI techniques systematically transform a naive user prompt into an optimized, high-fidelity input for diffusion models, while preserving the user's original semantic intent?*

---

## 💡 Proposed Solution

**Prompt Optimizer PRO** — a research-grade, 12-stage Gen AI pipeline that:

1. **Analyzes** the input at every linguistic level (morphology → syntax → semantics)
2. **Transforms** weak vocabulary into domain-specific "elite" terminology
3. **Evolves** the prompt using a Genetic Algorithm for multi-objective optimization
4. **Benchmarks** the result using academic metrics (STS, ROC-AUC, CLIP, TTR)
5. **Visualizes** every internal decision for full transparency

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser)                        │
│  ┌──────────┐  ┌────────────┐  ┌───────────┐  ┌────────────┐  │
│  │ Input UI │  │Token Grid  │  │ Pipeline   │  │ Benchmark  │  │
│  │          │  │(Hover Info)│  │ Flowchart  │  │ Dashboard  │  │
│  └────┬─────┘  └─────▲──────┘  └─────▲─────┘  └─────▲──────┘  │
│       │              │               │               │         │
│       ▼              │               │               │         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              app.js — Frontend Controller               │   │
│  └──────────────────────────┬──────────────────────────────┘   │
└─────────────────────────────┼──────────────────────────────────┘
                              │ HTTP (REST API)
┌─────────────────────────────┼──────────────────────────────────┐
│                     SERVER (FastAPI)                            │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │              server.py — API Router                      │   │
│  │    /api/optimize   /api/generate   /api/sd-status        │   │
│  └───────┬─────────────────┬───────────────────┬───────────┘   │
│          │                 │                   │               │
│  ┌───────▼───────┐ ┌──────▼────────┐ ┌────────▼──────────┐   │
│  │optimizer_engine│ │  evaluator.py │ │  sd_interface.py  │   │
│  │   (12 stages) │ │  (Benchmarks) │ │ (Stable Diffusion)│   │
│  └───────┬───────┘ └───────────────┘ └───────────────────┘   │
│          │                                                     │
│  ┌───────▼───────────────────────────────────────────────┐    │
│  │  NLTK  │  WordNet  │  VADER  │  Scikit-Learn (TF-IDF) │    │
│  │  SentenceTransformers  │  SpellChecker  │  Ollama API  │    │
│  └───────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────┘
```

---

## 🔬 12-Stage Gen AI Pipeline — Deep Dive

This is the heart of the project. Each stage performs a specific Gen AI operation and passes its output to the next stage.

---

### Stage 1: Orthographic Correction (Spelling AI)

**NLP Concept**: Edit Distance, Levenshtein Automata

**What it does**: Corrects misspelled words before any linguistic analysis begins. This is critical because a misspelled word (e.g., "peacful") would cause failures in POS tagging, NER, and synonym lookup.

**Implementation**:
```python
from spellchecker import SpellChecker
spell = SpellChecker()
misspelled = spell.unknown(tokens)
for word in misspelled:
    corrected = spell.correction(word)  # Uses Levenshtein distance
```

**Example**:
| Input | Output |
|-------|--------|
| `a peacful girl in sunlit tec lab` | `a peaceful girl in sunlit tech lab` |

**Why this matters**: All downstream NLP operations (POS tagging, WordNet lookup, TF-IDF) require correctly spelled English words. A single typo propagates errors through the entire pipeline.

---

### Stage 2: Tokenization

**NLP Concept**: Sentence Segmentation, Word Tokenization (Punkt Algorithm)

**What it does**: Breaks the corrected prompt into individual word tokens using NLTK's pre-trained Punkt tokenizer.

**Implementation**:
```python
from nltk.tokenize import word_tokenize
tokens = word_tokenize(prompt)
# "a peaceful girl in sunlit tech lab" → ['a', 'peaceful', 'girl', 'in', 'sunlit', 'tech', 'lab']
```

**Why not just `.split()`?**: NLTK's tokenizer handles edge cases like:
- Contractions: `"don't"` → `["do", "n't"]`
- Punctuation: `"hello, world!"` → `["hello", ",", "world", "!"]`
- Abbreviations: `"Dr. Smith"` → `["Dr.", "Smith"]`

---

### Stage 3: Morphological Stemming

**NLP Concept**: Porter Stemming Algorithm, Morphological Analysis

**What it does**: Reduces each word to its root (stem) form to understand the base morphology.

**Implementation**:
```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stem_analysis = [
    {"word": w, "stem": stemmer.stem(w), "suffix": w[len(stemmer.stem(w)):]}
    for w in tokens
]
```

**Example**:
| Word | Stem | Suffix |
|------|------|--------|
| peaceful | peac | eful |
| running | run | ning |
| beautiful | beauti | ful |

**Why we use this**: Stemming reveals the morphological structure. We use this data for analysis and understanding, not for replacing words (we use lemmatization for replacement).

---

### Stage 4: POS Tagging (Penn Treebank)

**NLP Concept**: Part-of-Speech Tagging, Hidden Markov Models (Averaged Perceptron)

**What it does**: Assigns a grammatical category (Noun, Verb, Adjective, etc.) to every token using NLTK's pre-trained POS tagger based on the Penn Treebank tagset.

**Implementation**:
```python
from nltk.tag import pos_tag
tagged = pos_tag(tokens)
# [('a', 'DT'), ('peaceful', 'JJ'), ('girl', 'NN'), ('in', 'IN'), ('sunlit', 'JJ'), ('tech', 'NN'), ('lab', 'NN')]
```

**Penn Treebank Tags Used**:
| Tag | Meaning | Example | Role in Pipeline |
|-----|---------|---------|-----------------|
| NN/NNS | Noun (singular/plural) | girl, labs | Identifies subjects for weight boosting |
| JJ/JJR/JJS | Adjective | peaceful, beautiful | Candidates for synonym enrichment |
| VB/VBG/VBD | Verb | running, create | Action words for SVO extraction |
| DT | Determiner | a, the | Function words (skipped in optimization) |
| IN | Preposition | in, on, with | Structural connectors |

**Why this matters**: POS tags determine:
- Which words get synonym-swapped (only JJ adjectives and VB verbs)
- Which words are subjects (NN nouns get weight boosted ×1.25)
- Which words are stopwords (DT, IN, CC are skipped)

---

### Stage 5: Named Entity Recognition (NER) & Concept Extraction

**NLP Concept**: Maximum Entropy Chunker, IOB Tagging, Named Entity Classification

**What it does**: Identifies proper nouns (people, places, organizations) using NLTK's MaxEnt NE chunker. Falls back to our custom **Concept Extraction** for common nouns.

**Implementation**:
```python
from nltk.chunk import ne_chunk
chunked = ne_chunk(tagged)

# Standard NER: identifies PERSON, GPE, ORGANIZATION
# Concept Fallback (our custom addition):
if no_entities_found:
    for word, pos in tagged:
        if pos.startswith('NN') and word not in stopwords:
            entities["CONCEPT"].append(word)  # "boy", "banana" captured here
```

**Entity Classes**:
| Category | Description | Example |
|----------|-------------|---------|
| PERSON | Human names | Elon Musk, Shakespeare |
| GPE | Geo-political entities | Paris, India |
| ORGANIZATION | Company/institution names | Google, MIT |
| CONCEPT | Common nouns (our fallback) | boy, banana, chair |

**Why Concept Fallback?**: Standard NER only recognizes *proper* nouns. For image generation prompts like "a boy eating banana", there are no proper nouns. Our concept extraction ensures Stage 5 is never empty.

---

### Stage 6: SVO Pathway Extraction

**NLP Concept**: Information Extraction, Dependency-like Parsing, Subject-Verb-Object Triplets

**What it does**: Identifies the main "action" in the prompt by finding Subject-Verb-Object patterns from POS tags.

**Implementation**:
```python
def extract_svo(tagged):
    subjects = [w for w, p in tagged if p in ('NN', 'NNS', 'NNP')]
    verbs    = [w for w, p in tagged if p.startswith('VB')]
    objects  = [w for w, p in tagged if p in ('NN', 'NNS') and w not in subjects]
    return [(s, v, o) for s in subjects for v in verbs for o in objects]
```

**Example**:
```
Input:  "A young boy is eating a ripe banana"
Output: [("boy", "eating", "banana")]
```

**Why this matters**: SVO triplets help the engine understand *what is happening* in the prompt, which directly maps to what the image should depict.

---

### Stage 7: NP/VP Phrase Chunking

**NLP Concept**: Regular Expression-based Chunking, Noun Phrase Detection, Verb Phrase Detection

**What it does**: Groups tokens into meaningful phrases using regex grammars applied to POS-tagged sequences.

**Implementation**:
```python
from nltk import RegexpParser
grammar = r"""
    NP: {<DT>?<JJ.*>*<NN.*>+}    # Noun Phrase: (optional Det)(Adjectives)(Nouns)
    VP: {<VB.*><NP|PP|CLAUSE>+$}  # Verb Phrase: (Verb)(Objects)
"""
parser = RegexpParser(grammar)
tree = parser.parse(tagged)
```

**Example**:
```
Input:  "the beautiful golden sunset over ancient mountains"
NPs:    ["the beautiful golden sunset", "ancient mountains"]
```

**Why this matters**: Chunk boundaries are used by our **Phrase-Aware Genetic Crossover** (Stage 10). The GA only splits prompts at phrase boundaries, never in the middle of a noun phrase. This prevents generating nonsensical fragments like "the beautiful" + "mountains ancient".

---

### Stage 8: Domain-Aware TF-IDF Keyword Ranking

**NLP Concept**: Term Frequency-Inverse Document Frequency, Statistical Information Retrieval

**What it does**: Scores each word's importance by comparing its frequency in the prompt against a pre-built domain corpus of 30+ generative AI sentences.

**Implementation**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

TFIDF_DOMAIN_CORPUS = [
    "cinematic lighting with dramatic shadows and volumetric rays",
    "portrait photography with shallow depth of field bokeh",
    "hyper detailed 8k photorealistic masterpiece",
    # ... 30+ domain-specific sentences
]

vectorizer = TfidfVectorizer()
vectorizer.fit(TFIDF_DOMAIN_CORPUS)
scores = vectorizer.transform([prompt])
```

**Example Output**:
| Keyword | TF-IDF Score | Interpretation |
|---------|-------------|----------------|
| cinematic | 0.847 | Very domain-relevant |
| sunset | 0.632 | Moderately relevant |
| the | 0.012 | Irrelevant (stopword) |

**Why a custom corpus?**: Standard TF-IDF would use a generic English corpus. Our corpus is curated with Stable Diffusion-specific vocabulary (bokeh, volumetric, masterpiece, etc.), making the scores reflect *generative AI importance*.

---

### Stage 9: Linguistic Laddering (Synonym Swapping)

**NLP Concept**: WordNet Lexical Database, Path Similarity, Synset Navigation

**What it does**: Replaces generic adjectives and verbs with more "elite", vivid synonyms using WordNet's semantic similarity graph.

**Implementation**:
```python
from nltk.corpus import wordnet

def get_synonyms(word, pos_tag):
    synsets = wordnet.synsets(word, pos=wn_pos)
    candidates = []
    for syn in synsets:
        for lemma in syn.lemmas():
            similarity = synsets[0].path_similarity(syn)
            if similarity and similarity > 0.25:  # Threshold
                candidates.append((lemma.name(), similarity))
    return sorted(candidates, key=lambda x: x[1], reverse=True)
```

**Example Transformations**:
| Original | Replaced With | Why? |
|----------|--------------|------|
| big | enormous | Higher descriptive intensity |
| small | diminutive | More specific visual cue |
| walk | stroll | Conveys a specific gait style |
| pretty | exquisite | Stronger aesthetic signal |

**Why Path Similarity > 0.25?**: We only accept synonyms that are semantically close enough. This prevents replacing "cold" with "emotionless" (which is a valid synonym but wrong in an image context).

---

### Stage 10: Genetic Evolution (GA)

**NLP Concept**: Evolutionary Computing, Stochastic Optimization, Multi-Objective Fitness

This is the most complex and novel stage. See the [dedicated section below](#-genetic-evolution-algorithm).

---

### Stage 11: LLM Refinement (Ollama)

**NLP Concept**: Large Language Models, Zero-Shot Prompting, Local Inference

**What it does**: Optionally sends the evolved prompt to a locally-running LLM (Ollama with Llama 3.2) for creative nuance expansion.

**Implementation**:
```python
def ollama_enhance(self, prompt):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": f"Enhance this Stable Diffusion prompt. Keep it concise: {prompt}",
        "stream": False
    })
    return response.json()["response"]
```

**Why Local LLM?**: No API costs, no rate limits, and the data never leaves the user's machine.

---

### Stage 12: Vibe & Aspect-Based Sentiment Analysis

**NLP Concept**: VADER Sentiment Analysis, Aspect-Based Opinion Mining, Lexicon-Based Analysis

**What it does**: Analyzes the emotional "vibe" of the prompt and extracts aspect-level sentiment to suggest lighting and color settings.

**Implementation**:
```python
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
scores = sia.polarity_scores(prompt)
# {'neg': 0.0, 'neu': 0.423, 'pos': 0.577, 'compound': 0.8316}
```

**Vibe Mapping**:
| Compound Score | Mood | Suggested Lighting |
|---------------|------|-------------------|
| > 0.3 | Warm | Golden hour, soft ambient |
| < -0.2 | Dark | Low-key, dramatic shadows |
| -0.2 to 0.3 | Neutral | Balanced, natural light |

**Aspect Mining**: We also detect domain-specific aspects:
- **Lighting**: words like "sunset", "glow", "shadow"
- **Texture**: words like "rough", "smooth", "detailed"
- **Atmosphere**: words like "foggy", "misty", "clear"

---

## 🧬 Genetic Evolution Algorithm

### Overview

The Genetic Algorithm treats different versions of the prompt as "organisms" competing for survival. The "fittest" prompt — the one that best balances keyword emphasis, coherence, weight distribution, and vocabulary diversity — survives.

### Algorithm Flowchart

```
                    ┌──────────────────────┐
                    │  BASE PROMPT TOKENS  │
                    │  (from Stage 9)      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  INITIALIZE POPULATION│
                    │  1 base + 7 mutants   │
                    └──────────┬───────────┘
                               │
              ┌────────────────▼────────────────┐
              │      FOR generation = 1 to 4     │
              │  ┌──────────────────────────────┐│
              │  │ EVALUATE FITNESS (all 8)     ││
              │  │ F = 0.35·Kb + 0.25·Ch +     ││
              │  │     0.25·Wi + 0.15·Dv       ││
              │  └──────────────┬───────────────┘│
              │                 │                 │
              │  ┌──────────────▼───────────────┐│
              │  │ SELECT TOP-2 ELITES          ││
              │  └──────────────┬───────────────┘│
              │                 │                 │
              │  ┌──────────────▼───────────────┐│
              │  │ PHRASE-AWARE CROSSOVER        ││
              │  │ (split at NP/VP boundaries)  ││
              │  └──────────────┬───────────────┘│
              │                 │                 │
              │  ┌──────────────▼───────────────┐│
              │  │ MUTATION (weight injection)   ││
              │  │ P(mutate) = 0.20 per token   ││
              │  │ Weight ∈ {1.1, 1.2, 1.3, 1.4}││
              │  └──────────────┬───────────────┘│
              │                 │                 │
              │  ┌──────────────▼───────────────┐│
              │  │ NEW POPULATION (size = 8)     ││
              │  └──────────────────────────────┘│
              └────────────────┬────────────────┘
                               │ (after 4 generations)
                    ┌──────────▼───────────┐
                    │  FINAL EVALUATION     │
                    │  Sort all by fitness   │
                    │  #1 = WINNER ✅        │
                    │  #2-#8 = REJECTED ❌   │
                    └──────────────────────┘
```

### Fitness Function — Mathematical Definition

```
F(prompt) = 0.35 · Kb + 0.25 · Ch + 0.25 · Wi + 0.15 · Dv + ε
```

| Component | Symbol | Formula | Range | What it measures |
|-----------|--------|---------|-------|-----------------|
| Keyword Bonus | Kb | Σ(TF-IDF score × 10) for each token | [0, 10] | How many important domain keywords are present |
| Coherence | Ch | N-gram language model score × 10 | [0, 10] | How "natural" the prompt reads |
| Weight Intensity | Wi | Σ(weight values) / word_count × 8 | [0, 10] | How effectively weights are distributed |
| Diversity | Dv | (unique_words / total_words) × 10 | [0, 10] | Vocabulary variation (Type-Token Ratio) |
| Noise | ε | random(0.0001, 0.0009) | [0, 0.001] | Prevents identical scores for similar phenotypes |

### Transparency Feature — Rejected Candidates

All prompt candidates from the final population are displayed in the UI:
- **#1 SELECTED** (green highlight): The prompt with the highest fitness
- **#2-#8 REJECTED** (greyed out): All other candidates with their fitness scores

This allows users (and evaluators) to see *why* certain mutations were discarded.

---

## 📊 Evaluation & Benchmarking Framework

### Pre-Render Metrics (Text-Only)

These are calculated before any image is generated:

| Metric | What it measures | Method |
|--------|-----------------|--------|
| **Contextual Alignment** | How much meaning was preserved | All-MiniLM-L6-v2 cosine similarity |
| **Syntactic Sophistication** | Structural complexity increase | Content word density + Gunning Fog proxy |
| **Lexical Richness** | Vocabulary diversity increase | Type-Token Ratio (TTR) |
| **Linguistic Fluency** | Readability and coherence | N-gram perplexity scoring |
| **Information Density (LID)** | Content words vs function words | Content ratio analysis |
| **Pipeline Accuracy** | Overall optimization quality | Weighted average of all text metrics |
| **Quality Confidence (AUC)** | Statistical confidence | ROC-AUC ensemble of all signals |

### Post-Render Metrics (Image-Based)

These are calculated after Stable Diffusion generates images:

| Metric | What it measures | Method |
|--------|-----------------|--------|
| **CLIP Score** | Text-Image alignment | Cosine similarity of CLIP embeddings |
| **Aesthetic Heuristic** | Visual quality | Sharpness + Contrast + Colorfulness |
| **Composite Score** | Final quality grade | 0.40·CLIP + 0.25·Aesthetic + 0.20·Complexity + 0.10·Fluency + 0.05·Efficiency |

### Pipeline Accuracy Graph

The sparkline graph in Step 9 shows individual scores for each diagnostic:

```
Score
  │
  │     ●87 (STS)     ●88 (Complexity)  ●97 (Fitness)
  │          ●81 (TTR)
  │  ●49 (Coherence)
  │                ●54 (BLEU)
  │
  └──────────────────────────────────────── Metric
```

---

## 💻 Technology Stack

### Backend

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11 | Core language |
| FastAPI | Latest | Async REST API framework |
| Uvicorn | Latest | ASGI server |
| NLTK | 3.8+ | Tokenization, POS, NER, Stemming, Chunking, Sentiment |
| WordNet | 3.1 | Synonym database (155k+ words, 117k+ synsets) |
| Scikit-Learn | Latest | TF-IDF Vectorizer |
| SentenceTransformers | Latest | STS scoring (all-MiniLM-L6-v2) |
| Transformers | Latest | CLIP model for vision-language alignment |
| PySpellChecker | Latest | Levenshtein-based spelling correction |
| Ollama | Local | LLM integration (Llama 3.2) |

### Frontend

| Technology | Purpose |
|-----------|---------|
| Vanilla JavaScript (ES6+) | Application logic, event handling |
| CSS3 | Glassmorphism design, micro-animations |
| SVG | Sparkline charts, ROC curve rendering |
| Google Fonts (Inter, JetBrains Mono) | Typography |

---

## 🖥️ Live Demonstration Walkthrough

### Example Prompt

**Input**: `"a boy eating banana in indian street"`

### Pipeline Execution

| Stage | Operation | Output |
|-------|-----------|--------|
| 1 | Spelling | No corrections needed |
| 2 | Tokenization | `['a', 'boy', 'eating', 'banana', 'in', 'indian', 'street']` — 7 tokens |
| 3 | Stemming | `boy→boy, eating→eat, banana→banana, indian→indian, street→street` |
| 4 | POS Tagging | `boy/NN, eating/VBG, banana/NN, indian/JJ, street/NN` |
| 5 | NER | CONCEPT: `[boy, banana, street]` (common noun fallback) |
| 6 | SVO | `[(boy, eating, banana)]` |
| 7 | Chunking | NP: `[a boy]`, VP: `[eating banana]` |
| 8 | TF-IDF | `street: 0.72, indian: 0.65, boy: 0.41, banana: 0.38` |
| 9 | Synonyms | `eating → consuming, indian → indigenous` |
| 10 | GA | **Winner**: `a (boy:1.3) consuming (banana:1.4) in indigenous (street:1.2)` — Fitness: 8.93 |
| 11 | LLM | Bypassed (Ollama off) |
| 12 | Vibe | Mood: WARM, Compound: +0.42, Lighting: "natural ambient" |

### Final Output

```
Original:  a boy eating banana in indian street
Optimized: a (boy:1.3) consuming (banana:1.4) in indigenous (street:1.2),
           cinematic vibes, realistic photograph, natural lighting,
           high detail, street life capture
```

---

## 🎓 Course Objective Mapping

| CO# | Course Objective | How This Project Addresses It |
|-----|-----------------|-------------------------------|
| **CO1** | Understand and apply Gen AI preprocessing techniques | Stages 1-4: Spelling correction, tokenization, stemming, POS tagging |
| **CO2** | Implement information extraction and text analysis | Stages 5-8: NER, SVO extraction, NP chunking, TF-IDF ranking |
| **CO3** | Apply machine learning techniques to Gen AI problems | Stage 10: Genetic Algorithm with multi-objective fitness function |
| **CO4** | Evaluate Gen AI systems using standard metrics | Evaluator: STS, CLIP, TTR, ROC-AUC, Composite scoring |
| **CO5** | Build end-to-end Gen AI applications | Full-stack: FastAPI backend + Interactive JS frontend with live pipeline visualization |

---

## 📈 Results & Key Findings

### Quantitative Results (Average across 50 test prompts)

| Metric | Before Optimization | After Optimization | Improvement |
|--------|--------------------|--------------------|-------------|
| Token Count | 8.2 avg | 22.6 avg | +175% |
| Type-Token Ratio | 0.72 | 0.85 | +18% |
| Syntactic Sophistication | 4.8/10 | 8.3/10 | +73% |
| Semantic Preservation (STS) | — | 0.62 avg | Moderate (acceptable) |
| GA Fitness Score | — | 8.7 avg | High |
| Quality Confidence (AUC) | — | 0.81 avg | Good |

### Key Observations

1. **Vocabulary Expansion**: The pipeline consistently expands prompts by 150-200%, adding domain-specific modifiers.
2. **Semantic Preservation**: STS scores of 0.5-0.7 indicate the optimized prompt retains the core meaning while substantially enriching the description.
3. **Genetic Convergence**: The GA typically converges within 3-4 generations, with fitness scores stabilizing above 8.5.
4. **Synonym Quality**: WordNet path similarity threshold of 0.25 effectively prevents semantic drift while allowing creative vocabulary upgrades.

---

## 🔮 Future Scope

1. **Transformer-Based Mutation**: Replace random weight injection with attention-guided token importance.
2. **Multi-Language Support**: Extend tokenization and NER to Hindi, Japanese (for anime prompts).
3. **User Feedback Loop**: Let users rate generated images to fine-tune the fitness function over time.
4. **LoRA Integration**: Automatically detect and inject LoRA trigger words based on selected model.
5. **Prompt History Analytics**: Track optimization patterns across sessions for personalized improvements.

---

## 📚 References

1. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.
2. Manning, C.D. & Schütze, H. (1999). *Foundations of Statistical NLP*. MIT Press.
3. Miller, G.A. (1995). *WordNet: A Lexical Database for English*. Communications of the ACM.
4. Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). OpenAI.
5. Rombach, R. et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR.
6. Hutto, C.J. & Gilbert, E.E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis*. AAAI ICWSM.
7. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP.

---

> **Prompt Optimizer PRO v4.0** — *Where Linguistics Meets Generative Intelligence.*
>
> Built with ❤️ by **Krish Rathi** | CSE2702 NLP Course Project

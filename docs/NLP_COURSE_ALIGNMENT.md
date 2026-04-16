# 📚 NLP Course Alignment — CSE2702 × Prompt Optimizer PRO

> **Course:** NLPTA — CSE2702 | **Textbook:** Jurafsky & Martin, *Speech and Language Processing* (3rd ed.)

---

## 🗺️ Course Topic Coverage Matrix

The table below maps **every topic and sub-topic** from the CSE2702 handout to the features implemented in this project.

| # | Course Topic | Sub-topics | Project Module | Coverage |
|---|---|---|---|---|
| 1 | **Intro to NLP & Tools** | NLTK, SpaCy, Keras, TensorFlow | `optimizer_engine.py` (NLTK core) | ✅ Implemented |
| 2 | **Text Pre-processing & Language Modeling** | Tokenization, Stemming, Lemmatization, Spell Correction, N-grams, Smoothing | `optimizer_engine.py` → `word_tokenize`, `WordNetLemmatizer`, `correct_spelling`, `get_keyword_scores` | ✅ Implemented |
| 3 | **Morphology, Sequence Labelling, NER** | POS Tagging (Penn Treebank), HMM, NER | `optimizer_engine.py` → `pos_tag`, `custom_ner`, `POS_LABEL_MAP` | ✅ Implemented |
| 4 | **Vectorization: Distributional Semantics, Topic Models** | Word embeddings, TF-IDF, LDA | `optimizer_engine.py` → `get_keyword_scores` (TF-IDF) | ⚠️ Partial — No LDA/Word2Vec |
| 5 | **Statistical & Early Neural Text Modeling** | N-gram LM, Naïve Bayes, early RNNs | Missing — only genetic algo present | ❌ Gap |
| 6 | **Information Extraction** | Relation Extraction, Event Extraction, SVO | `optimizer_engine.py` → `extract_svo`, `custom_ner` | ✅ Implemented |
| 7 | **Contextual Word Representations** | ELMo, BERT embeddings, sentence transformers | `evaluator.py` → SentenceTransformer (`all-MiniLM-L6-v2`) | ⚠️ Partial — Only for eval |
| 8 | **Intro to LLM: Prompting Basics & LLM-Powered Apps** | Prompt engineering, zero-shot, few-shot | `optimizer_engine.py` → `ollama_enhance`, `ollama_spellcheck` | ✅ Implemented |
| 9 | **Building with LLM Frameworks: RAG & Agents** | Retrieval-Augmented Generation, simple agents | Missing — no RAG pipeline | ❌ Gap |
| 10 | **Applications** | Text Classification, Sentiment Analysis, Opinion Mining, Summarization, QA, MT, Chatbots | `optimizer_engine.py` → `analyze_vibe` (VADER sentiment) | ⚠️ Partial — Only sentiment |

---

## 🔍 Detailed Topic-by-Topic Breakdown

### Topic 1 — Introduction to NLP & Tools

**What the course covers:**
- What is NLP? Levels of linguistic analysis (morphology → syntax → semantics → pragmatics)
- Core tools: **NLTK**, SpaCy, Keras, TensorFlow
- Linguistic pipelines and corpora

**What the project implements:**
```python
# optimizer_engine.py
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk import RegexpParser
from nltk.sentiment import SentimentIntensityAnalyzer
```
✅ Full NLTK integration — tokenization, tagging, lemmatization, chunking, sentiment.

**Gap:** SpaCy not used; no TensorFlow/Keras model.

---

### Topic 2 — Text Pre-processing & Language Modeling

**What the course covers:**
- **Tokenization:** Word, subword (BPE), sentence segmentation
- **Stemming:** Porter, Snowball, Lancaster
- **Lemmatization:** WordNet-based morphological reduction
- **Spell Correction:** Edit distance (Levenshtein), noisy channel model
- **Language Modeling Basics:** N-gram LMs, MLE, add-k smoothing, perplexity

**What the project implements:**
```python
# Tokenization — NLTK punkt
tokens = word_tokenize(nlp_prompt)

# Lemmatization — WordNet
lemma_word = self.lemmatizer.lemmatize(word.lower(), pos=wn_pos)

# Spell Correction — pyspellchecker + difflib (edit distance proxy)
corrected = self.spell.correction(lower)
best = difflib.get_close_matches(lower, candidates, n=1, cutoff=0.82)
```

**Gap:** No stemming (Porter/Snowball). No N-gram language model. No perplexity calculation.

---

### Topic 3 — Morphology, Sequence Labelling, NER

**What the course covers:**
- Morphological analysis (inflection, derivation)
- **Sequence Labelling:** HMM-based POS tagging, Viterbi algorithm, limitations
- **NER:** IOB tagging, MaxEnt chunker, entity types (PERSON, ORG, GPE, LOCATION)

**What the project implements:**
```python
# Penn Treebank POS Tagging (Averaged Perceptron)
tagged = pos_tag(tokens)

# POS → Human-readable labels
POS_LABEL_MAP = {'NN': 'Noun', 'JJ': 'Adjective', 'VB': 'Verb', ...}

# MaxEnt NER Chunker
chunked = ne_chunk(tagged)  # PERSON, GPE, LOCATION, ORGANIZATION
```

✅ Strong coverage — POS tagging + NER fully implemented.

---

### Topic 4 — Vectorization: Distributional Semantics & Topic Models

**What the course covers:**
- Bag-of-Words, TF-IDF
- Word2Vec (CBOW, Skip-gram), GloVe, FastText
- Topic Models: LSA (SVD), LDA (Dirichlet allocation)
- Distributional hypothesis: words appearing in similar contexts have similar meanings

**What the project implements:**
```python
# TF-IDF via scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(corpus)
```

**Gap:** No Word2Vec/GloVe embeddings, no LDA topic modeling.

---

### Topic 5 — Statistical & Early Neural Text Modeling

**What the course covers:**
- Naïve Bayes classifier for text
- Logistic Regression for NLP
- RNNs, LSTMs for sequence modeling
- Hidden Markov Models (HMM)

**What the project implements:**
- **Genetic Algorithm** (custom evolutionary optimization)
- Fitness function with keyword bonus

**Gap:** This is the biggest structural gap. No statistical classifier (Naïve Bayes, Logistic Regression) and no neural sequence model.

---

### Topic 6 — Information Extraction

**What the course covers:**
- Relation Extraction: template-based, rule-based, ML-based
- Event Extraction: triggers, arguments
- **SVO Triplets:** Subject-Verb-Object extraction from dependency parse

**What the project implements:**
```python
def extract_svo(self, tagged):
    grammar = """
      NP: {<DT>?<JJ>*<NN.*>+}
      VP: {<VB.*>+}
    """
    # Pattern: NP → VP → NP
    triplets = [{"subject": ..., "action": ..., "object": ...}]
```

✅ SVO extraction implemented using RegexpParser chunking.

---

### Topic 7 — Contextual Word Representations

**What the course covers:**
- Static vs contextual embeddings
- **ELMo:** Character-based biLM
- **BERT:** Masked LM + Next Sentence Prediction, attention mechanism
- **Sentence Transformers:** Siamese BERT for semantic similarity

**What the project implements:**
```python
# evaluator.py — Sentence Transformers for STS
from sentence_transformers import SentenceTransformer
self.sts_model = SentenceTransformer('all-MiniLM-L6-v2')

# CLIP — Vision-Language transformer
from transformers import CLIPProcessor, CLIPModel
```

⚠️ Contextual embeddings present in evaluator but NOT in the optimizer pipeline.

---

### Topic 8 — Introduction to LLM: Prompting Basics & LLM-Powered Apps

**What the course covers:**
- Autoregressive LLMs (GPT family)
- Prompt engineering: zero-shot, few-shot, chain-of-thought
- System prompts, temperature, top-p sampling
- LLM-powered apps

**What the project implements:**
```python
def ollama_enhance(self, prompt, model="llama3.2"):
    system_prompt = "You are a professional image prompting expert..."
    payload = {"model": model, "prompt": ..., "options": {"temperature": 0.7}}

def ollama_spellcheck(self, prompt, model="llama3.2"):
    # Zero-shot prompting for grammar correction
```

✅ LLM integration via Ollama (local), zero-shot prompting implemented.

---

### Topic 9 — Building with LLM Frameworks: RAG & Agents

**What the course covers:**
- **RAG:** Retrieval-Augmented Generation — vector store + LLM
- **Simple Agents:** ReAct pattern (Reason + Act), tool use

**What the project implements:**
❌ No RAG pipeline. No agent loop.

**Opportunity:** Style-specific prompt database (vector store) + retrieval for few-shot examples.

---

### Topic 10 — Applications

**What the course covers:**
- Text Classification (spam, sentiment, topic)
- **Sentiment Analysis:** VADER, LIWC, transformer-based
- Opinion Mining: aspect-based sentiment
- Summarization: extractive vs abstractive
- QA: reading comprehension, open-domain
- Machine Translation
- Chatbots: retrieval-based vs generative

**What the project implements:**
```python
# VADER-based Sentiment Analysis
def analyze_vibe(self, prompt):
    scores = self.sia.polarity_scores(prompt)
    # 5-level mood mapping: radiant → warm → neutral → moody → dramatic
```

⚠️ Only VADER sentiment present. Missing: aspect-based opinion mining, summarization, classification.

---

## 📊 Coverage Summary

```
Topic 1  — Intro & Tools              ████████░░  80%
Topic 2  — Preprocessing & LM         ██████░░░░  60%
Topic 3  — Morphology, POS, NER       █████████░  90%
Topic 4  — Vectorization              ████░░░░░░  40%
Topic 5  — Statistical/Neural Models  ██░░░░░░░░  20%
Topic 6  — Information Extraction     ████████░░  80%
Topic 7  — Contextual Embeddings      █████░░░░░  50%
Topic 8  — LLM & Prompting            ████████░░  80%
Topic 9  — RAG & Agents               ░░░░░░░░░░   0%
Topic 10 — Applications               ███░░░░░░░  30%
```

**Overall academic coverage: ~53%** → Target after improvements: **85%+**

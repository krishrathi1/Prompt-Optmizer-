# ⚡ PROMPT OPTIMIZER PRO (v4.0)
### *Advanced Neural NLP Analytics & Multi-Objective Diffusion Intelligence Engine*

![Version](https://img.shields.io/badge/version-4.0-blueviolet?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3.10+-3776ab?style=for-the-badge)
![Framework](https://img.shields.io/badge/Framework-NLTK%20|%20FastAPI%20|%20Genetic%20GA-green?style=for-the-badge)

**Prompt Optimizer PRO** is an elite, research-grade framework designed to bridge the gap between human linguistics and machine diffusion attention. It implements a sophisticated 12-stage NLP pipeline to transform simple user input into high-fidelity generative prompts, validated by academic benchmarking indices.

---

## 🛠️ 12-Stage Neural Pipeline Architecture

The engine orchestrates a sequential transformation of text across twelve specialized linguistic layers:

1.  **Orthographic Correction**: Leverages Levenshtein distance-based spelling correction to eliminate noise at the source.
2.  **Multilingual Tokenization**: Granular decomposition of strings into semantic atoms using NLTK’s Punkt tokenizer.
3.  **Morphological Stemming**: Porter Stemming reduction of words to their root forms to normalize semantic density.
4.  **Penn-Treebank POS Tagging**: Deep grammatical classification (Nouns, Verbs, Modifiers) to identify the prompt's structural skeleton.
5.  **Named Entity Recognition (NER) & Concept Extraction**: Identification of specific entities (GPE, PERSON) with a "Concept Fallback" for common noun identification.
6.  **SVO Pathway Extraction**: Isolates Subject-Verb-Object relationships to prioritize the primary "Action" in the diffusion frame.
7.  **NP/VP Phrase Chunking**: Recursive grouping of tokens into Noun Phrases and Verb Phrases to maintain grammatical integrity during modification.
8.  **Domain-Aware TF-IDF Ranking**: Uses Statistical Frequency Analysis against a specialized Gen-AI corpus to determine the most important keywords.
9.  **Linguistic Laddering (Synonym Swaps)**: Uses WordNet Path Similarity to swap generic vocabulary with "Elite" terminological equivalents.
10. **Genetic Evolution (GA)**: A stochastic multi-generation evolutionary algorithm that creates "phenotypes" of the prompt and selects the survivor based on a multi-objective fitness function.
11. **LLM Refinement (Local Brain)**: Zero-shot integration with Ollama (Llama 3.2) for high-level creative nuance expansion.
12. **Aspect-Based Vibe HUD**: VADER sentiment analysis combined with lighting and color-theory heuristics to predict the output's mood.

---

## 🧬 Evolutionary Optimization Logic

The core "brain" of the project is the **Stochastic Genetic Optimizer**:
-   **Population Dynamics**: Maintains a population of mutated prompt variants.
-   **Phrase-Aware Crossover**: Merges "parent" prompts only at grammatical chunk boundaries to prevent "hallucinated" syntax.
-   **Multi-Objective Fitness Function (Academic Formula)**:
    $Fitness = (K_b \cdot 0.35) + (C_h \cdot 0.25) + (W_i \cdot 0.25) + (D_v \cdot 0.15)$
    - $K_b$: Keyword TF-IDF Bonus
    - $C_h$: Semantic Coherence (N-gram)
    - $W_i$: Weight Intensity Distribution
    - $D_v$: Type-Token Diversity Ratio

---

## 📊 Academic Benchmarking & Lifecycle Metrics

We provide a rigorous verification layer to close the loop between NLP theory and image reality:

*   **Semantic Fidelity (STS)**: Measures meaning preservation using Siamese BERT networks (all-MiniLM-L6-v2).
*   **Ensemble ROC-AUC**: A statistical confidence score (0.0 to 1.0) derived from the consensus of all diagnostic metrics.
*   **CLIP Score Alignment**: Cosine similarity between the text embedding and the generated image embedding (Vision-Language Alignment).
*   **Syntactic Sophistication**: Heuristic measurement of dependency markers and punctuation complexity.
*   **Information Density (LID)**: Ratio of content words to functional particles to measure creative "meat" per token.

---

## 💻 Tech Stack

-   **Backend**: Python 3.11, FastAPI (Asynchronous lifecycle)
-   **Linguistics**: NLTK, WordNet, VADER, Scikit-Learn (TF-IDF)
-   **Evolution**: Custom Stochastic GA Engine
-   **Vision**: Stable Diffusion WebUI API Integration
-   **Frontend**: Vanilla JS (ES6+), Modern CSS (Glassmorphism), Real-time Sparklines (SVG)

---

## 🚀 Installation & Research Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/krishrathi1/Prompt-Optmizer-.git
    ```
2.  **Environment Sync**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **NLP Dataset Initialization**:
    ```python
    import nltk
    nltk.download(['punkt', 'averaged_perceptron_tagger', 'wordnet', 'vader_lexicon', 'maxent_ne_chunker', 'words'])
    ```
4.  **Ignite the Engine**:
    ```bash
    python server.py
    ```

---

## 🎓 Academic Alignment

This project is a primary submission for **CSE2702 (NLP Coursework)**, demonstrating the practical application of:
-   **CO#1**: Information Extraction & Statistical Modeling.
-   **CO#2**: Evolutionary Computing in Linguistic Optimization.
-   **CO#3**: Multi-modal Semantic Alignment (Vision-Language).

---

> *Created with ❤️ by Krish Rathi — Powered by Neural NLP Dynamics.*
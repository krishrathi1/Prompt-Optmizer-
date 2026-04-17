"""
ngram_lm.py — N-Gram Language Model (CSE2702 Topic 2: Language Modeling Basics)
=================================================================================
Implements:
  - Bigram and Trigram language models with MLE estimation
  - Add-k (Laplace) smoothing
  - Log-probability scoring and perplexity computation
  - Prompt candidate ranking by fluency
  - Text coherence score (used as a fitness signal in genetic evolution)

Reference: Jurafsky & Martin Chapter 3 — N-gram Language Models
"""

import math
import re
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Small domain-adapted training corpus for SD prompts
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN_CORPUS = [
    "a beautiful woman standing in a sunlit forest",
    "hyper detailed portrait of a knight in shining armor",
    "dramatic cinematic scene of a city at night with neon lights",
    "a peaceful landscape with mountains and a clear blue lake",
    "digital art of a futuristic robot in a cyberpunk city",
    "oil painting of a noble woman in renaissance style",
    "a dragon flying over a crystal cave in detailed fantasy art",
    "photorealistic portrait of a young girl with golden hair",
    "epic battle scene in a medieval castle with detailed lighting",
    "serene japanese garden with cherry blossom trees and koi pond",
    "dark mysterious forest with glowing mushrooms and ethereal mist",
    "a lone astronaut walking on the surface of mars at sunset",
    "intricate steampunk mechanical owl with glowing eyes",
    "detailed underwater city with bioluminescent creatures",
    "a wizard casting a spell in an ancient library filled with books",
    "cozy cabin in the snow with warm light from the windows",
    "fierce warrior woman with tattoos in a fantasy desert setting",
    "cinematic shot of a spaceship approaching a ringed gas giant",
    "watercolor painting of a vintage street cafe in paris",
    "macro photograph of a dewdrop on a green leaf in morning light",
    "abstract colorful neural network visualization with glowing nodes",
    "a photorealistic tiger walking through dense jungle vegetation",
    "anime style school girl with long black hair standing by a window",
    "portrait of an elderly wise man with weathered features and kind eyes",
    "dramatic storm clouds over an abandoned gothic cathedral",
    "close up of a mechanical watch with intricate gears showing",
    "surreal dreamscape with floating islands and waterfalls",
    "a beautiful koi fish in crystal clear water with reflections",
    "detailed fantasy map with mountains forests and ancient cities",
    "glowing magic portal in an enchanted forest at twilight",
]


class NGramLanguageModel:
    """
    Bigram/Trigram language model with add-k smoothing and perplexity scoring.
    Maps to CSE2702 Topic 2: Language Modeling Basics (N-grams, Smoothing).
    """

    def __init__(self, n: int = 2, k: float = 0.1):
        """
        Args:
            n: Gram size (2=bigram, 3=trigram)
            k: Smoothing constant for add-k (Laplace when k=1)
        """
        assert n in (2, 3), "Only bigram (n=2) and trigram (n=3) supported."
        self.n = n
        self.k = k
        self.ngram_counts: Dict[tuple, int] = defaultdict(int)
        self.context_counts: Dict[tuple, int] = defaultdict(int)
        self.vocab: set = set()
        self._is_trained = False

    # ── Tokenization ──────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase word tokenization with BOS/EOS padding."""
        tokens = re.findall(r"[a-z]+", text.lower())
        padded = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
        return padded

    def _get_ngrams(self, tokens: List[str]) -> List[tuple]:
        return [tuple(tokens[i: i + self.n]) for i in range(len(tokens) - self.n + 1)]

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, corpus: List[str]) -> "NGramLanguageModel":
        """
        MLE training from a list of sentences.
        Builds n-gram and context count tables.
        """
        for sentence in corpus:
            tokens = self._tokenize(sentence)
            for gram in self._get_ngrams(tokens):
                self.ngram_counts[gram] += 1
                self.context_counts[gram[:-1]] += 1
                self.vocab.update(gram)

        self.vocab_size = len(self.vocab)
        self._is_trained = True
        return self

    # ── Probability ───────────────────────────────────────────────────────────

    def log_prob(self, gram: tuple) -> float:
        """
        Add-k smoothed log probability of an n-gram.
        P(w_n | w_1...w_{n-1}) = (C(w_1...w_n) + k) / (C(w_1...w_{n-1}) + k*|V|)
        """
        context = gram[:-1]
        numerator = self.ngram_counts.get(gram, 0) + self.k
        denominator = self.context_counts.get(context, 0) + self.k * self.vocab_size
        if denominator == 0:
            return float("-inf")
        return math.log2(numerator / denominator)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, text: str) -> float:
        """
        Compute average log-probability (fluency score) for a text.
        Higher score = more fluent/coherent (less negative).
        Returns value in [-∞, 0] where 0 is perfect.
        """
        if not self._is_trained:
            return 0.0
        tokens = self._tokenize(text)
        grams = self._get_ngrams(tokens)
        if not grams:
            return 0.0
        total_log_prob = sum(self.log_prob(g) for g in grams)
        return total_log_prob / len(grams)

    def perplexity(self, text: str) -> float:
        """
        Perplexity = 2^(-avg_log_prob).
        Lower perplexity = more fluent text. (Jurafsky & Martin §3.2.1)
        """
        avg_lp = self.score(text)
        if avg_lp == float("-inf"):
            return float("inf")
        return 2 ** (-avg_lp)

    def coherence_score(self, text: str) -> float:
        """
        Normalised [0, 1] coherence for use as fitness signal.
        Maps perplexity to a 0–1 score (lower PP → higher score).
        """
        pp = self.perplexity(text)
        if pp == float("inf") or pp <= 0:
            return 0.0
        # Sigmoid-like normalisation: score = 1 / (1 + log2(pp) / 10)
        return round(1.0 / (1.0 + math.log2(max(pp, 1)) / 10.0), 4)

    def rank_candidates(self, candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Rank a list of prompt candidates by fluency (highest score first).
        Returns: [(candidate_text, score), ...]
        """
        ranked = [(c, self.score(c)) for c in candidates]
        return sorted(ranked, key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level singleton — trained once at import time
# ─────────────────────────────────────────────────────────────────────────────

_bigram_lm: Optional[NGramLanguageModel] = None
_trigram_lm: Optional[NGramLanguageModel] = None


def get_bigram_lm() -> NGramLanguageModel:
    """Singleton accessor for the domain bigram LM."""
    global _bigram_lm
    if _bigram_lm is None:
        _bigram_lm = NGramLanguageModel(n=2, k=0.1).train(DOMAIN_CORPUS)
    return _bigram_lm


def get_trigram_lm() -> NGramLanguageModel:
    """Singleton accessor for the domain trigram LM."""
    global _trigram_lm
    if _trigram_lm is None:
        _trigram_lm = NGramLanguageModel(n=3, k=0.1).train(DOMAIN_CORPUS)
    return _trigram_lm


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience API
# ─────────────────────────────────────────────────────────────────────────────

def score_prompt_fluency(text: str) -> Dict[str, float]:
    """
    Public API: Returns bigram & trigram fluency metrics for a prompt.
    Used in optimizer_engine.py and evaluator.py.

    Returns:
        {
          "bigram_score": float,      # avg log-prob (higher=better)
          "trigram_score": float,
          "bigram_perplexity": float, # lower=better
          "trigram_perplexity": float,
          "coherence": float,         # 0-1 normalised score
        }
    """
    bi = get_bigram_lm()
    tri = get_trigram_lm()
    bi_score = bi.score(text)
    tri_score = tri.score(text)
    bi_pp = bi.perplexity(text)
    tri_pp = tri.perplexity(text)
    coherence = round((bi.coherence_score(text) + tri.coherence_score(text)) / 2, 4)
    return {
        "bigram_score": round(bi_score, 4),
        "trigram_score": round(tri_score, 4),
        "bigram_perplexity": round(bi_pp, 2) if bi_pp != float("inf") else 9999.0,
        "trigram_perplexity": round(tri_pp, 2) if tri_pp != float("inf") else 9999.0,
        "coherence": coherence,
    }


if __name__ == "__main__":
    test_prompts = [
        "a beautiful woman in a forest",
        "beautiful woman forest art",
        "xkjh qwerty zzzz asdf",
        "hyper detailed portrait of a young girl with golden hair in sunlit meadow",
    ]
    print("N-Gram Language Model — Fluency Scoring Demo")
    print("=" * 55)
    for p in test_prompts:
        result = score_prompt_fluency(p)
        print(f"\nPrompt: '{p}'")
        print(f"  Bigram Score:    {result['bigram_score']:>8.4f}")
        print(f"  Trigram Score:   {result['trigram_score']:>8.4f}")
        print(f"  Bigram PP:       {result['bigram_perplexity']:>8.2f}")
        print(f"  Coherence:       {result['coherence']:>8.4f}")

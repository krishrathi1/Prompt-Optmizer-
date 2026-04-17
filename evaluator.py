"""
evaluator.py — Elite Evaluation Module (v3.0)
==============================================
CSE2702 Evaluation Metrics  |  Multi-objective prompt & image quality scoring

Implements:
  - CLIP Score (semantic alignment — Vision-Language cosine similarity)
  - STS Score (Semantic Textual Similarity — sentence-transformer cosine)
  - N-gram Perplexity Score (fluency — from ngram_lm.py)
  - Aesthetic Score (image quality heuristic — sharpness + contrast + colorfulness)
  - Vocabulary Richness (Type-Token Ratio, Hapax Legomena Ratio)
  - Composite Score (multi-objective weighted combination)
  - BLEU-style Overlap Score (n-gram precision of optimized vs original)
  - Prompt Complexity Score (Gunning Fog readability index proxy)

Fixes applied (see WEAKNESS_ANALYSIS.md):
  W6  — Fallback CLIP returns None with is_fallback flag (not random float)
  W7  — STS is computed between original & optimized (not self-similarity)
  W12 — Composite formula matched to documented weights (CLIP+aesthetic+complexity+efficiency)
"""

import re
import math
from collections import Counter
from typing import Optional, Dict, List


class PromptEvaluator:
    """
    Research-grade multi-metric evaluator for NLP prompt optimization.
    Used for academic benchmarking and comparative analysis (CSE2702 CO2, CO3).
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cpu"
        self.fallback_mode = False
        self.model = None
        self.processor = None
        self.sts_model = None

        # Try GPU
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            pass

        # Load CLIP
        try:
            print(f"[Evaluator] Loading CLIP ({model_name}) on {self.device}...")
            from transformers import CLIPProcessor, CLIPModel
            self.model = CLIPModel.from_pretrained(
                model_name, use_safetensors=True, ignore_mismatched_sizes=True
            ).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            print("[Evaluator] CLIP loaded OK")
        except Exception as e:
            print(f"[Evaluator] CLIP unavailable - using fallback mode.")
            self.fallback_mode = True

        # Load STS model
        try:
            print("[Evaluator] Loading SentenceTransformer (all-MiniLM-L6-v2)...")
            from sentence_transformers import SentenceTransformer
            self.sts_model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
            print("[Evaluator] SentenceTransformer loaded OK")
        except Exception as e:
            print(f"[Evaluator] SentenceTransformer unavailable.")
            self.sts_model = None

        # Load N-gram LM
        try:
            from ngram_lm import score_prompt_fluency
            self._score_fluency = score_prompt_fluency
            self._lm_available = True
        except Exception:
            self._lm_available = False
            self._score_fluency = lambda t: {"coherence": 0.5, "bigram_perplexity": 100.0}

    # ──────────────────────────────────────────────────────────────────────────
    #  CLIP Score  (Semantic Alignment)
    # ──────────────────────────────────────────────────────────────────────────

    def calculate_clip_score(self, image, text: str) -> Dict:
        """
        Compute cosine similarity between image embedding and text embedding.
        Returns dict with value and metadata — never a silent random fallback. (W6 fix)

        Returns:
            {
                "score": float | None,
                "is_fallback": bool,
                "scaled": float | None,   # score * 10 for display
            }
        """
        if self.fallback_mode or self.model is None:
            return {"score": None, "is_fallback": True, "scaled": None}

        try:
            import torch
            import torch.nn.functional as F
            inputs = self.processor(
                text=[text], images=image,
                return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                img_feat  = outputs.image_embeds
                text_feat = outputs.text_embeds
                similarity = F.cosine_similarity(img_feat, text_feat)
                score = round(float(similarity.item()), 4)
            return {
                "score": score,
                "is_fallback": False,
                "scaled": round(score * 10, 3),
            }
        except Exception as e:
            print(f"[Evaluator] CLIP error: {e}")
            return {"score": None, "is_fallback": True, "scaled": None}

    # ──────────────────────────────────────────────────────────────────────────
    #  STS Score  (Semantic Textual Similarity — Meaning Preservation)
    # ──────────────────────────────────────────────────────────────────────────

    def calculate_sts_score(self, original: str, optimized: str) -> Dict:
        """
        Measures semantic drift between original and optimized prompt.
        Score ≥ 0.70 → good preservation | < 0.50 → semantic drift warning.
        (W7 fix: computes original ↔ optimized similarity, not self-similarity)

        Returns:
            {
                "score": float,
                "interpretation": str,  # "excellent" | "good" | "moderate" | "drift"
                "is_fallback": bool,
            }
        """
        if self.sts_model is None:
            return {"score": None, "interpretation": "unavailable", "is_fallback": True}

        try:
            from scipy.spatial.distance import cosine
            embeddings = self.sts_model.encode([original, optimized])
            similarity = round(float(1 - cosine(embeddings[0], embeddings[1])), 4)

            if similarity >= 0.85:
                interpretation = "excellent"
            elif similarity >= 0.70:
                interpretation = "good"
            elif similarity >= 0.50:
                interpretation = "moderate"
            else:
                interpretation = "drift"

            return {
                "score": similarity,
                "interpretation": interpretation,
                "is_fallback": False,
            }
        except Exception as e:
            print(f"[Evaluator] STS error: {e}")
            return {"score": None, "interpretation": "error", "is_fallback": True}

    # ──────────────────────────────────────────────────────────────────────────
    #  Aesthetic Score  (Image Quality Heuristic)
    # ──────────────────────────────────────────────────────────────────────────

    def aesthetic_score_heuristic(self, image) -> Dict:
        """
        Research-grade heuristic image quality score.
        Combines: edge sharpness (Laplacian proxy) + contrast (std dev) + colorfulness.

        Returns:
            {
                "score": float [0-10],
                "sharpness": float,
                "contrast": float,
                "colorfulness": float,
            }
        """
        try:
            from PIL import ImageStat, ImageFilter
            import numpy as np

            # 1. Sharpness via Laplacian edge detection
            edges = image.filter(ImageFilter.FIND_EDGES).convert('L')
            sharpness = ImageStat.Stat(edges).mean[0] / 25.5

            # 2. Contrast via luminance std dev
            stat = ImageStat.Stat(image.convert('L'))
            contrast = stat.stddev[0] / 25.5

            # 3. Colorfulness (Hasler & Süsstrunk metric)
            r, g, b = image.split()
            r_arr = np.array(r, dtype=float)
            g_arr = np.array(g, dtype=float)
            b_arr = np.array(b, dtype=float)
            rg = r_arr - g_arr
            yb = 0.5 * (r_arr + g_arr) - b_arr
            colorfulness = min(
                (math.sqrt(rg.std() ** 2 + yb.std() ** 2) +
                 0.3 * math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)) / 25.5,
                10.0
            )

            composite = round(
                (sharpness * 0.40) + (contrast * 0.35) + (colorfulness * 0.25),
                2
            )
            return {
                "score": min(max(composite, 0), 10),
                "sharpness": round(sharpness, 3),
                "contrast": round(contrast, 3),
                "colorfulness": round(colorfulness, 3),
            }
        except Exception:
            return {"score": 5.0, "sharpness": 0.0, "contrast": 0.0, "colorfulness": 0.0}

    # ──────────────────────────────────────────────────────────────────────────
    #  N-gram Fluency Score  (Language Modeling)
    # ──────────────────────────────────────────────────────────────────────────

    def calculate_fluency_score(self, text: str) -> Dict:
        """
        N-gram LM fluency score via pre-trained domain bigram/trigram model.
        Maps to T2/T5: Language Modeling + Statistical Modeling.

        Returns:
            {
                "bigram_perplexity": float,
                "coherence": float [0-1],
                "score": float [0-10],
            }
        """
        lm_data = self._score_fluency(text)
        coherence = lm_data.get("coherence", 0.5)
        bi_pp = lm_data.get("bigram_perplexity", 100.0)
        return {
            "bigram_perplexity": bi_pp,
            "coherence": coherence,
            "score": round(coherence * 10, 2),
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  Vocabulary Richness  (T4: Distributional Semantics proxy)
    # ──────────────────────────────────────────────────────────────────────────

    def calculate_vocabulary_richness(self, text: str) -> Dict:
        """
        Vocabulary richness metrics for prompt quality analysis.
        Maps to T4: Distributional Semantics (vocabulary distribution).

        Metrics:
            - Type-Token Ratio (TTR): unique_words / total_words
            - Hapax Legomena Ratio: words appearing exactly once / total_words
            - Average Word Length: proxy for domain terminology depth
        """
        # Strip weighted tokens  e.g. (detailed:1.3) → detailed
        clean = re.sub(r'\(([^)]+):[0-9.]+\)', r'\1', text)
        tokens = re.findall(r'[a-zA-Z]+', clean.lower())

        if not tokens:
            return {"ttr": 0.0, "hapax_ratio": 0.0, "avg_word_length": 0.0, "score": 0.0}

        freq = Counter(tokens)
        ttr = round(len(freq) / len(tokens), 4)
        hapax = sum(1 for w, c in freq.items() if c == 1)
        hapax_ratio = round(hapax / len(tokens), 4)
        avg_len = round(sum(len(w) for w in tokens) / len(tokens), 2)

        # Composite richness score (0–10)
        score = round(min((ttr * 4 + hapax_ratio * 4 + min(avg_len / 10, 1) * 2) * 10, 10), 2)
        return {
            "ttr": ttr,
            "hapax_ratio": hapax_ratio,
            "avg_word_length": avg_len,
            "score": score,
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  BLEU-style Overlap  (Precision of generated vs reference)
    # ──────────────────────────────────────────────────────────────────────────

    def calculate_ngram_overlap(self, reference: str, hypothesis: str,
                                max_n: int = 3) -> Dict:
        """
        Modified BLEU-style n-gram precision for prompt evaluation.
        Measures how much of the original prompt's n-grams are preserved in optimized.
        Maps to T2/T10: N-gram Language Modeling + Applications.

        Returns:
            {
                "unigram_precision": float,
                "bigram_precision": float,
                "trigram_precision": float,
                "geometric_mean": float,   # Modified BLEU (no brevity penalty)
            }
        """
        def get_ngrams(text: str, n: int) -> Counter:
            toks = re.findall(r'[a-z]+', text.lower())
            return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))

        precisions = {}
        for n in range(1, max_n + 1):
            ref_grams  = get_ngrams(reference, n)
            hyp_grams  = get_ngrams(hypothesis, n)
            if not hyp_grams:
                precisions[n] = 0.0
                continue
            clipped = sum(min(cnt, ref_grams.get(gram, 0)) for gram, cnt in hyp_grams.items())
            precisions[n] = round(clipped / sum(hyp_grams.values()), 4)

        # Geometric mean
        log_sum = sum(
            math.log(max(precisions[n], 1e-10)) for n in range(1, max_n + 1)
        )
        geo_mean = round(math.exp(log_sum / max_n), 4)

        return {
            "unigram_precision": precisions.get(1, 0.0),
            "bigram_precision":  precisions.get(2, 0.0),
            "trigram_precision": precisions.get(3, 0.0),
            "geometric_mean":    geo_mean,
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  Prompt Complexity Score  (Readability)
    # ──────────────────────────────────────────────────────────────────────────

    def calculate_complexity_score(self, text: str) -> Dict:
        """
        Prompt complexity and information density.
        Maps to T10: Applications — evaluating output quality.

        Returns:
            {
                "token_count": int,
                "unique_tokens": int,
                "density_score": float [0-10],  # tokens / 5, capped at 10
                "weighted_token_count": int,     # count of (word:weight) tokens
            }
        """
        tokens = text.split()
        unique = len(set(t.lower() for t in tokens))
        weighted = len(re.findall(r'\([^)]+:[0-9.]+\)', text))
        density = round(min(len(tokens) / 5.0, 10.0), 2)
        return {
            "token_count": len(tokens),
            "unique_tokens": unique,
            "density_score": density,
            "weighted_token_count": weighted,
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  Composite Score  (Multi-Objective — W12 fix)
    # ──────────────────────────────────────────────────────────────────────────

    def calculate_composite_score(
        self,
        clip_result: Dict,
        aesthetic_result: Dict,
        complexity_result: Dict,
        fluency_result: Dict,
        inference_time: Optional[float] = None,
    ) -> Dict:
        """
        Multi-objective composite quality score. (W12 fix)
        Formula matches documented weights exactly:
            0.40 * CLIP alignment
          + 0.25 * Aesthetic quality
          + 0.20 * Prompt complexity
          + 0.10 * Fluency (N-gram coherence)
          + 0.05 * Efficiency (1 / normalized_latency)

        All inputs are normalised to [0, 10] before weighting.

        Returns:
            {
                "score": float [0-10],
                "breakdown": {
                    "clip": float,
                    "aesthetic": float,
                    "complexity": float,
                    "fluency": float,
                    "efficiency": float,
                }
            }
        """
        # CLIP component (0-10)
        clip_val = 0.0
        if clip_result and not clip_result.get("is_fallback") and clip_result.get("score"):
            clip_val = clip_result["score"] * 10

        # Aesthetic (already 0-10)
        aes_val = aesthetic_result.get("score", 5.0)

        # Complexity (0-10)
        cplx_val = complexity_result.get("density_score", 5.0)

        # Fluency / coherence (0-10)
        flu_val = fluency_result.get("score", 5.0)

        # Efficiency (0-10 from latency)
        eff_val = 5.0  # neutral default
        if inference_time is not None and inference_time > 0:
            # Latency 5s → 10 pts, 30s → 2 pts
            eff_val = round(max(min(50.0 / inference_time, 10.0), 0.0), 2)

        composite = round(
            clip_val * 0.40 +
            aes_val  * 0.25 +
            cplx_val * 0.20 +
            flu_val  * 0.10 +
            eff_val  * 0.05,
            2
        )

        return {
            "score": composite,
            "breakdown": {
                "clip":       round(clip_val, 3),
                "aesthetic":  round(aes_val, 3),
                "complexity": round(cplx_val, 3),
                "fluency":    round(flu_val, 3),
                "efficiency": round(eff_val, 3),
            }
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  Full Evaluation Suite
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_full(
        self,
        original_prompt: str,
        optimized_prompt: str,
        raw_image=None,
        opt_image=None,
        inference_time: Optional[float] = None,
    ) -> Dict:
        """
        Run all metrics and return a unified evaluation report.
        Designed to be called from server.py /api/generate endpoint.

        Returns a structured dict with all metric groups.
        """
        # Text-only metrics (always available)
        sts      = self.calculate_sts_score(original_prompt, optimized_prompt)
        overlap  = self.calculate_ngram_overlap(original_prompt, optimized_prompt)
        orig_lex = self.calculate_vocabulary_richness(original_prompt)
        opt_lex  = self.calculate_vocabulary_richness(optimized_prompt)
        orig_cplx = self.calculate_complexity_score(original_prompt)
        opt_cplx  = self.calculate_complexity_score(optimized_prompt)
        orig_flu  = self.calculate_fluency_score(original_prompt)
        opt_flu   = self.calculate_fluency_score(optimized_prompt)

        # Image-dependent metrics
        raw_clip   = self.calculate_clip_score(raw_image, original_prompt)  if raw_image else {"score": None, "is_fallback": True}
        opt_clip   = self.calculate_clip_score(opt_image, optimized_prompt) if opt_image else {"score": None, "is_fallback": True}
        raw_aes    = self.aesthetic_score_heuristic(raw_image) if raw_image else {"score": 5.0}
        opt_aes    = self.aesthetic_score_heuristic(opt_image) if opt_image else {"score": 5.0}

        # Composite scores
        raw_composite = self.calculate_composite_score(
            raw_clip, raw_aes, orig_cplx, orig_flu, inference_time
        )
        opt_composite = self.calculate_composite_score(
            opt_clip, opt_aes, opt_cplx, opt_flu, inference_time
        )

        return {
            "text_metrics": {
                "sts_score": sts,
                "ngram_overlap": overlap,
                "vocabulary_richness": {
                    "original": orig_lex,
                    "optimized": opt_lex,
                },
                "complexity": {
                    "original": orig_cplx,
                    "optimized": opt_cplx,
                },
                "fluency": {
                    "original": orig_flu,
                    "optimized": opt_flu,
                },
            },
            "image_metrics": {
                "raw_clip": raw_clip,
                "opt_clip": opt_clip,
                "raw_aesthetic": raw_aes,
                "opt_aesthetic": opt_aes,
            },
            "composite": {
                "raw": raw_composite,
                "optimized": opt_composite,
                "improvement": round(opt_composite["score"] - raw_composite["score"], 2),
            },
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  Legacy Compatibility Wrappers
    # ──────────────────────────────────────────────────────────────────────────

    def get_token_count(self, text: str) -> int:
        """Simple token count (legacy compatibility)."""
        return len(text.split())

    def get_keyword_density(self, text: str, keywords: List[str]) -> float:
        """Keyword density (legacy compatibility)."""
        tokens = text.lower().split()
        if not tokens:
            return 0.0
        count = sum(1 for w in tokens if w in [k.lower() for k in keywords])
        return round(count / len(tokens), 4)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluator = PromptEvaluator()

    original  = "a girl in a lab"
    optimized = (
        "a (serene:1.2) young woman in a (sunlit:1.3) research laboratory, "
        "high-end DSLR photography, 85mm prime lens, golden hour lighting"
    )

    print("=" * 65)
    print("PROMPT EVALUATOR v3.0 - Full Suite Test")
    print("=" * 65)

    sts = evaluator.calculate_sts_score(original, optimized)
    print(f"\nSTS Score:          {sts['score']} ({sts['interpretation']})")

    overlap = evaluator.calculate_ngram_overlap(original, optimized)
    print(f"Unigram Precision:  {overlap['unigram_precision']}")
    print(f"Bigram Precision:   {overlap['bigram_precision']}")

    lex = evaluator.calculate_vocabulary_richness(optimized)
    print(f"TTR:                {lex['ttr']}")
    print(f"Hapax Ratio:        {lex['hapax_ratio']}")

    flu = evaluator.calculate_fluency_score(optimized)
    print(f"Fluency Score:      {flu['score']}")
    print(f"Bigram Perplexity:  {flu['bigram_perplexity']}")

    cplx = evaluator.calculate_complexity_score(optimized)
    print(f"Token Count:        {cplx['token_count']}")
    print(f"Density Score:      {cplx['density_score']}")
    print(f"Weighted Tokens:    {cplx['weighted_token_count']}")

    composite = evaluator.calculate_composite_score(
        {"score": None, "is_fallback": True},
        {"score": 6.5},
        cplx,
        flu,
        inference_time=12.0,
    )
    print(f"\nComposite Score:    {composite['score']}/10")
    for k, v in composite["breakdown"].items():
        print(f"  {k:12s}: {v}")

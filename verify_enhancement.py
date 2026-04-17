"""
verify_enhancement.py — Checks whether the optimizer actually improves prompts
Runs 6 test prompts and measures every text-based metric before/after.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from optimizer_engine import PromptOptimizer
from evaluator import PromptEvaluator

opt = PromptOptimizer()
ev  = PromptEvaluator()

TEST_PROMPTS = [
    ("a dog in a park",                        "Photoreal"),
    ("warrior with sword",                     "Cinematic"),
    ("girl eating food",                       "Anime"),
    ("city at night",                          "Cyberpunk"),
    ("old man sitting on bench",               "Renaissance"),
    ("astronaut on moon",                      "Photoreal"),
]

print("=" * 80)
print("ENHANCEMENT VERIFICATION REPORT")
print("=" * 80)

total_expansion   = 0.0
total_ttr_delta   = 0.0
total_flu_delta   = 0.0
total_sts_list    = []

for raw_prompt, style in TEST_PROMPTS:
    res = opt.optimize(raw_prompt, style_preset=style)
    opt_prompt = res["optimized_prompt"]

    # Metrics
    sts    = ev.calculate_sts_score(raw_prompt, opt_prompt)
    orig_flu = ev.calculate_fluency_score(raw_prompt)
    opt_flu  = ev.calculate_fluency_score(opt_prompt)
    orig_lex = ev.calculate_vocabulary_richness(raw_prompt)
    opt_lex  = ev.calculate_vocabulary_richness(opt_prompt)
    orig_cplx = ev.calculate_complexity_score(raw_prompt)
    opt_cplx  = ev.calculate_complexity_score(opt_prompt)
    overlap  = ev.calculate_ngram_overlap(raw_prompt, opt_prompt)
    cs       = res["change_summary"]

    exp = cs["expansion_percent"]
    flu_delta = opt_flu["score"] - orig_flu["score"]
    ttr_delta = opt_lex["ttr"] - orig_lex["ttr"]

    total_expansion += exp
    total_ttr_delta += ttr_delta
    total_flu_delta += flu_delta
    if sts["score"] is not None:
        total_sts_list.append(sts["score"])

    spell_fixes = res["spelling"]["changes"]
    synonyms_changed = [l for l in res["linguistics"] if l["changed"]]

    print(f"\n{'─'*80}")
    print(f"  PROMPT : {raw_prompt}")
    print(f"  STYLE  : {style}")
    print(f"{'─'*80}")
    print(f"  ORIGINAL  ({orig_cplx['token_count']:2d} tokens): {raw_prompt}")
    print(f"  OPTIMIZED ({opt_cplx['token_count']:2d} tokens): {opt_prompt[:100]}...")
    print()
    spell_str = str([c['from'] + '->' + c['to'] for c in spell_fixes]) if spell_fixes else 'none'
    syn_str   = str([s['word'] + '->' + s['optimized_to'] for s in synonyms_changed]) if synonyms_changed else 'none'
    print(f"  Spell fixes        : {spell_str}")
    print(f"  Synonyms swapped   : {syn_str}")
    print(f"  Expansion          : +{exp}%  ({orig_cplx['token_count']} → {opt_cplx['token_count']} tokens)")
    print(f"  STS Preservation   : {sts['score']} ({sts['interpretation']})")
    print(f"  Unigram Overlap    : {overlap['unigram_precision']:.3f}  (how much orig survived)")
    print(f"  Fluency Before     : {orig_flu['score']:.2f}/10  (PP={orig_flu['bigram_perplexity']:.1f})")
    print(f"  Fluency After      : {opt_flu['score']:.2f}/10  (PP={opt_flu['bigram_perplexity']:.1f})")
    print(f"  Fluency Delta      : {flu_delta:+.2f}")
    print(f"  TTR Before         : {orig_lex['ttr']:.3f}")
    print(f"  TTR After          : {opt_lex['ttr']:.3f}")
    print(f"  GA Fitness         : {res['fitness_score']:.3f}")
    print(f"  LM Coherence       : {res['lm_scores']['coherence']:.3f}")
    print(f"  Aspects Detected   : {list(res['vibe']['aspects'].keys()) or 'none'}")
    print(f"  Mood               : {res['vibe']['mood'].upper()}")
    print(f"  NER Entities       : {res['entities_dict']}")
    print(f"  Noun Phrases       : {res['noun_phrases']}")

n = len(TEST_PROMPTS)
avg_exp = total_expansion / n
avg_flu = total_flu_delta / n
avg_ttr = total_ttr_delta / n
avg_sts = sum(total_sts_list) / len(total_sts_list) if total_sts_list else 0

print(f"\n{'='*80}")
print("AGGREGATE RESULTS")
print(f"{'='*80}")
print(f"  Avg Expansion       : +{avg_exp:.1f}%")
print(f"  Avg Fluency Delta   : {avg_flu:+.2f}/10")
print(f"  Avg TTR Delta       : {avg_ttr:+.3f}")
print(f"  Avg STS Preservation: {avg_sts:.3f}")
print()
print("  VERDICT:")
if avg_exp > 100 and avg_sts > 0.25:
    print("  [PASS] Optimizer significantly expands and enriches prompts")
    print("         while maintaining reasonable semantic intent.")
else:
    print("  [WARN] Enhancement may not be consistent across styles.")
print(f"{'='*80}")

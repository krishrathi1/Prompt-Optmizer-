import sys
sys.stdout.reconfigure(encoding='utf-8')

print('=== Testing ngram_lm.py ===')
from ngram_lm import score_prompt_fluency
result = score_prompt_fluency('a beautiful woman standing in a sunlit forest')
print(f'Fluency result: {result}')
print('ngram_lm.py: PASS\n')

print('=== Testing optimizer_engine.py ===')
from optimizer_engine import PromptOptimizer
opt = PromptOptimizer()
print(f'LM available:      {opt._lm_available}')
print(f'Stopwords loaded:  {len(opt._stopwords)} words')
print(f'Spell vocab size:  {len(opt._spell_vocab)} words')
print(f'TF-IDF fitted:     {opt._tfidf_vectorizer is not None}')

res = opt.optimize('a peacful girl in a sunlit tec lab', 'Photoreal')

print('\n--- Pipeline Log ---')
for line in res['pipeline_log']:
    print(line)

print('\n--- Optimized Prompt (first 120 chars) ---')
print(res['optimized_prompt'][:120])

cs = res['change_summary']
print('\n--- Change Summary ---')
print(cs['summary'])
print(f'Tokens: {cs["token_count_before"]} -> {cs["token_count_after"]} (+{cs["expansion_percent"]}%)')

lm = res['lm_scores']
print('\n--- LM Scores ---')
print(f'Bigram PP:  {lm.get("bigram_perplexity", 0):.2f}')
print(f'Coherence:  {lm.get("coherence", 0):.4f}')
print(f'GA Fitness: {res["fitness_score"]:.4f}')

print('\n--- Aspect Mining ---')
for asp, hits in res['vibe']['aspects'].items():
    print(f'  {asp}: {hits}')

print('\n--- Spelling Corrections ---')
for ch in res['spelling']['changes']:
    print(f'  {ch["from"]} -> {ch["to"]}')

print('\n--- Stem Analysis (first 5) ---')
for s in res['stem_analysis'][:5]:
    print(f'  {s["word"]} -> {s["stem"]}')

print('\noptimizer_engine.py: PASS')

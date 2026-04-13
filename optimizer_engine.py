import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk import RegexpParser
from nltk.sentiment import SentimentIntensityAnalyzer
import random
import re
import os
import difflib
import requests
from spellchecker import SpellChecker
from collections import defaultdict

def download_nltk_resources():
    local_nltk_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
    if not os.path.exists(local_nltk_path):
        os.makedirs(local_nltk_path)
    if local_nltk_path not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk_path)
    resources = [
        'punkt', 'averaged_perceptron_tagger', 'wordnet',
        'omw-1.4', 'vader_lexicon', 'maxent_ne_chunker', 'maxent_ne_chunker_tab', 'words', 'punkt_tab', 'averaged_perceptron_tagger_eng'
    ]

    resource_paths = {
        'punkt': ['tokenizers/punkt', 'tokenizers/punkt.zip'],
        'punkt_tab': ['tokenizers/punkt_tab/english/', 'tokenizers/punkt_tab.zip'],
        'averaged_perceptron_tagger': ['taggers/averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger.zip'],
        'averaged_perceptron_tagger_eng': ['taggers/averaged_perceptron_tagger_eng', 'taggers/averaged_perceptron_tagger_eng.zip'],
        'wordnet': ['corpora/wordnet', 'corpora/wordnet.zip'],
        'omw-1.4': ['corpora/omw-1.4', 'corpora/omw-1.4.zip'],
        'vader_lexicon': ['sentiment/vader_lexicon', 'sentiment/vader_lexicon.zip'],
        'maxent_ne_chunker': ['chunkers/maxent_ne_chunker', 'chunkers/maxent_ne_chunker.zip'],
        'maxent_ne_chunker_tab': ['chunkers/maxent_ne_chunker_tab/english_ace_multiclass/', 'chunkers/maxent_ne_chunker_tab.zip'],
        'words': ['corpora/words', 'corpora/words.zip'],
    }

    def resource_available(resource_name):
        for data_path in resource_paths.get(resource_name, []):
            try:
                nltk.data.find(data_path)
                return True
            except (LookupError, OSError):
                continue
        return False

    for resource in resources:
        if resource_available(resource):
            continue
        nltk.download(resource, download_dir=local_nltk_path, quiet=True)


POS_LABEL_MAP = {
    'NN': 'Noun', 'NNS': 'Noun', 'NNP': 'Noun', 'NNPS': 'Noun',
    'JJ': 'Adjective', 'JJR': 'Adjective', 'JJS': 'Adjective',
    'VB': 'Verb', 'VBD': 'Verb', 'VBG': 'Verb', 'VBN': 'Verb', 'VBP': 'Verb', 'VBZ': 'Verb',
    'RB': 'Adverb', 'RBR': 'Adverb', 'RBS': 'Adverb',
    'DT': 'Determiner', 'IN': 'Preposition', 'CC': 'Conjunction',
    'PRP': 'Pronoun', 'PRP$': 'Pronoun',
    'CD': 'Number', 'UH': 'Interjection',
}

POS_ROLE_MAP = {
    'Noun': 'Subject / Object',
    'Adjective': 'Modifier',
    'Verb': 'Action',
    'Adverb': 'Intensity Modifier',
    'Determiner': 'Article',
    'Preposition': 'Relation',
    'Conjunction': 'Connector',
    'Pronoun': 'Reference',
    'Number': 'Quantity',
    'Other': 'Functional',
}


class PromptOptimizer:
    def __init__(self):
        self._ne_chunker_retry_done = False
        self._spell_vocab = set()
        self._spell_buckets = defaultdict(list)
        try:
            download_nltk_resources()
            self.sia = SentimentIntensityAnalyzer()
            self.lemmatizer = WordNetLemmatizer()
            self.spell = SpellChecker()
            self._build_spelling_vocab()
        except Exception:
            self.sia = None
            self.lemmatizer = None
            self.spell = None

        self.expert_personas = {
            "Photoreal": "high-end DSLR photography, f/1.8 aperture, 85mm prime lens, razor-sharp focus, natural highlights, masterclass lighting, golden ratio composition",
            "Cinematic": "Arri Alexa footage, anamorphic bokeh, volumetric fog, high contrast, cinematic teal-and-orange grade, film grain, 2.39:1 aspect ratio",
            "Cyberpunk": "Vaporwave aesthetic, neon glow, wet reflective streets, futuristic urban sprawl, intricate raytracing, holographic interfaces, dark atmosphere",
            "Renaissance": "Chiaroscuro, Vermeer style, master oil painting, rich crackled pigment, classical framing, warm candlelight, Italian Renaissance composition",
            "Anime": "Ufotable studio style, dynamic cel-shading, high-octane color palette, ethereal light particles, intricate linework, key visual quality",
        }

    def _build_spelling_vocab(self):
        """Build a lightweight lexicon for typo correction."""
        vocab = set()
        try:
            vocab.update(
                w.lower() for w in nltk.corpus.words.words()
                if isinstance(w, str) and w.isalpha() and len(w) >= 2
            )
        except Exception:
            pass

        # Domain words common in image-generation prompts.
        vocab.update({
            "photoreal", "cinematic", "cyberpunk", "anime", "renaissance",
            "dslr", "bokeh", "volumetric", "raytracing", "chiaroscuro",
            "portrait", "landscape", "realistic", "lighting", "composition",
            "character", "school", "banana", "student", "walking", "eating",
        })

        self._spell_vocab = vocab
        self._spell_buckets = defaultdict(list)
        for word in self._spell_vocab:
            self._spell_buckets[word[0]].append(word)

    def _match_case(self, src_word, replacement):
        if src_word.isupper():
            return replacement.upper()
        if src_word[:1].isupper():
            return replacement.capitalize()
        return replacement

    def correct_spelling(self, prompt):
        """
        Correct misspellings using pyspellchecker and local vocab.
        Keeps punctuation/spaces intact and returns correction metadata.
        """
        if not prompt:
            return {"corrected_prompt": prompt, "changes": []}

        # Words to protect from correction
        protected = {"ai", "sd", "3d", "4k", "8k", "cgi", "dslr", "bokeh"}
        
        # Tokenize preserving punctuation
        parts = re.findall(r"[A-Za-z]+|[^A-Za-z]+", prompt)
        changes = []
        rebuilt = []

        for part in parts:
            if not part.isalpha():
                rebuilt.append(part)
                continue

            lower = part.lower()
            if len(lower) < 3 or lower in protected or lower in self._spell_vocab:
                rebuilt.append(part)
                continue

            # Try pyspellchecker first
            corrected = None
            if self.spell:
                # Find if it's unknown
                if self.spell.unknown([lower]):
                    corrected = self.spell.correction(lower)
            
            # Fallback to difflib if pyspellchecker fails or is unavailable
            if not corrected:
                candidates = [w for w in self._spell_buckets.get(lower[0], []) 
                             if abs(len(w) - len(lower)) <= 2]
                if candidates:
                    best = difflib.get_close_matches(lower, candidates, n=1, cutoff=0.82)
                    if best and best[0] != lower:
                        corrected = best[0]

            if corrected and corrected != lower:
                final_word = self._match_case(part, corrected)
                rebuilt.append(final_word)
                changes.append({"from": part, "to": final_word})
            else:
                rebuilt.append(part)

        return {
            "corrected_prompt": "".join(rebuilt),
            "changes": changes,
        }

    def ollama_enhance(self, prompt, model="llama3.2"):
        """Use local Ollama to brainstorm and enhance the prompt."""
        url = "http://localhost:11434/api/generate"
        system_prompt = (
            "You are a professional image prompting expert. "
            "Enhance the following prompt for Stable Diffusion. "
            "Describe textures, lighting, atmosphere, and camera details. "
            "Keep the core subject identical. Return ONLY the enhanced prompt. "
            "No chatter, no intro, just the text."
        )
        
        payload = {
            "model": model,
            "prompt": f"{system_prompt}\n\nUser prompt: {prompt}",
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 100}
        }
        
        try:
            logger.info(f"Requesting Ollama enhancement for model: {model}")
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                # Clean up potential model artifacts
                result = re.sub(r'^(Enhanced|Modified|Result|Prompt):\s*', '', result, flags=re.IGNORECASE)
                return result
            logger.warning(f"Ollama returned status {response.status_code}")
            return prompt
        except Exception as e:
            logger.error(f"Ollama connection error: {e}")
            return prompt

    # ------------------------------------------------------------------ #
    #  Core NLP helpers
    # ------------------------------------------------------------------ #

    def get_synonyms(self, word, pos):
        """Return the best synonym via WordNet, using semantic path similarity."""
        if not self.lemmatizer:
            return []
        
        wn_pos = None
        if pos.startswith('J'):
            wn_pos = wordnet.ADJ
        elif pos.startswith('V'):
            wn_pos = wordnet.VERB
        elif pos.startswith('N'):
            wn_pos = wordnet.NOUN
        
        if not wn_pos:
            return []

        # Lemmatize first (eating -> eat)
        lemma_word = self.lemmatizer.lemmatize(word.lower(), pos=wn_pos)
        synsets = wordnet.synsets(lemma_word, pos=wn_pos)
        if not synsets:
            return []
        
        original_synset = synsets[0]
        candidates = set()
        
        for syn in synsets:
            for l in syn.lemmas():
                name = l.name().replace('_', ' ')
                if name.lower() != lemma_word and name.lower() != word.lower() and len(name) > 1:
                    candidates.add(name)
        
        if not candidates:
            return []

        # Pick synonym with highest path similarity to original
        def score_synonym(s):
            s_synsets = wordnet.synsets(s.replace(' ', '_'), pos=wn_pos)
            if not s_synsets: return 0
            return original_synset.path_similarity(s_synsets[0]) or 0

        sorted_candidates = sorted(list(candidates), key=score_synonym, reverse=True)
        return sorted_candidates[:5]

    def get_negative_prompt(self, prompt, persona="Photoreal"):
        """Context-aware negative prompt shield."""
        standard_negatives = (
            "blurry, lowres, text, watermark, (worst quality:1.4), "
            "(low quality:1.4), signature, out of frame, jpeg artifacts"
        )
        is_portrait = bool(re.search(
            r'\b(woman|man|girl|boy|face|person|portrait|human)\b', prompt, re.I
        ))
        is_landscape = bool(re.search(
            r'\b(city|building|nature|forest|mountain|ocean|landscape|sky)\b', prompt, re.I
        ))

        context_negatives = ""
        if is_portrait:
            context_negatives = (
                "bad face, cross-eyed, deformed iris, extra fingers, "
                "mutated hands, bad anatomy, plastic skin, asymmetric face"
            )
        elif is_landscape:
            context_negatives = (
                "low resolution sky, tiling artifacts, messy foreground, "
                "distorted perspective, oversaturated, HDR clipping"
            )

        persona_negatives = {
            "Photoreal": "painting, cartoon, drawing, illustration, plastic skin, CGI",
            "Anime": "realistic, 3d render, photographic, grainy, noise, blurry",
            "Renaissance": "modern, futuristic, digital art, neon, plastic, photography",
            "Cyberpunk": "countryside, nature, daytime, pastoral, warm tones",
            "Cinematic": "amateur, snapshot, distorted lens, oversaturated, noise",
        }
        persona_neg = persona_negatives.get(persona, "")
        all_negs = [n for n in [standard_negatives, context_negatives, persona_neg] if n]
        return ", ".join(all_negs)

    def analyze_vibe(self, prompt):
        """Detect sentiment and suggest atmospheric lighting with high precision."""
        if not self.sia:
            return {
                "mood": "neutral",
                "lighting": "natural lighting",
                "color": "#94a3b8",
                "scores": {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0},
            }
        scores = self.sia.polarity_scores(prompt)
        compound = scores['compound']
        
        # Mapping continuous VADER score to 5-level lighting vocabulary
        if compound > 0.5:
            mood, color, light = "radiant", "#10b981", "radiant golden hour, bloom effect, glowing highlights"
        elif compound >= 0.05:
            mood, color, light = "warm", "#34d399", "warm studio lighting, soft diffused sunbeams"
        elif compound > -0.05:
            mood, color, light = "neutral", "#94a3b8", "balanced natural light, clean photography"
        elif compound >= -0.5:
            mood, color, light = "moody", "#60a5fa", "overcast, blue hour, cinematic fog, muted tones"
        else:
            mood, color, light = "dramatic", "#3b82f6", "low-key dramatic lighting, chiaroscuro, deep shadows"
            
        return {
            "mood": mood,
            "lighting": light,
            "color": color,
            "scores": scores,
        }

    def get_noun_phrases(self, tagged):
        """Extract noun phrases as chunks to preserve compound meaning."""
        grammar = "NP: {<DT>?<JJ>*<NN.*>+}"
        cp = RegexpParser(grammar)
        tree = cp.parse(tagged)
        phrases = []
        for subtree in tree:
            if isinstance(subtree, nltk.tree.Tree) and subtree.label() == 'NP':
                phrase = " ".join([word for word, tag in subtree.leaves()])
                phrases.append(phrase)
        return phrases

    def custom_ner(self, tagged):
        """Authentic Named Entity Recognition using NLTK chunker."""
        entities = []
        try:
            chunked = ne_chunk(tagged)
        except LookupError:
            # Retry once after bootstrap; if unavailable, gracefully skip NER.
            if self._ne_chunker_retry_done:
                return ""
            self._ne_chunker_retry_done = True
            download_nltk_resources()
            try:
                chunked = ne_chunk(tagged)
            except LookupError:
                return ""
        except Exception:
            return ""
        for subtree in chunked:
            if hasattr(subtree, 'label'):
                entity_name = " ".join([c[0] for c in subtree])
                entity_type = subtree.label()
                if entity_type == 'PERSON':
                    entities.append(f"hyper-detailed features for {entity_name}, subsurface scattering, micro-pore detail")
                elif entity_type in ('GPE', 'LOCATION', 'FACILITY'):
                    entities.append(f"atmospheric depth for {entity_name}, realistic textures, detailed surroundings")
                elif entity_type == 'ORGANIZATION':
                    entities.append(f"branding details for {entity_name}, architectural precision")
        return ", ".join(entities)

    # --- ADVANCED RESEARCH MODULES ---

    def get_specificity_data(self, word):
        """WordNet BFS to find the abstraction ladder (hypernym chain)."""
        synsets = wordnet.synsets(word, pos=wordnet.NOUN)
        if not synsets:
            return None
        
        # Get the primary synset
        s = synsets[0]
        paths = s.hypernym_paths()
        if not paths:
            return None
            
        # Longest path usually shows the richest hierarchy
        longest_path = max(paths, key=len)
        chain = [h.name().split('.')[0].replace('_', ' ') for h in longest_path]
        
        return {
            "root": chain[0],
            "ladder": chain,
            "depth": len(chain),
            "is_generic": len(chain) < 5
        }

    def extract_svo(self, tagged):
        """Extract Subject-Verb-Object triplets using pattern-based chunking."""
        grammar = """
          NP: {<DT>?<JJ>*<NN.*>+} 
          VP: {<VB.*>+}
        """
        cp = RegexpParser(grammar)
        tree = cp.parse(tagged)
        
        chunks = []
        for subtree in tree:
            if isinstance(subtree, nltk.tree.Tree):
                chunks.append((subtree.label(), " ".join([w for w, t in subtree.leaves()])))
        
        triplets = []
        for i in range(len(chunks) - 2):
            # Pattern: NP -> VP -> NP
            if chunks[i][0] == 'NP' and chunks[i+1][0] == 'VP' and chunks[i+2][0] == 'NP':
                triplets.append({
                    "subject": chunks[i][1],
                    "action": chunks[i+1][1],
                    "object": chunks[i+2][1]
                })
        return triplets

    # ------------------------------------------------------------------ #
    #  Genetic evolution
    # ------------------------------------------------------------------ #

    # --- NEW: TF-IDF Keyword Ranking ---
    def get_keyword_scores(self, prompt):
        """Use TF-IDF to identify semantically dense keywords."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            # Small reference corpus of high-quality descriptive prompts
            corpus = [
                "a hyper-detailed oil painting of a majestic dragon perched on a crystal mountain",
                "high-end DSLR photography of a peaceful girl in a sunlit tech lab, 85mm lens",
                "cyberpunk urban sprawl with neon glow and wet reflective streets, cinematic",
                "renaissance portrait of a noblewoman with chiaroscuro lighting and rich textures",
                "anime key visual of a dynamic battle scene with ethereal light particles"
            ]
            corpus.append(prompt)
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[-1]
            
            keyword_map = {feature_names[i]: float(scores[i]) for i in range(len(feature_names)) if scores[i] > 0}
            return keyword_map
        except Exception:
            return {}

    # ------------------------------------------------------------------ #
    #  Real Genetic evolution
    # ------------------------------------------------------------------ #

    def _crossover(self, parent1, parent2):
        """Recombine tokens from two parent prompts."""
        p1, p2 = list(parent1), list(parent2)
        split_point = random.randint(1, min(len(p1), len(p2)) - 1) if min(len(p1), len(p2)) > 1 else 0
        offspring = p1[:split_point] + p2[split_point:]
        return offspring

    def _mutate(self, tokens, rate=0.3):
        """Apply random mutations to token weights."""
        toks = list(tokens)
        for i in range(len(toks)):
            if random.random() < rate:
                # Only mutate unweighted tokens
                if not toks[i].startswith('('):
                    weight = random.choice([1.1, 1.2, 1.3, 1.4])
                    toks[i] = f"({toks[i]}:{weight})"
        return toks

    def _calculate_fitness(self, tokens, original_tokens, keyword_scores):
        """Fitness based on diversity, length, and keyword emphasis."""
        text = " ".join(tokens)
        word_count = len(text.split())
        weight_count = text.count(':1.')
        
        # Keyword emphasis bonus
        keyword_bonus = 0
        for token in tokens:
            clean_token = re.sub(r'[():123456789.]', '', token).lower()
            if clean_token in keyword_scores:
                keyword_bonus += keyword_scores[clean_token] * 10
        
        return word_count + (weight_count * 1.5) + keyword_bonus

    def evolve_prompt(self, base_tokens, keyword_scores, generations=3, pop_size=6):
        """Real Genetic Algorithm loop: Selection, Crossover, Mutation."""
        # Initial population
        population = [base_tokens]
        for _ in range(pop_size - 1):
            population.append(self._mutate(base_tokens, rate=0.4))
            
        for gen in range(generations):
            # 1. Selection
            scored_pop = [(self._calculate_fitness(p, base_tokens, keyword_scores), p) for p in population]
            scored_pop.sort(key=lambda x: x[0], reverse=True)
            
            # Keep top 2 winners (Elite)
            winners = [scored_pop[0][1], scored_pop[1][1]]
            
            # 2. Reproduction (Crossover + Mutation)
            new_pop = list(winners)
            while len(new_pop) < pop_size:
                child = self._crossover(winners[0], winners[1])
                child = self._mutate(child, rate=0.2)
                new_pop.append(child)
            population = new_pop
            
        final_best = max(population, key=lambda p: self._calculate_fitness(p, base_tokens, keyword_scores))
        return " ".join(final_best), float(self._calculate_fitness(final_best, base_tokens, keyword_scores))

    # ------------------------------------------------------------------ #
    #  Main pipeline
    # ------------------------------------------------------------------ #

    def optimize(self, prompt, style_preset="Photoreal", use_ollama=False):
        """Full 10-stage NLP optimization pipeline with local LLM (Ollama) support."""
        clean_prompt = prompt.strip()
        spellcheck = self.correct_spelling(clean_prompt)
        nlp_prompt = spellcheck["corrected_prompt"].strip()

        tokens = word_tokenize(nlp_prompt)
        tagged = pos_tag(tokens)
        
        # 1. Authentic NER and NP Chunking
        try:
            entities = self.custom_ner(tagged)
        except Exception:
            entities = ""
        try:
            noun_phrases = self.get_noun_phrases(tagged)
        except Exception:
            noun_phrases = []
        svo_triplets = self.extract_svo(tagged)
        
        # 2. TF-IDF Keyword Ranking
        keyword_scores = self.get_keyword_scores(nlp_prompt)
        
        # 3. Vibe and Sentiment
        vibe = self.analyze_vibe(nlp_prompt)

        # 4. Build rich linguistics, synonyms, and specificity
        linguistics = []
        base_mutation_tokens = []
        
        for word, pos in tagged:
            label = POS_LABEL_MAP.get(pos, 'Other')
            role = POS_ROLE_MAP.get(label, 'Functional')
            is_subject = pos.startswith('N')
            
            # Context-aware synonyms
            synonyms = self.get_synonyms(word, pos) if pos.startswith(('J', 'V')) else []
            replacement = (synonyms[0] if synonyms and pos.startswith('J') else word)
            
            # Specificity logic
            spec_data = self.get_specificity_data(word) if is_subject else None
            
            linguistics.append({
                "word": word,
                "pos": pos,
                "label": label,
                "role": role,
                "is_subject": is_subject,
                "tfidf_score": float(round(keyword_scores.get(word.lower(), 0), 3)),
                "optimized_to": replacement,
                "synonyms": synonyms,
                "specificity": spec_data,
                "changed": replacement != word,
            })
            base_mutation_tokens.append(replacement)

        # 5. Real Genetic Evolution (3 generations)
        evolved_text, fitness_score = self.evolve_prompt(base_mutation_tokens, keyword_scores)

        # 6. Brainstorm Enhancement (NEW Stage 9)
        final_nlp_text = evolved_text
        ollama_data = None
        if use_ollama:
            final_nlp_text = self.ollama_enhance(evolved_text)
            ollama_data = final_nlp_text

        # 7. Final Assembly
        persona_template = self.expert_personas.get(style_preset, self.expert_personas["Photoreal"])
        neg_prompt = self.get_negative_prompt(nlp_prompt, style_preset)

        final_parts = [final_nlp_text, persona_template, vibe['lighting']]
        if entities:
            final_parts.append(entities)
        final_prompt = ", ".join([p for p in final_parts if p])

        # 8. Pipeline stage metadata for UI (Extended to 10 steps)
        pipeline_stages = [
            {"step": 1, "name": "Spelling AI", "icon": "S", "color": "#f87171", "detail": f"Fixed {len(spellcheck['changes'])} typos", "data": spellcheck['changes']},
            {"step": 2, "name": "Tokenization", "icon": "T", "color": "#6366f1", "detail": f"{len(tokens)} tokens", "data": tokens},
            {"step": 3, "name": "SVO Extraction", "icon": "D", "color": "#8b5cf6", "detail": f"Extracted {len(svo_triplets)} triplets", "data": svo_triplets},
            {"step": 4, "name": "Keyword Ranking", "icon": "K", "color": "#3b82f6", "detail": "TF-IDF Density Analysis", "data": keyword_scores},
            {"step": 5, "name": "NP Chunking", "icon": "C", "color": "#06b6d4", "detail": f"Found {len(noun_phrases)} phrases", "data": noun_phrases},
            {"step": 6, "name": "Specificity analysis", "icon": "L", "color": "#0ea5e9", "detail": "BFS Abstraction Ladder built", "data": [l for l in linguistics if l['specificity']]},
            {"step": 7, "name": "Synonym Swapping", "icon": "W", "color": "#10b981", "detail": "Path Similarity substitution", "data": [l for l in linguistics if l['changed']]},
            {"step": 8, "name": "Genetic Evolution", "icon": "G", "color": "#ec4899", "detail": f"Evolution score: {fitness_score:.2f}", "data": {"final_score": fitness_score}},
            {"step": 9, "name": "Ollama Refinement", "icon": "B", "color": "#a855f7", "detail": "Local LLM Brainstorming" if use_ollama else "Bypassed", "data": ollama_data},
            {"step": 10, "name": "Vibe Analysis", "icon": "V", "color": vibe['color'], "detail": f"Mood: {vibe['mood'].upper()}", "data": vibe},
        ]

        pipeline_log = [
            f"[1] Spelling AI: {len(spellcheck['changes'])} token(s) fixed",
            f"[2] Tokenized {len(tokens)} words",
            f"[3] SVO Map built from {len(svo_triplets)} triplets",
            f"[4] TF-IDF Keyword Scores computed",
            f"[5] NP Chunking found {len(noun_phrases)} phrases",
            f"[6] WordNet Specificity analyzed",
            f"[7] Contextual synonyms applied",
            f"[8] GA Evolution complete (fitness: {fitness_score:.2f})",
            f"[9] Ollama Refining: {'Active' if use_ollama else 'Inactive'}",
            f"[10] Moody Analysis: {vibe['mood'].upper()}",
        ]

        return {
            "optimized_prompt": final_prompt,
            "corrected_prompt": nlp_prompt,
            "spelling": spellcheck,
            "negative_prompt": neg_prompt,
            "pipeline_log": pipeline_log,
            "pipeline_stages": pipeline_stages,
            "linguistics": linguistics,
            "vibe": vibe,
            "entities": entities,
            "noun_phrases": noun_phrases,
            "fitness_score": fitness_score,
            "settings": {"steps": 45 if use_ollama else 35, "cfg_scale": 9.5 if use_ollama else 8.0, "sampler": "DPM++ 2M Karras"},
        }


if __name__ == "__main__":
    opt = PromptOptimizer()
    res = opt.optimize("a peaceful girl in a sunlit tech lab", "Photoreal")
    print("=== OPTIMIZED PROMPT ===")
    print(res['optimized_prompt'])
    print("\n=== NEGATIVE ===")
    print(res['negative_prompt'])
    print(f"\n=== FITNESS SCORE ===")
    print(f"Genetic Fitness: {res['fitness_score']:.2f}")
    print(f"\n=== NOUN PHRASES ===")
    print(res['noun_phrases'])

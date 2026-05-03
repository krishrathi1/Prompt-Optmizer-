import csv
import random
import string

def generate_misspelling(word):
    if len(word) <= 3: return word
    
    noise_type = random.choice(['swap', 'delete', 'insert', 'replace', 'double'])
    idx = random.randint(0, len(word) - 1)
    
    if noise_type == 'swap' and idx < len(word) - 1:
        w_list = list(word)
        w_list[idx], w_list[idx+1] = w_list[idx+1], w_list[idx]
        return "".join(w_list)
    elif noise_type == 'delete':
        return word[:idx] + word[idx+1:]
    elif noise_type == 'insert':
        char = random.choice(string.ascii_lowercase)
        return word[:idx] + char + word[idx:]
    elif noise_type == 'replace':
        char = random.choice(string.ascii_lowercase)
        return word[:idx] + char + word[idx+1:]
    elif noise_type == 'double':
        return word[:idx] + word[idx] + word[idx:]
    
    return word

# Domain-specific words for Prompt Optimizer
vocab = [
    "peaceful", "monk", "mountain", "temple", "dusk", "cinematic", "lighting", "hyper", 
    "detailed", "photorealistic", "masterpiece", "portrait", "photography", "bokeh",
    "volumetric", "atmosphere", "mystical", "ancient", "cyberpunk", "futuristic",
    "neon", "glowing", "intricate", "ethereal", "majestic", "serene", "vibrant",
    "sharp", "focus", "elegant", "dynamic", "composition", "textured", "surface",
    "weathered", "organic", "synthetic", "landscape", "seascape", "architectural",
    "rendering", "unreal", "engine", "octane", "render", "digital", "painting",
    "concept", "art", "illustration", "sketch", "pencil", "watercolor", "oil",
    "canvas", "museum", "gallery", "exhibition", "bioluminescent", "nebula",
    "galaxy", "starry", "night", "twilight", "sunrise", "golden", "hour",
    "shadows", "contrast", "saturation", "exposure", "lens", "flare", "anamorphic"
]

# Add more general English words to reach 10k
# (In a real scenario, we'd use a large dictionary file)
general_words = ["house", "car", "tree", "river", "ocean", "forest", "desert", "street", "city", "village"] * 100
full_vocab = (vocab * 50) + general_words
random.shuffle(full_vocab)

data = []
for i in range(10000):
    original = random.choice(full_vocab)
    misspelled = generate_misspelling(original)
    data.append([original, misspelled])

with open('spelling_dataset_10k.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['original', 'misspelled'])
    writer.writerows(data)

print(f"Generated 10,000 rows in spelling_dataset_10k.csv")

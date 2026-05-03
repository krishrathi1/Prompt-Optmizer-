import csv
import random

subjects = [
    "monk", "warrior", "cyberpunk girl", "old man", "wizard", "robot", "astronaut", 
    "cat", "dragon", "samurai", "knight", "queen", "forest spirit", "mechanic",
    "scientist", "explorer", "goddess", "demon", "angel", "nomad"
]

locations = [
    "mountain temple", "neon city", "deep forest", "mars base", "ancient ruins",
    "underwater city", "floating island", "volcano", "space station", "medieval village",
    "cybernetic lab", "dreamscape", "desert oasis", "frozen tundra", "stellar nebula"
]

styles = [
    "Cinematic", "Photoreal", "Cyberpunk", "Renaissance", "Anime", "Dark Fantasy",
    "Steampunk", "Vaporwave", "Impressionist", "Minimalist"
]

modifiers = [
    "hyper detailed", "8k resolution", "volumetric lighting", "dramatic shadows",
    "intricate textures", "masterpiece quality", "sharp focus", "soft bokeh",
    "vibrant colors", "muted tones", "ethereal atmosphere", "majestic scale",
    "ray traced", "unreal engine 5", "octane render"
]

def generate_transformation():
    subject = random.choice(subjects)
    location = random.choice(locations)
    style = random.choice(styles)
    
    input_prompt = f"a {subject} in {location}"
    
    # Simulate the optimizer engine logic
    weighted_subject = f"({subject}:1.25)"
    selected_mods = random.sample(modifiers, k=random.randint(3, 5))
    
    output_prompt = f"{weighted_subject} in {location}, {style} style, " + ", ".join(selected_mods)
    
    return [input_prompt, output_prompt]

data = []
for i in range(10000):
    data.append(generate_transformation())

with open('transformation_dataset_10k.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['input_prompt', 'enhanced_prompt'])
    writer.writerows(data)

print(f"Generated 10,000 rows in transformation_dataset_10k.csv")

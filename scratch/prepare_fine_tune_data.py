import csv
import json
import random

def convert_to_jsonl(spelling_csv, transformation_csv, output_jsonl):
    data = []
    
    # Load Spelling Data
    with open(spelling_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "instruction": "Correct the spelling and typos in the following prompt fragment.",
                "input": row['misspelled'],
                "output": row['original']
            })
            
    # Load Transformation Data
    with open(transformation_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "instruction": "Optimize and enhance this simple prompt for high-quality Stable Diffusion image generation. Inject artistic modifiers and weights.",
                "input": row['input_prompt'],
                "output": row['enhanced_prompt']
            })
            
    random.shuffle(data)
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Converted {len(data)} rows to {output_jsonl}")

if __name__ == "__main__":
    convert_to_jsonl(
        'scratch/spelling_dataset_10k.csv', 
        'scratch/transformation_dataset_10k.csv', 
        'scratch/llama3_fine_tune_data.jsonl'
    )

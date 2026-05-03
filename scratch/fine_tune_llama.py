"""
Llama 3.2 Fine-Tuning Script for Prompt Optimizer PRO
======================================================
Uses Unsloth for 2x faster training and 70% less memory usage.
Targets: Spelling correction & Linguistic enhancement.
"""

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# 1. Configuration
model_name = "unsloth/Llama-3.2-3B-Instruct" # Base model
max_seq_length = 2048
load_in_4bit = True # Use 4bit quantization to save memory

# 2. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
)

# 3. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 4. Data Preparation
prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = prompt_style.format(instruction, input, output)
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("json", data_files="scratch/llama3_fine_tune_data.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# 5. Training Arguments
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Increase this for full training
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 6. Run Training
trainer_stats = trainer.train()

# 7. Save Model
model.save_pretrained("llama3_2_prompt_opt_lora") # Save LoRA
tokenizer.save_pretrained("llama3_2_prompt_opt_lora")

# 8. Export to GGUF (for Ollama)
# This requires unsloth to be installed with 'export' extras
# model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method = "q4_k_m")

print("Fine-tuning complete. Model saved to llama3_2_prompt_opt_lora")

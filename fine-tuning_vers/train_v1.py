# version 1: basic (only 2 modules q_proj, v_proj, base config)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_MODEL = './models/llama-3-sqlcoder-8b-base'
OUTPUT_DIR = './models/fine_tuned_v1'
prompt_file = 'prompt.md'
MAX_SEQ_LENGTH = 8192

with open(prompt_file, "r", encoding ="utf-8") as f:
    prompt_template = f.read()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map = "auto", dtype = torch.bfloat16, trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.config.pad_token_id = tokenizer.eos_token_id

def clean_schema(schema_dict):
    schema = '\n\n'.join(schema_dict.values())
    return schema

def generate_prompt(entry):
   
    question = entry.get("NL Query")
    schema = clean_schema(entry.get("Schema"))
    return {'text' :prompt_template.format(
        user_question = question,
        instructions = "Use Oracle 11g to generate an executable SQL for the following question, referencing the schema and its comments.",
        create_table_statements = schema) + entry['SQL Query'] + "\n```" + tokenizer.eos_token }

dataset = load_dataset('json', data_files={'train': 'train.jsonl', 'validation': 'val.jsonl'})
dataset = dataset.map(generate_prompt, remove_columns = dataset['train'].column_names)

print("Configuring LoRA ...\n")
lora_config = LoraConfig(
    r = 8,
    lora_alpha = 16,
    target_modules = ['q_proj', 'v_proj'],
    lora_dropout = 0.05,
    bias = 'none',
    task_type = "CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = SFTConfig(
    output_dir = OUTPUT_DIR,
    num_train_epochs = 5,
    per_device_eval_batch_size= 2,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    learning_rate = 2e-4,
    bf16 = True,
    logging_steps = 10,
    eval_strategy = "epoch",
    save_strategy= "epoch",
    save_total_limit = 2,
    load_best_model_at_end = True,
    metric_for_best_model = "eval_loss",
    greater_is_better = False,
    logging_dir = "./logs",
    report_to = 'tensorboard',
    warmup_ratio = 0.03,
    lr_scheduler_type = "cosine",
    max_length = MAX_SEQ_LENGTH,
    dataset_text_field = "text"
)

trainer = SFTTrainer(
    model = model,
    args = training_args,
    train_dataset = dataset['train'],
    eval_dataset = dataset['validation'],
    processing_class = tokenizer
)

print("Starting LoRA training...\n")

trainer.train()

print("LoRA training complete!\n")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done!! Model saved to", OUTPUT_DIR)



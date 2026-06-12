import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import argparse as ag
from retriever import get_tables

# End-to-End system inference, gets the schema from the retriever and includes it in the promt
# The predicted SQL are then saved in a json line file along with the NL query, gold SQL query and language (of NL)

parser = ag.ArgumentParser()
parser.add_argument('--ver', type=str, required=True)
parser.add_argument('--beam', type=int, default=4)
parser.add_argument('--input', type=str, default='test')

args = parser.parse_args()

BASE_MODEL = "../models/llama-3-sqlcoder-8b-base"
FT_MODEL = f"../models/fine_tuned_{args.ver}"


PROMPT_FILE = "../data_files/prompt.md"
OUTPUT_FILE = f"./pred/pred_{args.ver}_retriever_{args.input}.jsonl"
INPUT_FILE = f'../dataset/{args.input}.jsonl'

MAX_NEW_TOKENS = 512
MAX_INPUT_TOKENS = 7680  

with open(PROMPT_FILE, "r", encoding="utf-8") as f, open('../data_files/instructions.md', 'r', encoding='utf-8') as fl:
    prompt_template = f.read()
    instrc = fl.read()



def clean_schema(schema_dict):
    schema = '\n\n'.join(schema_dict.values())
    return schema

def generate_prompt(q):
    question = q.get("NL Query")
    schema = clean_schema(get_tables(question,10))

    return prompt_template.format(
        user_question=question,
        instructions=instrc, 
        create_table_statements=schema,
    )


def extract_sql(raw: str) -> str:
    match = re.search(r"```(?:sql)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    sql = match.group(1).strip() if match else raw.strip()
    return sql.split(";")[0].strip() + ";"


print("Loading model...")

if args.ver.lower() == 'base':
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
else:
    tokenizer = AutoTokenizer.from_pretrained(FT_MODEL) 

tokenizer.model_max_length = 8192 

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, 
    device_map="auto",
    trust_remote_code=True,
    dtype =  torch.bfloat16
)

if args.ver != 'base':
    model = PeftModel.from_pretrained(model, FT_MODEL) # Remove if base model 

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=False,
    num_beams= args.beam,
    max_new_tokens=MAX_NEW_TOKENS,
    return_full_text=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

print(f"Running inference on {len(data)} questions...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for i, q in enumerate(data):
        prompt = generate_prompt(q)

        n_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        if n_tokens > MAX_INPUT_TOKENS:
            print(f"  [Q{i}] Warning: prompt is {n_tokens} tokens (limit ~{MAX_INPUT_TOKENS})")

        output = pipe(prompt)[0]["generated_text"]
        sql = extract_sql(output)

        out.write(json.dumps({
            "NL Query": q.get("NL Query"),
            "Gold_SQL": q.get("SQL Query"),
            "Predicted_SQL": sql,
            "Language": q.get("Language"),
            "Difficulty": q.get("Difficulty")}, ensure_ascii=False) + "\n")

print("Done!")

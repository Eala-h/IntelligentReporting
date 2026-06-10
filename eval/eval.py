import json
import argparse as ag
from db_connect import connect, disconnect # Cnnection using oracledb (not included in repo for privacy)
from ex_eval import ex_accuracy, error_log
from cm_eval import cm_accuracy

parser = ag.ArgumentParser()

# base or fine-tuned model
parser.add_argument('--ver', type=str, required=True, default='base')

# Noise: N = generation done with gold schema , Y = Generation done using a prompt with noise + gold schema (not the retriever)
parser.add_argument('--noise', type=str, default='N') 

args = parser.parse_args()

if args.noise == 'N':
    noise = "gold"
    input_file = f'/pred/pred_{args.ver}_gold.jsonl'
else:
    noise = "noise"
    input_file = f'/pred/pred_{args.ver}_prompt2.jsonl'

output_file = f'/tests/eval_{args.ver}_{noise}.json'

connection, cursor = connect() 

with open(input_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

result_log = []
total_ex = ar_ex = en_ex = simple_ex = inter_ex = hard_ex = 0
total_cm = ar_cm = en_cm = simple_cm = inter_cm = hard_cm = 0.0
total_recall = total_precision = 0.0
ar_count = en_count = simple_count = inter_count = hard_count = 0

for i, entry in enumerate(data):

    nl = entry.get('NL Query')
    gold = entry.get('Gold_SQL')
    pred = entry.get('Predicted_SQL')
    lang = entry.get('Language')
    diff = entry.get('Difficulty')

    ex_score = ex_accuracy(i, pred, gold, cursor)
    total_ex += ex_score

    cm_score = cm_accuracy(pred, gold)
    total_cm += cm_score['Overall F1']

    total_recall += cm_score['Overall Recall']
    total_precision += cm_score['Overall Precision']

    if lang == 'Ar':
        ar_ex += ex_score
        ar_cm += cm_score['Overall F1']
        ar_count += 1
    else:
        en_ex += ex_score
        en_cm += cm_score['Overall F1']
        en_count += 1
        
    if diff == 'S':
        simple_ex += ex_score
        simple_cm += cm_score['Overall F1']
        simple_count += 1
    elif diff == 'I':
        inter_ex += ex_score
        inter_cm += cm_score['Overall F1']
        inter_count += 1
    elif diff == 'H':
        hard_ex += ex_score
        hard_cm += cm_score['Overall F1']
        hard_count += 1
        
    result_log.append({
        "Question":nl,
        "Language": lang,
        "Difficulty": diff,
        "Gold SQL": gold,
        "Predicted SQL": pred,
        "EX Score": ex_score,
        "CM Scores": cm_score,},)

    
print(f'EX Ar: {(ar_ex/ar_count) * 100:.2f}%\nEX En: {(en_ex/en_count) * 100:.2f}%\nCM Ar: {(ar_cm/ar_count) * 100:.2f}\nCM En: {(en_cm/en_count) * 100:.2f}\n')
result_log.append({
            "Result of": f"{ver} on validation set without space fixing and with retriever k = 10",   
            "Total EX": f'{(total_ex/len(data)) * 100:.2f}%',
            "EX En": f'{(en_ex/en_count) * 100:.2f}%',
            "EX Ar": f'{(ar_ex/ar_count) * 100:.2f}%',
            "EX Difficulty": f"Simple: {(simple_ex/simple_count) * 100:.2f}, Intermediate: {(inter_ex/inter_count) * 100:.2f}, Hard: {(hard_ex/hard_count) * 100:.2f}",
            "Avg CM": f'{(total_cm/len(data)) * 100:.2f}%',
            "CM En": f'{(en_cm/en_count) * 100:.2f}%',
            "CM Ar": f'{(ar_cm/ar_count) * 100:.2f}%',
            "CM Difficulty": f"Simple: {(simple_cm/simple_count) * 100:.2f}, Intermediate: {(inter_cm/inter_count) * 100:.2f}, Hard: {(hard_cm/hard_count) * 100:.2f}",
            "No. of": f"Ar: {ar_count}, En: {en_count}, Simple: {simple_count}, Intermediate: {inter_count}, Hard: {hard_count}",
    },)

with open(output_file, 'a', encoding='utf-8') as f:
    json.dump(result_log, f, ensure_ascii=False, indent=4)

with open('Error_log.txt', 'a', encoding='utf-8') as f:
    f.write(f'File: {output_file}\n')
    f.write('\n'.join(error_log) if error_log else 'No errors')
    f.write('\n\n')

disconnect(connection, cursor)

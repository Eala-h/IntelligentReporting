# Final Retriever Version

import json
import faiss
import bm25s
from sentence_transformers import SentenceTransformer
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.snowball import ArabicStemmer
import re


ar = ArabicStemmer()
en = EnglishStemmer()

model = SentenceTransformer('models/multilingual-e5-base')

def text_stemmer(text, language = 'En'): 
    tokens = text.lower().replace('_', ' ').replace('/', ' ').split()
    res = []
    if language == 'En':
        for token in tokens:
            res.append(en.stem(token))
    else: 
        for token in tokens:
            match = re.search('([\u0600-\u06FF]*)([A-Za-z]+)([\u0600-\u06FF]*)', token)
            ignore = ['', 'ال', 'و']
            if match:
                if match.group(1) not in ignore:
                    res.append(ar.stem(match.group(1)))
                res.append(en.stem(match.group(2)))
                if match.group(3) not in ignore:
                    res.append(ar.stem(match.group(3)))
            else:
                res.append(ar.stem(token))
                
    return ' '.join(res)

def create_passage(table): 
    parts = [f"passage: {table['table'].lower()}."]
    parts.append(f"Columns: {', '.join(table['columns'])}")
    if table.get('description_En'):
        parts.append(table['description_En'])
        desc_en = text_stemmer(table['description_En'])
        parts.append(desc_en)
    if table.get('description_Ar'):
        parts.append(table['description_Ar'])
        desc_ar = text_stemmer(table['description_Ar'], 'Ar')
        parts.append(desc_ar)
    if table.get('comments') != 'None':
        parts.append(f"Comments: {', '.join(table['comments'])}")
    return ' '.join(parts)



with open('../data_files/tables.json', 'r', encoding='utf-8') as f:
    all_tables = json.load(f)

table_names = [t['table'] for t in all_tables]
passages = [create_passage(t) for t in all_tables]


with open('../data_files/empty_tables.json', 'r') as fl:
    empty_tables = json.load(fl)
empty_tables = set(e.lower() for e in empty_tables) 

# Tokenize and index BM25
retriever = bm25s.BM25()
corpus_tokens = bm25s.tokenize(passages)
retriever.index(corpus_tokens)

index = faiss.read_index('faiss_index/schema_index.faiss')


def retrieve_faiss(nl_query, top_k=5):
    nl_embedding = model.encode([f'query: {nl_query}'], normalize_embeddings=True, convert_to_numpy=True)
    scores, indices = index.search(nl_embedding, top_k)

    result = []
    for score, i in zip(scores[0], indices[0]):
        result.append({"table": table_names[i], "score": f'{float(score):.4f}'})
    return result

def exact_name_match(nl_query):
    check_ar = re.search('[\u0600-\u06FF]+', nl_query)
    if check_ar:
        nl_query = text_stemmer(nl_query, 'Ar')

    query_lower = nl_query.lower()  
    matched = []
    for table in table_names:
        table_lower = table.lower()
        table_readable = table_lower.replace('_', ' ')
        
        if table_readable in query_lower or table_lower in query_lower:
            matched.append(table)
            continue
        elif text_stemmer(table_readable) in query_lower or text_stemmer(table_lower) in query_lower:
            matched.append(table)
            continue
        
        parts = [p for p in table_lower.split('_') if len(p) > 3]
        if any(part in query_lower for part in parts):
            matched.append(table)

    return matched

def retrieve_bm25(nl_query, top_k=10):
    query_tokens = bm25s.tokenize(nl_query)
    indices, scores = retriever.retrieve(query_tokens, k=top_k)
    result = []
    for score, i in zip(scores[0], indices[0]):
        if score > 0:
            result.append({"table": table_names[i], "score": f'{float(score):.4f}'})
    return result

def retrieve_hybrid(nl_query, top_k):
    result_faiss = retrieve_faiss(nl_query, top_k)
    result_bm = retrieve_bm25(nl_query, top_k)
    tables_faiss = [r['table'].lower() for r in result_faiss]
    tables_bm = [r['table'].lower() for r in result_bm]
    scores = {}
    for rank, table in enumerate(tables_bm):
        scores[table] = scores.get(table, 0) + 1 / (60 + rank + 1)
    for rank, table in enumerate(tables_faiss):
        scores[table] = scores.get(table, 0) + 1 / (60 + rank + 1)

    exact_matches = exact_name_match(nl_query)
    for table in exact_matches:
        t = table.lower()
        if t in scores:
            scores[t] = 1.500  
        else:
            scores[t] = 1.000 

    merged = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
    return [{"table": t, "score": f"{scores[t]:.4f}"} for t in merged if t.lower() not in empty_tables][:top_k]


work_set = '../dataset/test.jsonl'

with open(work_set, 'r', encoding='utf-8') as f:
    test = [json.loads(line) for line in f if line.strip()]

result_log_faiss = []
result_log_bm = []
avg_f = 0
avg_b = 0
avg_h = 0
prec = 0
ar_f = ar_b = ar_h = ar_count = ar_p = 0
en_f = en_b = en_h = en_count = en_p = 0
count_total = 0
result_log = []

for t in test:
    k = 10
    question = t.get('NL Query')
    tables = set(t.get('Tables'))
    lang = t.get('Language')

    result_faiss = retrieve_faiss(question, k)
    result_bm = retrieve_bm25(question, k)
    
    tables_faiss = set(r['table'].lower() for r in result_faiss)
    tables_bm = set(r['table'].lower() for r in result_bm)
    
    result_hybrid = retrieve_hybrid(question, k)
    tables_hybrid = set(r['table'].lower() for r in result_hybrid)

    tp_faiss = len(tables & tables_faiss)
    recall_faiss = tp_faiss / len(tables) if tables else 0

    tp_bm = len(tables & tables_bm)
    recall_bm = tp_bm / len(tables) if tables else 0

    avg_f += recall_faiss
    avg_b += recall_bm
    
    tp = len(tables & tables_hybrid)
    recall = tp / len(tables) if tables else 0
    avg_h += recall
    precision = tp / len(tables_hybrid)
    prec += precision


    if lang == 'Ar':
        ar_f += recall_faiss
        ar_b += recall_bm
        ar_h += recall
        ar_p += precision
        ar_count += 1
    elif lang == 'En':
        en_f += recall_faiss
        en_b += recall_bm
        en_h += recall
        en_p += precision
        en_count += 1
    
    count_total += 1
    result_log.append({"NL Query": question,
                       "Gold Tables": list(tables),
                       "Retrieved": [r['table'].lower() for r in result_hybrid],
                       "Scores": [r['score'] for r in result_hybrid],
                       "Recall": recall})

#recalls = [y['Recall'] for y in result_log]
print(f"Recalls Faiss AVG: {avg_f/count_total:.4f}")
print(f"Recalls BM25 AVG: {avg_b/count_total:.4f}\n")
   

avg_recall = avg_h / count_total
result_log.append({'Result of': f'Version Final (stemming + exact match) on {work_set} with top_k = {k} cap',
                   'Recall Hybrid' : f'{avg_recall:.4f}',
                   'Recall FAISS': f'{avg_f/count_total:.4f}',
                   'Recall BM25': f'{avg_b/count_total:.4f}',
                   'ARABIC Recalls Faiss AVG' : f'{ar_f/ar_count:.4f}',
                   'ARABIC Recalls BM25 AVG': f'{ar_b/ar_count:.4f}',
                   'ARABIC Recalls Hybrid AVG': f'{ar_h/ar_count:.4f}',
                   'ENGLISH Recalls Faiss AVG': f'{en_f/en_count:.4f}',
                   'ENGLISH Recalls BM25 AVG': f'{en_b/en_count:.4f}',
                   'ENGLISH Recalls Hybrid AVG': f'{en_h/en_count:.4f}'},)


print(f"Recalls Hybrid AVG: {avg_recall:.4f}\nPrecision: {prec/count_total:.4f}\n")

print(f"ARABIC Recalls Faiss AVG: {ar_f/ar_count:.4f}")
print(f"ARABIC Recalls BM25 AVG: {ar_b/ar_count:.4f}")
print(f"ARABIC Recalls Hybrid AVG: {ar_h/ar_count:.4f}\n")
print(f"ARABIC Precision: {ar_p/ar_count:.4f}\n")

print(f"ENGLISH Recalls Faiss AVG: {en_f/en_count:.4f}")
print(f"ENGLISH Recalls BM25 AVG: {en_b/en_count:.4f}")
print(f"ENGLISH Recalls Hybrid AVG: {en_h/en_count:.4f}")
print(f"ENGLISH Precision: {en_p/en_count:.4f}")

"""
with open('Results.json', 'a', encoding='utf-8') as f:
    json.dump(result_log, f, indent= 4, ensure_ascii=False)"""

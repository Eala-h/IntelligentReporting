import json
import faiss
import bm25s
from sentence_transformers import SentenceTransformer
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.snowball import ArabicStemmer
import re


ar = ArabicStemmer()
en = EnglishStemmer()

model = SentenceTransformer('../retriever_tests/models/multilingual-e5-base') 

def text_stemmer(text, language = 'En'): 
    tokens = text.lower().replace('_', ' ').replace('/', ' ').split() 
    res = []
    if language == 'En':
        for token in tokens:
            res.append(en.stem(token))
    else: 
        # Split and stem English words in Arabic queries "code-switching" (e.g. عدد الusers )
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
                # Stem Arabic words
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
    # The file contains a structured dict for each table as following: {"table": , "columns": [], "description_En": , "description_Ar": , "comments": []}
    tables = json.load(f)

table_names = [t['table'] for t in tables]
passages = [create_passage(t) for t in tables]


with open('../data_files/empty_tables.json', 'r') as fl:
    # The file contains a list with the names of tables with no data
    empty_tables = json.load(fl)
empty_tables = set(e.lower() for e in empty_tables) 

# Tokenize and index passages using BM25
retriever = bm25s.BM25()
corpus_tokens = bm25s.tokenize(passages)
retriever.index(corpus_tokens)

# Load FAISS index
index = faiss.read_index('schema_index.faiss')


def retrieve_faiss(nl_query, top_k):
    # Embed the NL query using "Multilingual-E5-Base" (found in the top of the script)
     nl_embedding = model.encode([f'query: {nl_query}'], normalize_embeddings=True, convert_to_numpy=True)
    # Search using/within FAISS index
    scores, indices = index.search(nl_embedding, top_k)

    result = []
    for score, i in zip(scores[0], indices[0]):
        result.append({"table": table_names[i], "score": f'{float(score):.3f}'})
    return result

def exact_name_match(nl_query):
    # Check the language of the NL query, if in Arabic --> stem to check for English words within (to match to table names)
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
        # Match parts of the table name + Exclude any abbreviations or cryptic names (e.g. USR, BML)
        parts = [p for p in table_lower.split('_') if len(p) > 3] 
        if any(part in query_lower for part in parts):
            matched.append(table)

    return matched

def retrieve_bm25(nl_query, top_k):
    query_tokens = bm25s.tokenize(nl_query)
    indices, scores = retriever.retrieve(query_tokens, k=top_k)
    result = []
    for score, i in zip(scores[0], indices[0]):
        if score > 0:
            result.append({"table": table_names[i], "score": f'{float(score):.3f}'})
    return result

def retrieve_hybrid(nl_query, top_k):
    result_faiss = retrieve_faiss(nl_query, top_k)
    result_bm = retrieve_bm25(nl_query, top_k)
    tables_faiss = [r['table'].lower() for r in result_faiss]
    tables_bm = [r['table'].lower() for r in result_bm]
    scores = {}
    # Find score using Reciprocal Rank Fusion (RRF). Depends on the order/rank the retrieved tables are in, the higher the rank the better the score
    # E.g. table X in BM25 list rank/order = 2 --> rrf score = 0 + 1 / 63 = 0.015
    # in FAISS list rank/order = 0 --> rrf score = 0.015 (score from BM25) + 1 / 61 = 0.032
    for rank, table in enumerate(tables_bm):
        scores[table] = scores.get(table, 0) + 1 / (60 + rank + 1) 
    for rank, table in enumerate(tables_faiss):
        scores[table] = scores.get(table, 0) + 1 / (60 + rank + 1)

    # Get table name exact match and add to the top of the list
    exact_matches = exact_name_match(nl_query)
    for table in exact_matches:
        t = table.lower()
        if t in scores:
            scores[t] = 1.500 # Tables already found in the retrieved tables --> placed first
        else:
            scores[t] = 1.000 

    merged = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
    return [{"table": t, "score": f"{scores[t]:.3f}"} for t in merged if t.lower() not in empty_tables][:top_k]

with open('../data_files/tables.sql', 'r', encoding='utf-8') as f:
    # The file contains all the datbase tables in DDL format
    full_schema = f.read()

def get_tables(nl_query, top_k):
    # Retrieve the tables to be passed to the SQL generator
    retrieved = retrieve_hybrid(nl_query, top_k)
    retrieved_table_names = set(r['table'].lower() for r in retrieved)
    extracted_table=[]  
    for table in retrieved_table_names:
        pattern = re.compile(rf"CREATE\s+TABLE\s+{re.escape(table)}\b[\s\S]*?(?=^\s*-{{5,}}\s*$)", re.IGNORECASE | re.MULTILINE)
        match = pattern.findall(full_schema)
        extracted_table.extend(match)
    record = dict(zip(retrieved_table_names, extracted_table))
    return record
    

#Test run   
'''nl = 'Give me the first names of all users.'
print(get_tables(nl, 5))'''

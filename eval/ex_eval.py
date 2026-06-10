import oracledb
import sys
import json

# Logging execution errors from the generated SQL queries
error_log = []
def exe_errors(n, e):
    error_log.append(f"SQL Error in query {n}: {e}")

# Removing SQL markings (```) found in the prompt, and removing (;) to execute correctly via oracledb
def clean_sql(sql_string):
    if not sql_string:
        return "" 
    sql_string = sql_string.replace("```", "").strip()
    sql_string = " ".join(sql_string.split())
    return sql_string.strip().rstrip(';')

# Normalizing (NULL) values to strings and keeping uniform numerical values
def normalize_value(item):
    if item is None:
        return "none"
    
    s = str(item).strip().lower()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else f"{f:.6f}".rstrip('0')
    except ValueError:
        return s
    
# Rearranging the excutions result column by column to mitigate false negatives due to differences between the generated and gold column order and names.
def canonicalize(cursor_rows):
    normalized = []

    for row in cursor_rows:
        normalized.append(tuple(normalize_value(val) for val in row))
 
    normalized = sorted(normalized)

    # For empty execution results (returns no data)
    if not normalized:
        return []
    
    # Transpose row --> column (creating value vectors)
    num_cols = len(normalized[0])

    columns = []
    for col_idx in range(num_cols):
        col_values = tuple(row[col_idx] for row in normalized)
        columns.append(col_values)

    #
    columns_sorted = sorted(columns)

    # Re-build the result table 
    canonical = []
    num_rows = len(normalized)

    for row_idx in range(num_rows):
        row = tuple(col[row_idx] for col in columns_sorted)
        canonical.append(row)

    return canonical

def ex_accuracy(num, pred_sql, gold_sql, cursor):
    try:
        pred = clean_sql(pred_sql)
        gold = clean_sql(gold_sql)
        
        cursor.execute(gold)
        gold_rows = cursor.fetchall()
        gold_frame = canonicalize(gold_rows)
    
        cursor.execute(pred)
        pred_rows = cursor.fetchall()
        pred_frame = canonicalize(pred_rows)

        return 1 if gold_frame == pred_frame else 0 # Exact (cell to cell) match
    
    except Exception as e:
        exe_errors(num, e)
        return 0


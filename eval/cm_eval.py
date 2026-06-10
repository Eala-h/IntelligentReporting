import sqlglot
from sqlglot import exp


def _in_subquery(node) -> bool:
    return node.find_ancestor(exp.Subquery) is not None


def _unwrap_col(node):
    if isinstance(node, exp.Column):
        return node
    if isinstance(node, (exp.Lower, exp.Upper, exp.Trim)):
        return _unwrap_col(node.this)
    if isinstance(node, exp.Cast):
        return _unwrap_col(node.this)
    if isinstance(node, (exp.Anonymous, exp.Func)):
        for arg in node.args.values():
            if isinstance(arg, list):
                for item in arg:
                    result = _unwrap_col(item)
                    if result:
                        return result
            else:
                result = _unwrap_col(arg)
                if result:
                    return result
    return None


def get_components(sql):
    components = {
        "All Columns": set(),
        "Tables":         set(),
        "CTE":            set(),
        "Joins":          set(),
        "Filters":        set(),
        "Group by":       set(),
        "Having":         set(),
        "Aggregations":   set(),
    }

    try:
        tree = sqlglot.parse_one(sql, read="oracle")
    except Exception:
        return components

    # CTE
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    components["CTE"].update(cte_names)

    # Tables
    for t in tree.find_all(exp.Table):
        name = t.name.lower()
        if name and name not in cte_names:
            components["Tables"].add(name)

    # Joins: explicit ON columns (as in: tableY y join tableX x on x.column = y.column)
    join_col_names = set()
    for j in tree.find_all(exp.Join):
        for col in j.find_all(exp.Column):
            join_col_names.add(col.name.lower())


    # Implicit joins: col = col in WHERE (as in: FROM tableX, tableY WHERE x.column = y.column)
    #Equivalent to inner joins found in Oracle g11, also named: equi-join) 
    imp_join_pairs = set()
    for where in tree.find_all(exp.Where):
        for eq in where.find_all(exp.EQ):
            l, r = eq.left, eq.right
            if isinstance(l, exp.Column) and isinstance(r, exp.Column):
                if not _in_subquery(eq):
                    a, b = l.name.lower(), r.name.lower()
                    imp_join_pairs.add((a, b))
                    imp_join_pairs.add((b, a))
                    join_col_names.add(a)
                    join_col_names.add(b)

    components["Joins"].update(join_col_names)

    pseudo_cols = {"rownum", "rn"} # rownum and rn are used to sereve as LIMIT since LIMIT is not an Oracle g11 clause.
    
    # All columns used in the query
    for sel in tree.find_all(exp.Select):
        for col in sel.find_all(exp.Column):
            if col.name.lower() not in pseudo_cols:
              components["All Columns"].add(col.name.lower())   

    # Filters
    def add_filter(node):
        col = _unwrap_col(node)
        if col and col.name.lower() not in pseudo_cols:
            components["Filters"].add(col.name.lower())

    for where in tree.find_all(exp.Where):
        # EQ + skip equi-joins
        for eq in where.find_all(exp.EQ):
            l, r = eq.left, eq.right
            lc, rc = _unwrap_col(l), _unwrap_col(r)
            if lc and rc:
                pair = (lc.name.lower(), rc.name.lower())
                if pair not in imp_join_pairs:
                    add_filter(l)
            elif lc:
                add_filter(l)

        # NEQ
        for neq in where.find_all(exp.NEQ):
            add_filter(neq.left)

        # Range
        for cls in (exp.GT, exp.LT, exp.GTE, exp.LTE):
            for node in where.find_all(cls):
                add_filter(node.left)

        # IS / IS NOT NULL
        for ss in where.find_all(exp.Is):
            add_filter(ss.this)

        # IN:  both directions (col IN (vals) and 'val' IN (cols))
        for nn in where.find_all(exp.In):
            add_filter(nn.this)
            for item in nn.expressions:
                add_filter(item)

        # BETWEEN
        for bet in where.find_all(exp.Between):
            add_filter(bet.this)

        # LIKE / ILIKE (unwraps LOWER(col) LIKE '...')
        for cls in (exp.Like, exp.ILike):
            for like in where.find_all(cls):
                add_filter(like.this)

        # NOT LIKE
        for nlike in where.find_all(exp.Not):
            inner = nlike.this
            if isinstance(inner, (exp.Like, exp.ILike)):
                add_filter(inner.this)

    # Group by
    for group in tree.find_all(exp.Group):
        for col in group.find_all(exp.Column):
            components["Group by"].add(col.name.lower())

    # Having
    for having in tree.find_all(exp.Having):
        for col in having.find_all(exp.Column):
            components["Having"].add(col.name.lower())

    # Aggregations: (func, column) pairs
    agr_types = (exp.Count, exp.Avg, exp.Max, exp.Min, exp.Sum)
    for agr in tree.find_all(agr_types):
        func = type(agr).__name__.lower()
        inner_cols = [c.name.lower() for c in agr.find_all(exp.Column)]
        if inner_cols:
            for c in inner_cols:
                components["Aggregations"].add(f"{func}({c})")
        else:
            components["Aggregations"].add(f"{func}(*)")

    return components


def cm_accuracy(pred_sql, gold_sql):
    gold_comp = get_components(gold_sql)
    pred_comp = get_components(pred_sql)

    scores = {} # for total (macro) f1 score 
    comp_pres = [] # for total precision
    comp_rec = [] # for totla recall


    for key in gold_comp:
        g_set = gold_comp[key]
        p_set = pred_comp[key]

        if not g_set and not p_set:
            scores[key] = -1.0
            continue

        if not g_set or not p_set:
            scores[key] = 0.0
            comp_pres.append(0.0)
            comp_rec.append(0.0)

            continue

        tp        = len(g_set & p_set)
        precision = tp / len(p_set)
        recall    = tp / len(g_set)
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

        scores[key] = round(f1, 3)
        comp_rec.append(recall)
        comp_pres.append(precision)


    relevant = [v for v in scores.values() if v != -1.0]
    scores["Overall F1"]        = round(sum(relevant) / len(relevant), 3) if relevant else 0.0
    scores["Overall Precision"] = round(sum(comp_pres) / len(comp_pres), 3) if comp_pres else 0.0
    scores["Overall Recall"]    = round(sum(comp_rec) / len(comp_rec), 3) if comp_rec else 0.0

    return scores

'''
if __name__ == "__main__":
    import json


versions = ['base', 'v1','v2','v3','v4','v5','v6']

for ver in versions:
    with open(f'../final_tests/pred_{ver}_gold_val.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    total_cm = 0.0

    for entry in data:

        question = entry.get('NL Query')
        language = entry.get('Language')
        gold = entry.get('Gold_SQL')
        pred = entry.get('Predicted_SQL')
        dif = entry.get('Difficulty')

        cm_scores = cm_accuracy(pred, gold)

        total_cm += cm_scores["Overall F1"]

    
    print(f"CM of {ver} = {(total_cm/len(data))*100}")
'''



import networkx as nx
import re

ALIAS = {'title': 't',
		 'kind_type': 'kt',
		 'movie_info': 'mi',
		 'movie_info_idx': 'mii',
		 'info_type': 'it',
		 'cast_info': 'ci',
		 'role_type': 'rt',
		 'keyword': 'k',
		 'name': 'n',
		 'movie_companies': 'mc',
		 'movie_keyword': 'mk',
		 'company_name': 'cn',
		 'company_type': 'ct',
		 'aka_name': 'an',
		 'person_info': 'pi'}
reverse_alias = {value: key for key, value in ALIAS.items()}

def find_connected_clusters(pairs):
    # Create a graph
    G = nx.Graph()

    # Add edges to the graph from the pairs
    G.add_edges_from(pairs)

    # Find all connected components (clusters)
    clusters = list(nx.connected_components(G))

    # Convert each set to a list (optional)
    clusters = [list(cluster) for cluster in clusters]

    return clusters


def parse_where(sql_query, sql_alias):
    if "::float" in sql_query:
        sql_query = sql_query.replace("::float", "")
    elif "::int" in sql_query:
        sql_query = sql_query.replace("::int", "")
    # really fucking dumb
    bad_str1 = "mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    bad_str2 = "mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    if bad_str1 in sql_query:
        sql_query = sql_query.replace(bad_str1, "")

    if bad_str2 in sql_query:
        sql_query = sql_query.replace(bad_str2, "")

    pattern = r'WHERE\s+(.*)'

    # Search for the pattern in the SQL query
    match = re.search(pattern, sql_query, re.IGNORECASE | re.DOTALL)
    joins = []
    preds = []
    pred_cols = []
    pred_types = []
    pred_vals = []

    col2id = {}

    if match:
        sub_str = match.group(1).strip().strip(';')
        pred_parts = sub_str.split(' AND ')
        for part in pred_parts:
            if ' = ' in part:
                left = part.split(' = ')[0]
                right = part.split(' = ')[1]
                if left.split('.')[0] in sql_alias and right.split('.')[0] in sql_alias:
                    # is join condition
                    joins.append(part)
                else:
                    if part[0] == '(' and part[-1] == ')':
                        processed_part = part[1:-1]
                    else:
                        processed_part = part

                    col_name = processed_part.split(' = ')[0]
                    col_val = processed_part.split(' = ')[-1]

                    if col_name not in col2id:
                        col2id[col_name] = len(preds)

                        preds.append([part])
                        pred_cols.append([col_name])
                        pred_types.append(['eq'])
                        col_val = col_val.strip("'")
                        pred_vals.append([int(col_val)])
                    else:
                        preds[col2id[col_name]].append(part)
                        pred_cols[col2id[col_name]].append(col_name)
                        pred_types[col2id[col_name]].append('eq')
                        pred_vals[col2id[col_name]].append(int(col_val))
            else:
                # filter condition
                if ' <= ' in part:
                    if part[0] == '(' and  part[-1] == ')' :
                        processed_part = part[1:-1]
                    else:
                        processed_part = part

                    col_name = processed_part.split(' <= ')[0]
                    col_val = int(processed_part.split(' <= ')[-1])

                    if col_name not in col2id:
                        col2id[col_name] = len(preds)

                        preds.append([part])
                        pred_cols.append([col_name])
                        pred_types.append(['lt'])
                        pred_vals.append([[None, col_val]])
                    else:
                        preds[col2id[col_name]].append(part)
                        pred_cols[col2id[col_name]].append(col_name)
                        pred_types[col2id[col_name]].append('lt')
                        pred_vals[col2id[col_name]].append([None, col_val])

                elif ' >= ' in part:
                    if part[0] == '(' and part[-1] == ')':
                        processed_part = part[1:-1]
                    else:
                        processed_part = part

                    col_name = processed_part.split(' >= ')[0]
                    col_val = int(processed_part.split(' >= ')[-1])

                    if col_name not in col2id:
                        col2id[col_name] = len(preds)

                        preds.append([part])
                        pred_cols.append([col_name])
                        pred_types.append(['lt'])
                        pred_vals.append([[col_val, None]])
                    else:
                        preds[col2id[col_name]].append(part)
                        pred_cols[col2id[col_name]].append(col_name)
                        pred_types[col2id[col_name]].append('lt')
                        pred_vals[col2id[col_name]].append([col_val, None])
                elif ' in ' in part: # for the case of 'in'
                    if part[0] == '(' and part[-1] == ')':
                        processed_part = part[1:-1]
                    else:
                        processed_part = part

                    col_name = processed_part.split(' in ')[0]
                    col_val = processed_part.split(' in ')[-1]

                    if ' OR n.gender IS NULL' in col_val:
                        col_val = col_val.replace(' OR n.gender IS NULL', '')

                    if col_name not in col2id:
                        col2id[col_name] = len(preds)

                        preds.append([part])
                        pred_cols.append([col_name])
                        pred_types.append(['in'])
                        col_val = col_val.strip("()").replace("'", "").split(",")
                        pred_vals.append([col_val])
                    else:
                        preds[col2id[col_name]].append(part)
                        pred_cols[col2id[col_name]].append(col_name)
                        pred_types[col2id[col_name]].append('in')
                        col_val = col_val.strip("()").replace("'", "").split(",")
                        pred_vals[col2id[col_name]].append(col_val)

        return joins, preds, pred_cols, pred_types, pred_vals

    else:
        return [], [], [], [], []

def extract_from_clause(sql_query):
    # Regular expression pattern to find the content after FROM
    if "::float" in sql_query:
        sql_query = sql_query.replace("::float", "")
    elif "::int" in sql_query:
        sql_query = sql_query.replace("::int", "")
    # really fucking dumb
    bad_str1 = "mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    bad_str2 = "mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND"
    if bad_str1 in sql_query:
        sql_query = sql_query.replace(bad_str1, "")

    if bad_str2 in sql_query:
        sql_query = sql_query.replace(bad_str2, "")

    sql_query = sql_query.strip("\n").strip()
    pattern = r'FROM\s+(.*?)(\s+WHERE|\s*$)'

    # Search for the pattern in the SQL query
    match = re.search(pattern, sql_query, re.IGNORECASE | re.DOTALL)

    tables = []
    aliases = []

    if match:
        from_string = match.group(1).strip()
        t_parts = from_string.split(',')
        for full_t in t_parts:
            full_t = full_t.lower().strip().strip(';')
            tables.append(full_t.split(' as ')[0])
            aliases.append(full_t.split(' as ')[1])

        return tables, aliases
    else:
        return None


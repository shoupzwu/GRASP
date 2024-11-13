import copy
from dsb_utlities.NeuroCDF_helpers import *
import random

OPS = ['<', '<=', '>', '>=', '=']
IN_BUCKETS = 100

def get_table_info(file_path='./queries/column_min_max_vals_dsb.csv'):
	lines = open(file_path, 'r').readlines()

	table_size_list = []
	table2id = {}
	col2id = {}
	global_col2id = {}
	col2minmax = {}
	join_col_list = []

	join_cols = ['t.id', 'mc.movie_id', 'ci.movie_id', 'mi.movie_id', 'mi_idx.movie_id', 'mk.movie_id']
	col_count = 0
	global_count = 0

	for line in lines[1:]:
		parts = line.strip().split(',')

		col = parts[0]
		table = col.split('.')[0]
		min_v = int(parts[1])
		max_v = int(parts[2])
		card = int(parts[3])

		col2minmax[col] = [min_v, max_v + 1]

		if table not in table2id:
			col_count = 0
			table2id[table] = len(table2id)
			table_size_list.append(card)

		col2id[col] = col_count
		global_col2id[col] = global_count
		if col in join_cols:
			join_col_list.append(col_count)

		col_count += 1
		global_count += 1

	return table2id, table_size_list, col2minmax, col2id, join_col_list, global_col2id

def normalize_val(val, min_v, max_v):
	return float(val - min_v) / (max_v - min_v)

def read_query_file(table2id, col2minmax, col2id, global_col2id, num_q=30000, test_size=0, file_path='./queries/dsb.csv'):
	lines = open(file_path, 'r').readlines()
	training_queries = []
	training_cards = []

	test_queries = []
	test_cards = []
	random.seed(42)

	single_table_query_in_train_prob = 0.0

	template2choice = {}
	q_count = 0
	single_table_q_count = 0
	max_num_preds = 0
	all_tables = []
	all_joins = []

	for line in lines:
		table2predicates = {} # (table_id, list_of_predicates)
		table2qreps = {}
		parts = line.strip().split('#')
		tables = parts[0].split(',')
		joins = parts[1].split(',')
		predicates = parts[2].split(',')
		card = int(parts[3])

		# ### test single tables
		# if len(tables) > 1:
		# 	continue
		# ########

		for join in joins:
			if join not in all_joins:
				all_joins.append(join)

		is_valid = True
		for t in tables:
			table_id = table2id[t.split(' ')[1]]
			table2predicates[table_id] = []
			table2qreps[table_id] = []
			if t not in all_tables:
				all_tables.append(t)

		if len(predicates) >= 3:
			# having predicate
			num_p = int(len(predicates)/3)

			if num_p > max_num_preds:
				max_num_preds = num_p

			for p_id in range(num_p):
				col = predicates[p_id * 3]
				op = predicates[p_id * 3 + 1]
				val = int(predicates[p_id * 3 + 2])

				min_val = col2minmax[col][0]
				max_val = col2minmax[col][1]

				if val < min_val or val > max_val:
					is_valid = False
					break

				t = col.split('.')[0]
				table_id = table2id[t]

				col_id = col2id[col]
				global_col_id = global_col2id[col]
				if op != '=':
					if op == '<':
						op = '<='
						nol_val = normalize_val(val,  min_val, max_val)
						table2predicates[table_id].append([col_id, [[0., nol_val]]])

					elif op == '>':
						op = '>='
						nol_val = normalize_val(val + 1, min_val, max_val)
						table2predicates[table_id].append([col_id, [[nol_val, 1.]]])

					elif op == '<=':
						nol_val = normalize_val(val + 1, min_val, max_val)
						table2predicates[table_id].append([col_id, [[0., nol_val]]])

					elif op == '>=':
						nol_val = normalize_val(val, min_val, max_val)
						table2predicates[table_id].append([col_id, [[nol_val, 1.]]])

					table2qreps[table_id].append([col_id, op, [nol_val]])

				else: # op='='
					nol_val1 = normalize_val(val, min_val, max_val)
					nol_val2 = normalize_val(val + 1, min_val, max_val)

					table2predicates[table_id].append([col_id, [[nol_val1, nol_val2]]])

					table2qreps[table_id].append([col_id, '=', [nol_val1]])

		if is_valid:
			query_info = (table2predicates, table2qreps, sorted(tables), joins)

			if len(tables) == 5:
				training_queries.append(query_info)
				training_cards.append(card)
				q_count += 1

			else:
				if tuple(sorted(tables)) not in template2choice:
					ran_number = random.uniform(0, 1)
					if ran_number < 0.2:
						template2choice[tuple(sorted(tables))] = True
					else:
						template2choice[tuple(sorted(tables))] = False

				if template2choice[tuple(sorted(tables))]:
					training_queries.append(query_info)
					training_cards.append(card)
					# test_queries.append(table2predicates)
					# test_cards.append(card)
					q_count += 1
				else:
					test_queries.append(query_info)
					test_cards.append(card)
					q_count += 1

			if q_count > num_q + test_size:
				break
			
	print('number of training qs')
	print(len(training_queries))
	template2queries = {}
	template2cards = {}

	for query_info, card in zip(training_queries, training_cards):
		table_list = query_info[-2]
		table_list = tuple(table_list)

		if table_list not in template2queries:
			template2queries[table_list] = [query_info]
			template2cards[table_list] = [card]
		else:
			template2queries[table_list].append(query_info)
			template2cards[table_list].append(card)

	test_template2queries = {}
	test_template2cards = {}
	
	print(len(test_queries))

	for query_info, card in zip(test_queries, test_cards):
		table_list = query_info[-2]
		table_list = tuple(table_list)

		if table_list not in test_template2queries:
			test_template2queries[table_list] = [query_info]
			test_template2cards[table_list] = [card]
		else:
			test_template2queries[table_list].append(query_info)
			test_template2cards[table_list].append(card)

	### shuffle training sets
	for table_list in template2queries:
		zipped = list(zip(template2queries[table_list], template2cards[table_list]))
		random.shuffle(zipped)

		new_qs, new_cards = zip(*zipped)	

		template2queries[table_list] = list(new_qs)
		template2cards[table_list] = list(new_cards)

	return template2queries, template2cards, test_template2queries, test_template2cards, max_num_preds, all_tables, all_joins






import copy
import hashlib
import torch
import torch.nn as nn
# from ..utilits import *
import numpy as np
import random
import csv
import os
from CEB_utlities.query_representation.query import *
from CEB_utlities.query_representation.sql_parser import *
from CEB_utlities.query_representation.generate_bitmap import *
import pickle

import re
from collections import defaultdict
from datetime import date, datetime


def default_serializer(obj):
	"""JSON serializer for objects not serializable by default json code"""
	if isinstance(obj, (date, datetime)):
		return obj.isoformat()
	elif isinstance(obj, np.ndarray):
		return obj.tolist()
	raise TypeError(f"Type {type(obj)} not serializable")


TEXT_COLS = ['t.title', 'mi.info', 'ci.note', 'n.name',
             'n.name_pcode_nf', 'n.name_pcode_cf', 'n.surname_pcode',
             'k.keyword', 'cn.name', 'pi.info']

IN_TEXT_COLS = ['t.title', 'mi.info', 'ci.note',
                'n.name_pcode_nf', 'n.name_pcode_cf', 'n.surname_pcode',
                'k.keyword', 'cn.name']

CATEGORICAL_COLS = ['kt.kind', 'rt.role', 'n.gender', 'cn.country_code', 'ct.kind']
CATEGORICAL_COLS_IDS = [1, 1, 4, 2, 1]
CATEGORICAL_COLS_VALS = {
	'kt.kind': ['episode', 'movie', 'tv mini series', 'tv movie', 'tv series', 'video game', 'video movie'],
	'rt.role': ['actor', 'actress', 'cinematographer', 'composer', 'costume designer', 'director', 'editor', 'guest',
	            'miscellaneous crew', 'producer', 'production designer', 'writer'],
	'n.gender': ['NULL', 'f', 'm'],
	'cn.country_code': ['NULL', '[ad]', '[ae]', '[af]', '[ag]', '[ai]', '[al]', '[am]', '[an]', '[ao]', '[ar]', '[as]',
	                    '[at]', '[au]', '[aw]', '[az]', '[ba]', '[bb]', '[bd]', '[be]', '[bf]', '[bg]', '[bh]', '[bi]',
	                    '[bj]', '[bl]', '[bm]', '[bn]', '[bo]', '[br]', '[bs]', '[bt]', '[bw]', '[by]', '[bz]', '[ca]',
	                    '[cd]', '[cg]', '[ch]', '[ci]', '[cl]', '[cm]', '[cn]', '[co]', '[cr]', '[cshh]', '[cu]',
	                    '[cv]', '[cy]', '[cz]', '[ddde]', '[de]', '[dk]', '[dm]', '[do]', '[dz]', '[ec]', '[ee]',
	                    '[eg]', '[er]', '[es]', '[et]', '[fi]', '[fj]', '[fo]', '[fr]', '[ga]', '[gb]', '[gd]', '[ge]',
	                    '[gf]', '[gg]', '[gh]', '[gi]', '[gl]', '[gn]', '[gp]', '[gr]', '[gt]', '[gu]', '[gw]', '[gy]',
	                    '[hk]', '[hn]', '[hr]', '[ht]', '[hu]', '[id]', '[ie]', '[il]', '[im]', '[in]', '[iq]', '[ir]',
	                    '[is]', '[it]', '[je]', '[jm]', '[jo]', '[jp]', '[ke]', '[kg]', '[kh]', '[ki]', '[kn]', '[kp]',
	                    '[kr]', '[kw]', '[ky]', '[kz]', '[la]', '[lb]', '[lc]', '[li]', '[lk]', '[lr]', '[ls]', '[lt]',
	                    '[lu]', '[lv]', '[ly]', '[ma]', '[mc]', '[md]', '[me]', '[mg]', '[mh]', '[mk]', '[ml]', '[mm]',
	                    '[mn]', '[mo]', '[mq]', '[mr]', '[mt]', '[mu]', '[mv]', '[mx]', '[my]', '[mz]', '[na]', '[nc]',
	                    '[ne]', '[ng]', '[ni]', '[nl]', '[no]', '[np]', '[nr]', '[nz]', '[om]', '[pa]', '[pe]', '[pf]',
	                    '[pg]', '[ph]', '[pk]', '[pl]', '[pm]', '[pr]', '[ps]', '[pt]', '[py]', '[qa]', '[ro]', '[rs]',
	                    '[ru]', '[rw]', '[sa]', '[sd]', '[se]', '[sg]', '[si]', '[sj]', '[sk]', '[sl]', '[sm]', '[sn]',
	                    '[so]', '[sr]', '[suhh]', '[sv]', '[sy]', '[sz]', '[td]', '[tf]', '[tg]', '[th]', '[tj]',
	                    '[tk]', '[tl]', '[tm]', '[tn]', '[to]', '[tr]', '[tt]', '[tv]', '[tw]', '[tz]', '[ua]', '[ug]',
	                    '[um]', '[us]', '[uy]', '[uz]', '[va]', '[ve]', '[vg]', '[vi]', '[vn]', '[xyu]', '[ye]',
	                    '[yucs]', '[za]', '[zm]', '[zw]'],
	'ct.kind': ['distributors', 'miscellaneous companies', 'production companies', 'special effects companies']}

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

JOIN_MAP_IMDB = {}
JOIN_MAP_IMDB["title.id"] = "movie_id"  # pk
JOIN_MAP_IMDB["movie_info.movie_id"] = "movie_id"
JOIN_MAP_IMDB["cast_info.movie_id"] = "movie_id"
JOIN_MAP_IMDB["movie_keyword.movie_id"] = "movie_id"
JOIN_MAP_IMDB["movie_companies.movie_id"] = "movie_id"
# JOIN_MAP_IMDB["movie_link.movie_id"] = "movie_id"
JOIN_MAP_IMDB["movie_info_idx.movie_id"] = "movie_id"
# JOIN_MAP_IMDB["movie_link.linked_movie_id"] = "movie_id"
JOIN_MAP_IMDB["aka_title.movie_id"] = "movie_id"
JOIN_MAP_IMDB["complete_cast.movie_id"] = "movie_id"

JOIN_MAP_IMDB["movie_keyword.keyword_id"] = "keyword"
JOIN_MAP_IMDB["keyword.id"] = "keyword"  # pk

JOIN_MAP_IMDB["name.id"] = "person_id"  # pk
JOIN_MAP_IMDB["person_info.person_id"] = "person_id"
JOIN_MAP_IMDB["cast_info.person_id"] = "person_id"
JOIN_MAP_IMDB["aka_name.person_id"] = "person_id"

JOIN_MAP_IMDB["title.kind_id"] = "kind_id"
JOIN_MAP_IMDB["aka_title.kind_id"] = "kind_id"
JOIN_MAP_IMDB["kind_type.id"] = "kind_id"  # pk

JOIN_MAP_IMDB["cast_info.role_id"] = "role_id"
JOIN_MAP_IMDB["role_type.id"] = "role_id"  # pk

JOIN_MAP_IMDB["cast_info.person_role_id"] = "char_id"
JOIN_MAP_IMDB["char_name.id"] = "char_id"  # pk

JOIN_MAP_IMDB["movie_info.info_type_id"] = "info_id"
JOIN_MAP_IMDB["movie_info_idx.info_type_id"] = "info_id"
JOIN_MAP_IMDB["person_info.info_type_id"] = "info_id"
JOIN_MAP_IMDB["info_type.id"] = "info_id"  # pk

JOIN_MAP_IMDB["movie_companies.company_type_id"] = "company_type"
JOIN_MAP_IMDB["company_type.id"] = "company_type"  # pk

JOIN_MAP_IMDB["movie_companies.company_id"] = "company_id"
JOIN_MAP_IMDB["company_name.id"] = "company_id"  # pk

# JOIN_MAP_IMDB["movie_link.link_type_id"] = "link_id"
# JOIN_MAP_IMDB["link_type.id"] = "link_id" # pk

JOIN_MAP_IMDB["complete_cast.status_id"] = "subject"
JOIN_MAP_IMDB["complete_cast.subject_id"] = "subject"
JOIN_MAP_IMDB["comp_cast_type.id"] = "subject"  # pk

JOIN_MAP_IMDB_UPPER_BOUND = {
	'movie_id': 2528312,
	'keyword': 134170,
	'person_id': 4167491,
	'kind_id': 7,
	'role_id': 12,
	'info_id': 113,
	'company_type': 4,
	'company_id': 234997,
}

FILTER_COLS = {'title': ['t.production_year', 't.title'],
               'kind_type': ['kt.kind'],
               'keyword': ['k.keyword'],
               'movie_info': ['mi.info'],
               'movie_info_idx': ['mii.info'],
               'info_type': ['it.id'],
               'cast_info': ['ci.note'],
               'role_type': ['rt.role'],
               'name': ['n.gender', 'n.name', 'n.name_pcode_nf', 'n.name_pcode_cf', 'n.surname_pcode'],
               'movie_companies': [],
               'movie_keyword': [],
               'company_name': ['cn.country_code', 'cn.name'],
               'company_type': ['ct.kind'],
               'aka_name': [],
               'person_info': ['pi.info']}

LIKE_COLS = {'title': ['t.title'],
             'kind_type': ['kt.kind'],
             'movie_info': ['mi.info'],
             'name': ['n.name', 'n.name_pcode_nf', 'n.name_pcode_cf', 'n.surname_pcode'],
             'company_name': ['cn.name'],
             'person_info': ['pi.info']}

# LIKE_COLS = {'t.title', 'kt.kind', 'mi.info', 'n.name',
# 			 'n.name_pcode_nf', 'n.name_pcode_cf', 'n.surname_pcode', 'cn.name', 'pi.info'}

PLAIN_FILTER_COLS = []
for t in FILTER_COLS:
	PLAIN_FILTER_COLS.extend(FILTER_COLS[t])

TABLE_SIZES = {'title': 2528312,
               'kind_type': 7,
               'keyword': 134170,
               'movie_info': 14835720,
               'movie_info_idx': 1380035,
               #    'movie_link':29997,
               'info_type': 113,
               'cast_info': 36244344,
               'role_type': 12,
               'name': 4167491,
               'movie_companies': 2609129,
               'movie_keyword': 4523930,
               'company_name': 234997,
               'company_type': 4,
               'aka_name': 901343,
               'person_info': 2963664}

PURE_LIKE_COLS = list(set(TEXT_COLS) - set(IN_TEXT_COLS))
IN_BUCKETS = 100

single_table_query_in_train_prob = 0.1
sub_templates_in_training_ratio = 0.1


class UnionFind:
	def __init__(self):
		self.parent = {}

	def find(self, item):
		if item not in self.parent:
			self.parent[item] = item
		if self.parent[item] != item:
			self.parent[item] = self.find(self.parent[item])
		return self.parent[item]

	def union(self, item1, item2):
		root1 = self.find(item1)
		root2 = self.find(item2)
		if root1 != root2:
			self.parent[root2] = root1


def find_connected_clusters(list_of_lists):
	uf = UnionFind()

	for sublist in list_of_lists:
		for i in range(len(sublist) - 1):
			uf.union(sublist[i], sublist[i + 1])

	clusters = defaultdict(list)
	for item in uf.parent:
		root = uf.find(item)
		clusters[root].append(item)

	return list(clusters.values())


def get_table_id(table):
	keys = list(FILTER_COLS.keys())
	return keys.index(table)


def get_categorical_cols_info(file_directory='/Users/bytedance/Documents/data/join-order-benchmark-master/imdb/'):
	print('reading')
	categorical_cols_all_values = {}
	for col_name, col_id in zip(CATEGORICAL_COLS, CATEGORICAL_COLS_IDS):
		distinct_values = set()
		alias_table = col_name.split('.')[0]
		table = reverse_alias[alias_table]

		with open(file_directory + '{}.csv'.format(table), 'r', newline='') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='"', escapechar='\\')
			for row in reader:
				if len(row) >= 1:
					distinct_values.add(row[col_id])

		sorted_values = sorted(distinct_values, key=lambda x: (x != '', x))
		categorical_cols_all_values[col_name] = sorted_values

	return categorical_cols_all_values


def deterministic_hash(string):
	return int(hashlib.sha1(str(string).encode("utf-8")).hexdigest(), 16)


def _handle_like(val):
	like_val = val[0]
	pfeats = np.zeros(IN_BUCKETS + 2)
	regex_val = like_val.replace("%", "")
	pred_idx = deterministic_hash(regex_val) % IN_BUCKETS
	pfeats[pred_idx] = 1.00

	for v in regex_val:
		pred_idx = deterministic_hash(str(v)) % IN_BUCKETS
		pfeats[pred_idx] = 1.00

	for i, v in enumerate(regex_val):
		if i != len(regex_val) - 1:
			pred_idx = deterministic_hash(v + regex_val[i + 1]) % IN_BUCKETS
			pfeats[pred_idx] = 1.00

	for i, v in enumerate(regex_val):
		if i < len(regex_val) - 2:
			pred_idx = deterministic_hash(v + regex_val[i + 1] + \
			                              regex_val[i + 2]) % IN_BUCKETS
			pfeats[pred_idx] = 1.00

	pfeats[IN_BUCKETS] = len(regex_val)

	# regex has num or not feature
	if bool(re.search(r'\d', regex_val)):
		pfeats[IN_BUCKETS + 1] = 1

	return pfeats


def drop_trailing_number(s):
	return re.sub(r'\d+$', '', s)


def get_table_info(file_path='./queries/column_min_max_vals_imdb.csv'):
	lines = open(file_path, 'r').readlines()

	table_join_keys = {}
	table_text_cols = {}
	table_normal_cols = {}
	col_type = {}
	col2minmax = {}
	table_dim_list = []
	table_like_dim_list = []
	table_list = []

	### get min/max for cols
	for line in lines[1:]:
		parts = line.strip().split(',')

		col = parts[0]
		min_v = int(parts[1])
		max_v = int(parts[2])

		col2minmax[col] = [min_v, max_v + 1]
	#########

	### build table info
	for table in FILTER_COLS:
		# build filter cols info
		table_cols = FILTER_COLS[table]
		table_dim_list.append(len(table_cols))

		if table in LIKE_COLS:
			table_like_dim_list.append(len(LIKE_COLS[table]))
		else:
			table_like_dim_list.append(0)

		table_list.append(table)
		table_text_cols[table] = []
		table_normal_cols[table] = []
		for col in table_cols:
			full_col_name = col
			if full_col_name in TEXT_COLS:
				table_text_cols[table].append(full_col_name)
				col_type[full_col_name] = 'text'
				if full_col_name in IN_TEXT_COLS:
					col2minmax[col] = [0, IN_BUCKETS + 1]
			else:
				table_normal_cols[table].append(full_col_name)
				if full_col_name in CATEGORICAL_COLS:
					col_type[full_col_name] = 'categorical'
					num_dist_vals = len(CATEGORICAL_COLS_VALS[full_col_name])
					col2minmax[col] = [0, num_dist_vals]
				else:
					col_type[full_col_name] = 'number'

	table_key_groups = {}
	### build table join keys info
	for col_name in JOIN_MAP_IMDB:
		table = col_name.split('.')[0]
		# table = reverse_alias[alias_table]
		key_group = JOIN_MAP_IMDB[col_name]


		if key_group not in table_key_groups:
			table_key_groups[key_group] = [table]
		else:
			if table not in table_key_groups[key_group]:
				table_key_groups[key_group].append(table)

		if table not in table_join_keys:
			table_join_keys[table] = [key_group]
		else:
			if key_group not in table_join_keys[table]:
				table_join_keys[table].append(key_group)

	table_sizes = TABLE_SIZES

	return (table_list, table_dim_list, table_like_dim_list, table_sizes, table_key_groups,
	        table_join_keys, table_text_cols, table_normal_cols, col_type, col2minmax)


def normalize_val(val, min_v, max_v, col_info=None):
	res = float(val - min_v) / (max_v - min_v)

	if res > 1:
		print('error')
		print(val)
		print(min_v)
		print(max_v)
		if col_info is not None:
			print(col_info)
	if res < 0:
		return 0.
	elif res > 1:
		return 1.
	else:
		return res


def parse_sql(sql, col2minmax):
	tables, aliases = extract_from_clause(sql)
	joins, preds, pred_cols, pred_types, pred_vals = parse_where(sql, aliases)
	query_info = process_a_query(tables, aliases, joins, preds, pred_cols, pred_types, pred_vals, col2minmax)

	return query_info


def process_a_query(tables, aliases, joins, preds, pred_cols, pred_types, pred_vals, col2minmax):
	table2normal_predicates = {}
	table2text_predicates = {}
	table2qreps = {}

	for alias_t in aliases:
		# table_id = get_table_id(full_t)
		table2normal_predicates[alias_t] = []
		table2text_predicates[alias_t] = []
		table2qreps[alias_t] = []

	queried_tables = set([])
	### parse pred info
	for i in range(len(preds)):
		# per table
		vals_visited = []
		for j, pred in enumerate(preds[i]):
			col = pred_cols[i][j]
			op = pred_types[i][j]
			val = pred_vals[i][j]

			alias_t = col.split('.')[0]
			queried_tables.add(alias_t)
			full_t = reverse_alias[drop_trailing_number(alias_t)]
			# table_id = get_table_id(full_t)

			standard_col_name = ALIAS[full_t] + '.' + col.split('.')[1]
			col_id = FILTER_COLS[full_t].index(standard_col_name)

			if standard_col_name not in PURE_LIKE_COLS:
				if val in vals_visited:
					break
				else:
					vals_visited.append(val)

				min_val = col2minmax[standard_col_name][0]
				max_val = col2minmax[standard_col_name][1]

				if op == 'lt':
					## handles something like year<2010 and year<2015
					if len(vals_visited) == 1:
						if val[0] is None:
							table2normal_predicates[alias_t].append(
								[col_id, [[0., normalize_val(val[1], min_val, max_val)]]])
							table2qreps[alias_t].append(
								[col_id, op, [0., normalize_val(val[1], min_val, max_val)]])
						elif val[1] is None:
							table2normal_predicates[alias_t].append(
								[col_id, [[normalize_val(val[0], min_val, max_val), 1.]]])
							table2qreps[alias_t].append(
								[col_id, op, [normalize_val(val[0], min_val, max_val), 1.]])
						else:
							table2normal_predicates[alias_t].append(
								[col_id,
								 [[normalize_val(val[0], min_val, max_val), normalize_val(val[1], min_val, max_val)]]])
							table2qreps[alias_t].append(
								[col_id, op, [normalize_val(val[0], min_val, max_val),
								              normalize_val(val[1], min_val, max_val)]])
					else:
						### len(vals_visited) >= 2
						for p in table2normal_predicates[alias_t]:
							if p[0] == col_id:
								if val[0] is None:
									p[1][0][1] = normalize_val(val[1], min_val, max_val)
								# table2normal_predicates[alias_t].append([col_id, [[0., normalize_val(val[1], min_val, max_val)]]])
								elif val[1] is None:
									p[1][0][0] = normalize_val(val[0], min_val, max_val)
								# table2normal_predicates[alias_t].append([col_id, [[normalize_val(val[0], min_val, max_val), 1.]]])
								else:
									p[1][0] = [normalize_val(val[0], min_val, max_val),
									           normalize_val(val[1], min_val, max_val)]
								break
						for p in table2qreps[alias_t]:
							if p[0] == col_id:
								if val[0] is None:
									p[2][1] = normalize_val(val[1], min_val, max_val)
								# table2normal_predicates[alias_t].append([col_id, [[0., normalize_val(val[1], min_val, max_val)]]])
								elif val[1] is None:
									p[2][0] = normalize_val(val[0], min_val, max_val)
								# table2normal_predicates[alias_t].append([col_id, [[normalize_val(val[0], min_val, max_val), 1.]]])
								else:
									p[2] = [normalize_val(val[0], min_val, max_val),
									        normalize_val(val[1], min_val, max_val)]
								break

				elif op == 'eq':
					if isinstance(val, dict):
						val = int(val['literal'])
					normal_val1 = normalize_val(val, min_val, max_val)
					normal_val2 = normalize_val(val + 1, min_val, max_val)
					table2normal_predicates[alias_t].append([col_id, [[normal_val1, normal_val2]]])

					table2qreps[alias_t].append([col_id, op, normal_val1])

				elif op == 'in':
					normal_val_list = []
					### hack for OR n.gender
					if full_t == 'name' and val == ['NULL']:
						### has been processed before
						continue

					if pred.strip(" ").endswith('OR n.gender IS NULL)'):
						val.append('NULL')

					if standard_col_name in CATEGORICAL_COLS_VALS:
						qrep_val_list = np.zeros(len(CATEGORICAL_COLS_VALS[standard_col_name]))
						for item_val in val:
							idx_val = CATEGORICAL_COLS_VALS[standard_col_name].index(item_val)
							normal_val1 = normalize_val(idx_val, min_val, max_val)
							normal_val2 = normalize_val(idx_val + 1, min_val, max_val)
							normal_val_list.append([normal_val1, normal_val2])

							qrep_val_list[idx_val] = 1.

					elif standard_col_name in IN_TEXT_COLS:
						# the case of text cols with feature hash
						bucket_list = []
						qrep_val_list = np.zeros(IN_BUCKETS)
						for item_val in val:
							bucket_idx = deterministic_hash(str(item_val)) % IN_BUCKETS
							qrep_val_list[bucket_idx] = 1.
							if bucket_idx not in bucket_list:
								bucket_list.append(bucket_idx)
								normal_val1 = normalize_val(bucket_idx, min_val, max_val)
								normal_val2 = normalize_val(bucket_idx + 1, min_val, max_val)
								normal_val_list.append([normal_val1, normal_val2])
					else:
						# the case of 'it.id'
						qrep_val_list = np.zeros(113)
						for item_val in val:
							item_val = int(item_val)
							qrep_val_list[item_val - 1] = 1.
							normal_val1 = normalize_val(item_val, min_val, max_val)
							normal_val2 = normalize_val(item_val + 1, min_val, max_val)
							normal_val_list.append([normal_val1, normal_val2])

					table2normal_predicates[alias_t].append([col_id, normal_val_list])
					table2qreps[alias_t].append([col_id, op, qrep_val_list])

				elif op == 'like':
					pfeats = _handle_like(val)
					like_col_id = LIKE_COLS[full_t].index(standard_col_name)
					col_vector = np.zeros(len(LIKE_COLS[full_t]))
					col_vector[like_col_id] = 1.
					table2text_predicates[alias_t].append(np.concatenate((col_vector, pfeats)))
					table2qreps[alias_t].append([col_id, op, pfeats])
			else:
				# text cols, for LIKE
				pfeats = _handle_like(val)
				like_col_id = LIKE_COLS[full_t].index(standard_col_name)
				col_vector = np.zeros(len(LIKE_COLS[full_t]))
				col_vector[like_col_id] = 1.
				table2text_predicates[alias_t].append(np.concatenate((col_vector, pfeats)))
				table2qreps[alias_t].append([col_id, op, pfeats])

	### parse join info
	key_groups = {}
	for join in joins:
		parts = join.strip().split(' = ')

		alias_table1 = parts[0].split('.')[0]
		alias_table2 = parts[1].split('.')[0]

		col1 = parts[0].split('.')[1]
		full_key_name1 = reverse_alias[drop_trailing_number(alias_table1)] + '.' + col1
		key_group = JOIN_MAP_IMDB[full_key_name1]

		if key_group not in key_groups:
			key_groups[key_group] = [[alias_table1, alias_table2]]
		else:
			key_groups[key_group].append([alias_table1, alias_table2])

	# merge same groups
	for key_group in key_groups:
		table_list = key_groups[key_group]
		key_groups[key_group] = find_connected_clusters(table_list)

	query_info = [table2normal_predicates, table2text_predicates, table2qreps, key_groups, sorted(aliases)]

	return query_info

def generate_key_group_mask(table_list, key_groups):
	per_t_mask = []
	per_t_group_mask = []

	if len(list(table_list)) > 1:
		for key_group in key_groups:
			key_group_mask = []
			key_t_group_mask = []
			group_t_list = key_groups[key_group]
			for ts in group_t_list:
				intersect_ts = list(set(ts) & set(list(table_list)))
				if len(intersect_ts) >= 2:
					ts_mask = [1 if t in list(table_list) else 0 for t in ts]
					ts_indicator = 1
				else:
					ts_mask = np.zeros(len(ts))
					ts_indicator = 0
				key_group_mask.append(ts_mask)
				key_t_group_mask.append(ts_indicator)
			per_t_mask.append(key_group_mask)
			per_t_group_mask.append(key_t_group_mask)

	else:
		is_found = False
		for key_group in key_groups:
			key_group_mask = []
			key_t_group_mask = []
			group_t_list = key_groups[key_group]
			for ts in group_t_list:
				ts_mask = np.zeros(len(ts))
				ts_indicator = 0

				if (table_list[0] in ts) and (not is_found):
					ts_mask[ts.index(table_list[0])] = 1
					ts_indicator = 1
					is_found = True

				key_group_mask.append(ts_mask)
				key_t_group_mask.append(ts_indicator)
			per_t_mask.append(key_group_mask)
			per_t_group_mask.append(key_t_group_mask)

	return per_t_mask, per_t_group_mask

def read_query_file_batched(col2minmax, num_q=10000, test_size=1000, directory_list=None,
                            saved_ditectory="/home/processed_workloads/imdb/"):
	## num_q: max number of queries per JOB template

	templates_name = []
	for fullname in directory_list:
		templates_name.append(fullname.split("/")[-2])

	templates_name = ",".join(templates_name)
	
	workload_file_path = "{}grasp{}-{}-".format(saved_ditectory, templates_name, sub_templates_in_training_ratio)

	if os.path.exists(workload_file_path + 'template2queries.pkl'):
		print("load template2queries")
		with open(workload_file_path + 'template2queries.pkl', "rb") as pickle_file:
			template2queries = pickle.load(pickle_file)

		print("load template2cards")
		with open(workload_file_path + 'template2cards.pkl', "rb") as pickle_file:
			template2cards = pickle.load(pickle_file)

		print("load template2pgcards")
		with open(workload_file_path + 'template2pgcards.pkl', "rb") as pickle_file:
			template2pgcards = pickle.load(pickle_file)

		print("load test_template2queries")
		with open(workload_file_path + 'test_template2queries.pkl', "rb") as pickle_file:
			test_template2queries = pickle.load(pickle_file)

		print("load test_template2cards")
		with open(workload_file_path + 'test_template2cards.pkl', "rb") as pickle_file:
			test_template2cards = pickle.load(pickle_file)

		print("load test_template2pgcards")
		with open(workload_file_path + 'test_template2pgcards.pkl', "rb") as pickle_file:
			test_template2pgcards = pickle.load(pickle_file)

		print("load colid2featlen_per_table ")
		with open(workload_file_path + 'colid2featlen_per_table.pkl', "rb") as pickle_file:
			colid2featlen_per_table = pickle.load(pickle_file)

		return template2queries, template2cards, template2pgcards, \
		       test_template2queries, test_template2cards, test_template2pgcards, colid2featlen_per_table

	training_queries = []
	training_cards = []
	pg_training_cards = []

	test_queries = []
	test_cards = []
	pg_test_cards = []

	saved_predicates = {}

	colid2featlen_per_table = {}
	for t_name in ALIAS:
		colid2featlen_per_table[t_name] = {}

	tablelist2choice = {}
	random.seed(42)

	for directory in directory_list:
		print("processing {}".format(directory))
		files = os.listdir(directory)
		# Filter out files that do not have a .csv extension
		files = [file for file in files if file.endswith('.pkl')]

		for qid, file in enumerate(files):
			file_path = os.path.join(directory, file)
			qrep = load_qrep(file_path)

			table2normal_predicates = {}
			table2text_predicates = {}
			table2qreps = {}

			tables, aliases = get_tables(qrep)

			global_origial_alias_to_aliases = {}
			for alias in aliases:
				original_alias = drop_trailing_number(alias)
				if original_alias not in global_origial_alias_to_aliases:
					global_origial_alias_to_aliases[original_alias] = [alias]
				else:
					global_origial_alias_to_aliases[original_alias].append(alias)

			joins = get_joins(qrep)
			preds, pred_cols, pred_types, pred_vals = get_predicates(qrep)

			trues = get_true_cardinalities(qrep)
			ests = get_postgres_cardinalities(qrep)

			card = 0.
			pg_card = 0

			# get the true card
			for k, v in trues.items():
				if len(k) == len(tables):
					card = trues[k]
					pg_card = ests[k]

			for alias_t in aliases:
				# table_id = get_table_id(full_t)
				table2normal_predicates[alias_t] = []
				table2text_predicates[alias_t] = []
				table2qreps[alias_t] = []

			queried_tables = set([])
			### parse pred info
			for i in range(len(preds)):
				# per table
				vals_visited = []
				for j, pred in enumerate(preds[i]):
					col = pred_cols[i][j]
					op = pred_types[i][j]
					val = pred_vals[i][j]

					alias_t = col.split('.')[0]
					queried_tables.add(alias_t)
					full_t = reverse_alias[drop_trailing_number(alias_t)]
					# table_id = get_table_id(full_t)

					standard_col_name = ALIAS[full_t] + '.' + col.split('.')[1]
					col_id = FILTER_COLS[full_t].index(standard_col_name)

					if standard_col_name not in PURE_LIKE_COLS:
						if val in vals_visited:
							break
						else:
							vals_visited.append(val)

						min_val = col2minmax[standard_col_name][0]
						max_val = col2minmax[standard_col_name][1]

						if op == 'lt':
							## handles something like year<2010 and year<2015
							if len(vals_visited) == 1:
								if val[0] is None:
									table2normal_predicates[alias_t].append(
										[col_id, [[0., normalize_val(val[1], min_val, max_val, col)]]])
									table2qreps[alias_t].append(
										[col_id, op, [0., normalize_val(val[1], min_val, max_val, col)]])
								elif val[1] is None:
									table2normal_predicates[alias_t].append(
										[col_id, [[normalize_val(val[0], min_val, max_val, col), 1.]]])
									table2qreps[alias_t].append(
										[col_id, op, [normalize_val(val[0], min_val, max_val, col), 1.]])
								else:
									table2normal_predicates[alias_t].append(
										[col_id, [[normalize_val(val[0], min_val, max_val, col),
										           normalize_val(val[1], min_val, max_val, col)]]])
									table2qreps[alias_t].append(
										[col_id, op, [normalize_val(val[0], min_val, max_val, col),
										              normalize_val(val[1], min_val, max_val, col)]])
							else:
								### len(vals_visited) >= 2
								for p in table2normal_predicates[alias_t]:
									if p[0] == col_id:
										if val[0] is None:
											p[1][0][1] = normalize_val(val[1], min_val, max_val, col)
										# table2normal_predicates[alias_t].append([col_id, [[0., normalize_val(val[1], min_val, max_val)]]])
										elif val[1] is None:
											p[1][0][0] = normalize_val(val[0], min_val, max_val, col)
										# table2normal_predicates[alias_t].append([col_id, [[normalize_val(val[0], min_val, max_val), 1.]]])
										else:
											p[1][0] = [normalize_val(val[0], min_val, max_val, col),
											           normalize_val(val[1], min_val, max_val, col)]
										break
								for p in table2qreps[alias_t]:
									if p[0] == col_id:
										if val[0] is None:
											p[2][1] = normalize_val(val[1], min_val, max_val, col)
										# table2normal_predicates[alias_t].append([col_id, [[0., normalize_val(val[1], min_val, max_val)]]])
										elif val[1] is None:
											p[2][0] = normalize_val(val[0], min_val, max_val, col)
										# table2normal_predicates[alias_t].append([col_id, [[normalize_val(val[0], min_val, max_val), 1.]]])
										else:
											p[2] = [normalize_val(val[0], min_val, max_val, col),
											        normalize_val(val[1], min_val, max_val, col)]
										break

							if col_id not in colid2featlen_per_table[full_t]:
								colid2featlen_per_table[full_t][col_id] = 2
							elif 2 > colid2featlen_per_table[full_t][col_id]:
								colid2featlen_per_table[full_t][col_id] = 2

						elif op == 'eq':
							if isinstance(val, dict):
								val = int(val['literal'])
							normal_val1 = normalize_val(val, min_val, max_val)
							normal_val2 = normalize_val(val + 1, min_val, max_val)
							table2normal_predicates[alias_t].append([col_id, [[normal_val1, normal_val2]]])

							table2qreps[alias_t].append([col_id, op, normal_val1])
							if col_id not in colid2featlen_per_table[full_t]:
								colid2featlen_per_table[full_t][col_id] = 1

						elif op == 'in':
							normal_val_list = []

							### hack for OR n.gender
							if full_t == 'name' and val == ['NULL']:
								### has been processed before
								continue

							if pred.strip(" ").endswith('OR n.gender IS NULL)'):
								val.append('NULL')

							if standard_col_name in CATEGORICAL_COLS_VALS:
								qrep_val_list = np.zeros(len(CATEGORICAL_COLS_VALS[standard_col_name]))
								for item_val in val:
									idx_val = CATEGORICAL_COLS_VALS[standard_col_name].index(item_val)
									normal_val1 = normalize_val(idx_val, min_val, max_val)
									normal_val2 = normalize_val(idx_val + 1, min_val, max_val)
									normal_val_list.append([normal_val1, normal_val2])

									qrep_val_list[idx_val] = 1.

								if col_id not in colid2featlen_per_table[full_t]:
									colid2featlen_per_table[full_t][col_id] = len(
										CATEGORICAL_COLS_VALS[standard_col_name])
								elif len(CATEGORICAL_COLS_VALS[standard_col_name]) > colid2featlen_per_table[full_t][
									col_id]:
									colid2featlen_per_table[full_t][col_id] = len(
										CATEGORICAL_COLS_VALS[standard_col_name])

							elif standard_col_name in IN_TEXT_COLS:
								# the case of text cols with feature hash
								bucket_list = []
								qrep_val_list = np.zeros(IN_BUCKETS)
								for item_val in val:
									bucket_idx = deterministic_hash(str(item_val)) % IN_BUCKETS
									qrep_val_list[bucket_idx] = 1.
									if bucket_idx not in bucket_list:
										bucket_list.append(bucket_idx)
										normal_val1 = normalize_val(bucket_idx, min_val, max_val)
										normal_val2 = normalize_val(bucket_idx + 1, min_val, max_val)
										normal_val_list.append([normal_val1, normal_val2])

									if col_id not in colid2featlen_per_table[full_t]:
										colid2featlen_per_table[full_t][col_id] = IN_BUCKETS
									elif IN_BUCKETS > colid2featlen_per_table[full_t][col_id]:
										colid2featlen_per_table[full_t][col_id] = IN_BUCKETS
							else:
								# the case of 'it.id'
								qrep_val_list = np.zeros(113)
								if col_id not in colid2featlen_per_table[full_t]:
									colid2featlen_per_table[full_t][col_id] = 113
								elif 113 > colid2featlen_per_table[full_t][col_id]:
									colid2featlen_per_table[full_t][col_id] = 113
								for item_val in val:
									item_val = int(item_val)
									qrep_val_list[item_val - 1] = 1.
									normal_val1 = normalize_val(item_val, min_val, max_val)
									normal_val2 = normalize_val(item_val + 1, min_val, max_val)
									normal_val_list.append([normal_val1, normal_val2])

							table2normal_predicates[alias_t].append([col_id, normal_val_list])
							table2qreps[alias_t].append([col_id, op, qrep_val_list])

						elif op == 'like':
							pfeats = _handle_like(val)
							like_col_id = LIKE_COLS[full_t].index(standard_col_name)
							col_vector = np.zeros(len(LIKE_COLS[full_t]))
							col_vector[like_col_id] = 1.
							table2text_predicates[alias_t].append(np.concatenate((col_vector, pfeats)))
							table2qreps[alias_t].append([col_id, op, pfeats])
							if col_id not in colid2featlen_per_table[full_t]:
								colid2featlen_per_table[full_t][col_id] = len(pfeats)
							elif len(pfeats) > colid2featlen_per_table[full_t][col_id]:
								colid2featlen_per_table[full_t][col_id] = len(pfeats)
					else:
						# text cols, for LIKE
						pfeats = _handle_like(val)
						like_col_id = LIKE_COLS[full_t].index(standard_col_name)
						col_vector = np.zeros(len(LIKE_COLS[full_t]))
						col_vector[like_col_id] = 1.
						table2text_predicates[alias_t].append(np.concatenate((col_vector, pfeats)))
						table2qreps[alias_t].append([col_id, op, pfeats])
						if col_id not in colid2featlen_per_table[full_t]:
							colid2featlen_per_table[full_t][col_id] = len(pfeats)
						elif len(pfeats) > colid2featlen_per_table[full_t][col_id]:
							colid2featlen_per_table[full_t][col_id] = len(pfeats)

			### parse join info
			key_groups = {}

			for join in joins:
				parts = join.strip().split(' = ')

				alias_table1 = parts[0].split('.')[0]
				alias_table2 = parts[1].split('.')[0]

				col1 = parts[0].split('.')[1]

				full_key_name1 = reverse_alias[drop_trailing_number(alias_table1)] + '.' + col1
				key_group = JOIN_MAP_IMDB[full_key_name1]

				if len(global_origial_alias_to_aliases[drop_trailing_number(alias_table1)]) == 1:
					alias_table1 = drop_trailing_number(alias_table1)

				if len(global_origial_alias_to_aliases[drop_trailing_number(alias_table2)]) == 1:
					alias_table2 = drop_trailing_number(alias_table2)

				if key_group not in key_groups:
					key_groups[key_group] = [[alias_table1, alias_table2]]
				else:
					key_groups[key_group].append([alias_table1, alias_table2])

			key_groups = {key: key_groups[key] for key in sorted(key_groups)}
			# merge same groups
			for key_group in key_groups:
				table_list = key_groups[key_group]
				table_list.sort()
				key_groups[key_group] = find_connected_clusters(table_list)

			old_key_list = list(table2normal_predicates.keys())
			for t in old_key_list:
				if len(global_origial_alias_to_aliases[drop_trailing_number(t)]) == 1:
					table2normal_predicates[drop_trailing_number(t)] = table2normal_predicates.pop(t)
					table2text_predicates[drop_trailing_number(t)] = table2text_predicates.pop(t)
					table2qreps[drop_trailing_number(t)] = table2qreps.pop(t)

			new_aliases = []
			for alias in aliases:
				if len(global_origial_alias_to_aliases[drop_trailing_number(alias)]) == 1:
					new_aliases.append(drop_trailing_number(alias))
				else:
					new_aliases.append(alias)

			table_mask, table_group_mask = generate_key_group_mask(new_aliases, key_groups)
			query_info = [table2normal_predicates, table2text_predicates, table2qreps, key_groups, key_groups,
			              sorted(new_aliases),
			              sorted(new_aliases), table_mask, table_group_mask]

			if qid < num_q:
				if len(tables) >= 2:
					if not tuple(sorted(list(aliases))) in tablelist2choice:
						compare_ratio = sub_templates_in_training_ratio
						r_number = random.uniform(0, 1)
						if r_number < compare_ratio:
							tablelist2choice[tuple(sorted(list(aliases)))] = True  # training
						else:
							tablelist2choice[tuple(sorted(list(aliases)))] = False  # test

					training_queries.append(query_info)
					training_cards.append(card)
					pg_training_cards.append(pg_card)

					### add subqueries as test queries
					for k, v in trues.items():

						origial_alias_to_aliases = {}
						for alias in k:
							original_alias = drop_trailing_number(alias)
							if original_alias not in origial_alias_to_aliases:
								origial_alias_to_aliases[original_alias] = [alias]
							else:
								origial_alias_to_aliases[original_alias].append(alias)

						global_new_k = []
						local_new_k = []
						k_to_newk = {}

						for alias in k:
							if len(global_origial_alias_to_aliases[drop_trailing_number(alias)]) == 1:
								global_new_k.append(drop_trailing_number(alias))
							else:
								global_new_k.append(alias)

						for alias in global_new_k:
							if len(origial_alias_to_aliases[drop_trailing_number(alias)]) == 1:
								local_new_k.append(drop_trailing_number(alias))
								k_to_newk[alias] = drop_trailing_number(alias)
							else:
								local_new_k.append(alias)
								k_to_newk[alias] = alias

						if not tuple(sorted(list(local_new_k))) in tablelist2choice:
							if len(list(local_new_k)) == 1:
								compare_ratio = single_table_query_in_train_prob
							else:
								compare_ratio = sub_templates_in_training_ratio
							r_number = random.uniform(0, 1)
							if r_number < compare_ratio:
								tablelist2choice[tuple(sorted(list(local_new_k)))] = True  # training
							else:
								tablelist2choice[tuple(sorted(list(local_new_k)))] = False  # test


						sub_card = trues[k]
						sub_pg_card = ests[k]

						sub_table2normal_predicates = {}
						sub_table2text_predicates = {}
						sub_table2qreps = {}

						sub_table2normal_predicates_str = {k_to_newk[t]: table2normal_predicates[t] for t in
						                                   global_new_k}
						sub_table2text_predicates_str = {k_to_newk[t]: table2text_predicates[t] for t in global_new_k}

						for alias_t in new_aliases:
							# table_id = get_table_id(full_t)
							sub_table2normal_predicates[alias_t] = []
							sub_table2text_predicates[alias_t] = []
							sub_table2qreps[alias_t] = []

						for t in global_new_k:
							sub_table2normal_predicates[t] = table2normal_predicates[t]
							sub_table2text_predicates[t] = table2text_predicates[t]
							sub_table2qreps[t] = table2qreps[t]

						sub_key_groups = {}
						for key_group in key_groups:
							group_t_list = key_groups[key_group]
							for ts in group_t_list:
								intersect_ts = list(set(ts) & set(list(global_new_k)))

								if len(intersect_ts) >= 2:
									new_intersect_ts = [k_to_newk[t] for t in intersect_ts]
									if key_group not in sub_key_groups:
										sub_key_groups[key_group] = [new_intersect_ts]
									else:
										sub_key_groups[key_group].append(new_intersect_ts)

						table_mask, table_group_mask = generate_key_group_mask(global_new_k, key_groups)
						sub_query_info = [sub_table2normal_predicates, sub_table2text_predicates, sub_table2qreps,
						                  key_groups, sub_key_groups, sorted(new_aliases), sorted(list(local_new_k)),
						                  table_mask, table_group_mask]

						q_json_str = (json.dumps(sub_table2normal_predicates_str, sort_keys=True,
						                         default=default_serializer)
						              + json.dumps(sub_table2text_predicates_str, sort_keys=True,
						                           default=default_serializer))

						t_list_key = tuple(sorted(list(local_new_k)))
						if t_list_key not in saved_predicates:
							saved_predicates[t_list_key] = set([])

						if q_json_str not in saved_predicates[t_list_key]:
							subquery_joins = subplan_to_joins(qrep, k)

							for join in subquery_joins:

								parts = join.strip().split(' = ')
								alias_table1 = parts[0].split('.')[0]
								alias_table2 = parts[1].split('.')[0]

								col1 = parts[0].split('.')[1]

								if len(global_origial_alias_to_aliases[drop_trailing_number(alias_table1)]) == 1:
									alias_table1 = drop_trailing_number(alias_table1)

								if len(global_origial_alias_to_aliases[drop_trailing_number(alias_table2)]) == 1:
									alias_table2 = drop_trailing_number(alias_table2)

							saved_predicates[t_list_key].add(q_json_str)

							if tablelist2choice[tuple(sorted(list(local_new_k)))]:
								training_queries.append(sub_query_info)
								training_cards.append(sub_card)
								pg_training_cards.append(sub_pg_card)
							else:
								test_queries.append(sub_query_info)
								test_cards.append(sub_card)
								pg_test_cards.append(sub_pg_card)
			else:
				pass
	
	template2queries = {}
	template2cards = {}
	template2pgcards = {}
	print(len(training_queries))

	for query_info, card, pg_card in zip(training_queries, training_cards, pg_training_cards):
		table_list = query_info[-3]
		table_list = tuple(table_list)

		if table_list not in template2queries:
			template2queries[table_list] = [query_info]
			template2cards[table_list] = [card]
			template2pgcards[table_list] = [pg_card]
		else:
			template2queries[table_list].append(query_info)
			template2cards[table_list].append(card)
			template2pgcards[table_list].append(pg_card)

	### shuffle training sets
	for table_list in template2queries:
		zipped = list(zip(template2queries[table_list], template2cards[table_list], template2pgcards[table_list]))
		random.shuffle(zipped)

		new_qs, new_cards, new_pgcards = zip(*zipped)

		template2queries[table_list] = list(new_qs)
		template2cards[table_list] = list(new_cards)
		template2pgcards[table_list] = list(new_pgcards)

	test_template2queries = {}
	test_template2cards = {}
	test_template2pgcards = {}

	processed_test_template2queries = {}
	processed_test_template2cards = {}
	processed_test_template2pgcards = {}

	for query_info, card, pg_card in zip(test_queries, test_cards, pg_test_cards):
		table_list = query_info[-3]
		table_list = tuple(table_list)

		if table_list not in test_template2queries:
			test_template2queries[table_list] = [query_info]
			test_template2cards[table_list] = [card]
			test_template2pgcards[table_list] = [pg_card]
		else:
			test_template2queries[table_list].append(query_info)
			test_template2cards[table_list].append(card)
			test_template2pgcards[table_list].append(pg_card)

	### random sample 100 unseen templates for test, each template has at least 10 queries.
	### this is for effcient evaluation, since the results are close to a full test set.
	
	unseen_templates = list(test_template2queries.keys())
	random.shuffle(unseen_templates)
	unseen_templates = unseen_templates[:100]
	for template in unseen_templates:
		if len(test_template2queries[template]) < 10:
			continue

		test_queries = test_template2queries[template]
		test_cards = test_template2cards[template]
		test_pgcards = test_template2pgcards[template]

		combined = list(zip(test_queries, test_cards, test_pgcards))
		random.shuffle(combined)
		test_queries, test_cards, test_pgcards = zip(*combined)

		test_queries = list(test_queries)[:100]
		test_cards = list(test_cards)[:100]
		test_pgcards = list(test_pgcards)[:100]

		processed_test_template2queries[template] = test_queries
		processed_test_template2cards[template] = test_cards
		processed_test_template2pgcards[template] = test_pgcards

	print("start template2queries to file")
	with open(workload_file_path + 'template2queries.pkl', "wb") as pickle_file:
		pickle.dump(template2queries, pickle_file)

	print("start template2cards to file")
	with open(workload_file_path + 'template2cards.pkl', "wb") as pickle_file:
		pickle.dump(template2cards, pickle_file)

	print("start template2pgcards to file")
	with open(workload_file_path + 'template2pgcards.pkl', "wb") as pickle_file:
		pickle.dump(template2pgcards, pickle_file)

	print("start test_template2queries to file")
	with open(workload_file_path + 'test_template2queries.pkl', "wb") as pickle_file:
		pickle.dump(processed_test_template2queries, pickle_file)

	print("start test_template2cards to file")
	with open(workload_file_path + 'test_template2cards.pkl', "wb") as pickle_file:
		pickle.dump(processed_test_template2cards, pickle_file)

	print("start test_template2pgcards to file")
	with open(workload_file_path + 'test_template2pgcards.pkl', "wb") as pickle_file:
		pickle.dump(processed_test_template2pgcards, pickle_file)

	print("start colid2featlen_per_table to file")
	with open(workload_file_path + 'colid2featlen_per_table.pkl', "wb") as pickle_file:
		pickle.dump(colid2featlen_per_table, pickle_file)

	print("finished pickle writing")

	return template2queries, template2cards, template2pgcards, processed_test_template2queries, \
	       processed_test_template2cards, processed_test_template2pgcards, colid2featlen_per_table


def get_table_to_ops(template2queries, test_template2queries):
	t2ops = {}
	for template in template2queries:
		training_queries = template2queries[template]
		for qid, (
		q, q_context, q_reps, parent_key_groups, key_groups_per_q, parent_table_tuple, table_tuple, table_mask,
		table_group_mask) in enumerate(training_queries):
			for t in q_reps:
				for rep_tuple in q_reps[t]:
					op = rep_tuple[1]
					if t not in t2ops:
						t2ops[t] = [op]
					else:
						if op not in t2ops[t]:
							t2ops[t].append(op)

	for template in test_template2queries:
		training_queries = test_template2queries[template]
		for qid, (
		q, q_context, q_reps, parent_key_groups, key_groups_per_q, parent_table_tuple, table_tuple, table_mask,
		table_group_mask) in enumerate(training_queries):
			for t in q_reps:
				for rep_tuple in q_reps[t]:
					op = rep_tuple[1]
					if t not in t2ops:
						t2ops[t] = [op]
					else:
						if op not in t2ops[t]:
							t2ops[t].append(op)
	return t2ops

import copy
import hashlib
import torch.nn as nn
import numpy as np
import random
import csv
import os
import re
from collections import defaultdict
import ujson as json
from datetime import date, datetime

NULL_count = 0
product_count1 = 0
product_count2 = 0
wrong_qids = []

COL_TYPES = {'ap_tax_id': 'string', 'ap_l_name': 'string', 'ap_f_name': 'string', 'ap_ca_id': 'number',
			 'h_s_symb': 'string', 'h_ca_id': 'number', 'wl_c_id': 'number', 'dm_date': 'date',
			 's_symb': 'string', 'th_t_id': 'number', 'ca_id': 'number', 'hs_s_symb': 'string',
			 'hs_ca_id': 'number', 't_id': 'number', 'tt_id': 'string', 'co_name': 'string', 't_s_symb': 'string',
			 't_dts': 'time', 'ct_t_id': 'number', 'lt_s_symb': 'string', 'ch_tt_id': 'string', 'ch_c_tier': 'number',
			 'nx_co_id': 'number', 'c_id': 'number', 'c_tax_id': 'string', 's_issue': 'string', 's_co_id': 'number',
			 'fi_co_id': 'number', 'cx_c_id': 'number', 't_ca_id': 'number', 'in_name': 'string', 'b_id': 'number',
			 't_st_id': 'string', 'hh_t_id': 'number', 'se_t_id': 'number', 'co_id': 'number', 'dm_s_symb': 'string',
			 'cr_ex_id': 'string', 'cr_tt_id': 'string', 'cr_c_tier': 'number', 'cr_from_qty': 'number', 'cr_to_qty': 'number',
			 'sc_name': 'string', 'ca_c_id': 'number', 'cp_co_id': 'number'}


CATEGORICAL_COLS_VALS = {
    "s_issue": [
        "COMMON",
        "PREF_A",
        "PREF_B",
        "PREF_C",
        "PREF_D"
    ],
    "t_st_id": [
        "CMPT",
        "CNCL",
        "PNDG",
        "SBMT"
    ],
    "tt_id": [
        "TLB",
        "TLS",
        "TMB",
        "TMS",
        "TSL"
    ],
    "ch_tt_id": [
        "TLB",
        "TLS",
        "TMB",
        "TMS",
        "TSL"
    ],
    "cr_ex_id": [
        "AMEX",
        "NASDAQ",
        "NYSE",
        "PCX"
    ],
    "cr_tt_id": [
        "TLB",
        "TLS",
        "TMB",
        "TMS",
        "TSL"
    ],
    "sc_name": [
        "Basic Materials",
        "Capital Goods",
        "Conglomerates",
        "Consumer Cyclical",
        "Consumer Non-Cyclical",
        "Energy",
        "Financial",
        "Healthcare",
        "Services",
        "Technology",
        "Transportation",
        "Utilities"
    ]
}

JOIN_MAP_TPCE = {'trade_request.tr_s_symb': 's_symb',
				 'security.s_symb': 's_symb',
				 'daily_market.dm_s_symb': 's_symb',
				 'watch_item.wi_s_symb': 's_symb',
				 'last_trade.lt_s_symb': 's_symb',
				 'trade.t_s_symb': 's_symb',
				 'holding_summary.hs_s_symb': 's_symb',

				 'watch_list.wl_id': 'wl_id', 'watch_item.wi_wl_id': 'wl_id', 
				 
				 'exchange.ex_id': 'ex_id', 'security.s_ex_id': 'ex_id', 
				 
				 'company_competitor.cp_comp_co_id': 'co_id', 'security.s_co_id': 'co_id', 'company.co_id': 'co_id', 

				 'exchange.ex_ad_id': 'ad_id', 'address.ad_id': 'ad_id', 'company.co_ad_id': 'ad_id', 
				 
				 'address.ad_zc_code': 'zc_code', 'zip_code.zc_code': 'zc_code', 
				 
				 'trade.t_tt_id': 'tt_id', 'trade_type.tt_id': 'tt_id', 
				 
				 'news_xref.nx_ni_id': 'ni_id', 'news_item.ni_id': 'ni_id', 
				 
				 'customer_taxrate.cx_tx_id': 'tx_id', 'taxrate.tx_id': 'tx_id', 
				 
				 'status_type.st_id': 'st_id', 'trade.t_st_id': 'st_id', 
				 
				 'company_competitor.cp_in_id': 'in_id', 'industry.in_id': 'in_id', 'company.co_in_id': 'in_id', 
				 
				 'holding_history.hh_h_t_id': 'hh_h_t_id', 
				 
				 'broker.b_id': 'b_id', 'customer_account.ca_b_id': 'b_id', 'trade_request.tr_b_id': 'b_id', 
				 
				 'sector.sc_id': 'sc_id', 'industry.in_sc_id': 'sc_id', 
				 
				 'holding_summary.hs_ca_id': 'ca_id', 'customer_account.ca_id': 'ca_id', 
				 
				 'customer_account.ca_c_id': 'c_id', 'customer.c_id': 'c_id'}



FILTER_COLS = {'account_permission': ['ap_tax_id', 'ap_l_name', 'ap_f_name', 'ap_ca_id'],
			   'holding': ['h_s_symb', 'h_ca_id'],
			   'watch_item': [],
			   'watch_list': ['wl_c_id'],
			   'last_trade': ['lt_s_symb'],
			   'security': ['s_symb', 's_issue', 's_co_id'],
			   'daily_market': ['dm_date', 'dm_s_symb'],
			   'company': ['co_name', 'co_id'],
			   'address': [], 'zip_code': [], 'exchange': [],
			   'trade_history': ['th_t_id'],
			   'customer_account': ['ca_id', 'ca_c_id'],
			   'holding_summary': ['hs_s_symb', 'hs_ca_id'],
			   'trade': ['t_id', 't_s_symb', 't_dts', 't_ca_id', 't_st_id'],
			   'trade_type': ['tt_id'],
			   'cash_transaction': ['ct_t_id'],
			   'charge': ['ch_tt_id', 'ch_c_tier'],
			   'news_xref': ['nx_co_id'],
			   'news_item': [],
			   'customer': ['c_id', 'c_tax_id'],
			   'financial': ['fi_co_id'], 'taxrate': [],
			   'customer_taxrate': ['cx_c_id'], 'status_type': [],
			   'industry': ['in_name'],
			   'broker': ['b_id'],
			   'holding_history': ['hh_t_id'],
			   'settlement': ['se_t_id'],
			   'commission_rate': ['cr_ex_id', 'cr_tt_id', 'cr_c_tier', 'cr_from_qty', 'cr_to_qty'],
			   'trade_request': [],
			   'sector': ['sc_name'],
			   'company_competitor': ['cp_co_id']}


PLAIN_FILTER_COLS = []
for t in FILTER_COLS:
	PLAIN_FILTER_COLS.extend(FILTER_COLS[t])

CATEGORICAL_COLS = ['s_issue', 't_st_id', 'tt_id', 'ch_tt_id',
			 'cr_ex_id', 'cr_tt_id', 'sc_name']

TABLE_SIZES = {
    "account_permission": 14214,
    "holding": 1155773,
    "watch_item": 198048,
    "watch_list": 2000,
    "last_trade": 1370,
    "security": 1370,
    "daily_market": 1787850,
    "company": 1000,
    "address": 3004,
    "zip_code": 14741,
    "exchange": 4,
    "trade_history": 35018534,
    "customer_account": 10000,
    "holding_summary": 99689,
    "trade": 14596142,
    "trade_type": 5,
    "cash_transaction": 13414683,
    "charge": 15,
    "news_xref": 2000,
    "news_item": 2000,
    "customer": 2000,
    "financial": 20000,
    "taxrate": 320,
    "customer_taxrate": 4000,
    "status_type": 5,
    "industry": 102,
    "broker": 20,
    "holding_history": 19409764,
    "settlement": 14580928,
    "commission_rate": 240,
    "trade_request": 6041,
    "sector": 12,
    "company_competitor": 3000
}

IN_BUCKETS = 100
random.seed(42)

def is_int(s):
    return bool(re.match(r"^[+-]?\d+$", s))

def is_float(s):
    return bool(re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$", s))

def is_int_or_float(s):
    return is_int(s) or is_float(s)

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

def convert_dates_to_timestamps(dates):
    """
    Convert a list of date strings to Unix timestamps.

    Parameters:
    dates (list of str): The list of date strings to convert.

    Returns:
    list of int: The list of Unix timestamps.
    """
    timestamps = []
    for date_str in dates:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        timestamp = int(dt.timestamp())
        timestamps.append(timestamp)
    return timestamps

def check_empty(key_groups):
    nump = 0
    for k in key_groups:
        nump += len(key_groups[k])

    if nump == 0:
        return True
    else:
        return False

def has_intersection(list1, list2):
    return any(item in list2 for item in list1)

def convert_datetimes_to_timestamps(datetimes):
    """
    Convert a list of datetime strings to Unix timestamps.

    Parameters:
    datetimes (list of str): The list of datetime strings to convert.

    Returns:
    list of int: The list of Unix timestamps.
    """
    timestamps = []
    for datetime_str in datetimes:
        dt = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")
        timestamp = int(dt.timestamp())
        timestamps.append(timestamp)
    return timestamps

def get_table_id(table):
	keys = list(FILTER_COLS.keys())
	return keys.index(table)

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

def get_table_info(file_path='./queries/tpce/db_info/col_min_max.json'):
	with open(file_path, 'r') as file:
		min_max_info = json.load(file)

	table_join_keys = {}
	table_text_cols = {}
	table_normal_cols = {}
	col_type = {}
	col2minmax = {}
	table_dim_list = []
	table_like_dim_list = []
	table_list = []

	### get min/max for cols
	for col_name in min_max_info:
		col_min_max = min_max_info[col_name]

		if col_name == 'dm_date':
			col_min_max = convert_dates_to_timestamps(col_min_max)
		elif col_name =='t_dts':
			col_min_max = convert_datetimes_to_timestamps(col_min_max)

		min_v = int(col_min_max[0])
		max_v = int(col_min_max[1])

		col2minmax[col_name] = [min_v, max_v + 1]
	#########

	### build table info
	for table in FILTER_COLS:
		# build filter cols info
		table_cols = FILTER_COLS[table]
		table_dim_list.append(len(table_cols))

		table_list.append(table)
		table_text_cols[table] = []
		table_normal_cols[table] = []
		for col in table_cols:
			full_col_name = col

			table_normal_cols[table].append(full_col_name)
			if full_col_name in CATEGORICAL_COLS:
				col_type[full_col_name] = 'categorical'
				num_dist_vals = len(CATEGORICAL_COLS_VALS[full_col_name])
				col2minmax[col] = [0, num_dist_vals]
			else:
				if full_col_name in col2minmax:
					col_type[full_col_name] = 'number'
				else:
					col_type[full_col_name] = 'string'
					col2minmax[col] = [0, IN_BUCKETS+1]

	table_key_groups = {}
	### build table join keys info
	for col_name in JOIN_MAP_TPCE:
		table = col_name.split('.')[0]
		key_group = JOIN_MAP_TPCE[col_name]
		# if key_group not in ['movie_id', 'kind_id', 'person_id', 'role_id', 'info_id',
		# 					 'keyword', 'company_id', 'company_type']:
		# 	continue

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

def normalize_val(val, min_v, max_v):
	val = int(val)
	res = float(val - min_v) / (max_v - min_v)
	if res < 0:
		return 0.
	elif res > 1:
		return 1.
	else:
		return res


def truncate_time(time_str):
	# Define the formats to check against
	formats = ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S']

	for fmt in formats:
		try:
			# Attempt to parse the string with the current format
			parsed_time = datetime.strptime(time_str, fmt)

			# If the format includes milliseconds, truncate to seconds
			if fmt == '%Y-%m-%d %H:%M:%S.%f':
				parsed_time = parsed_time.replace(microsecond=0)

			# Return the timestamp
			return parsed_time.timestamp()
		except ValueError:
			# If parsing fails, try the next format
			continue

	# If none of the formats match, return None or raise an error
	return None


def process_val(val, col_name):
	if col_name == 'dm_date':
		val = val.strip("'")
		dt = datetime.strptime(val, "%Y-%m-%d")
		val = int(dt.timestamp())
		return val
	elif col_name == 't_dts':
		val = val.strip("'")
		dt = truncate_time(val)
		val = int(dt)
		return val
	val = val.strip("'")
	return val

def template_to_group_order(ori_key_groups):
	if len(ori_key_groups):
		key_groups = copy.deepcopy(ori_key_groups)
		start_join_key = list(key_groups.keys())[0]
		current_connected_tables = copy.copy(key_groups[start_join_key][0])

		keys = [start_join_key]
		tables = [copy.copy(key_groups[start_join_key][0])]

		###### start traverse all key groups
		if len(key_groups[start_join_key]) == 1:
			del key_groups[start_join_key]
		else:
			key_groups[start_join_key].pop(0)

		while not check_empty(key_groups):
			is_progressed = False
			for join_key in key_groups:
				for t_list in key_groups[join_key]:
					if has_intersection(t_list, current_connected_tables):
						keys.append(join_key)
						tables_per_group = []
						for t in t_list:
							tables_per_group.append(t)
							if t not in current_connected_tables:
								current_connected_tables.append(t)
						key_groups[join_key].remove(t_list)
						tables.append(tables_per_group)

						is_progressed = True

			if not is_progressed:
				return None, None

		return keys, tables
	else:
		return None, None

def parse_a_query(col2minmax, join_query, all_joins, table2ops, is_final_join=False, full_key_groups=None, qid=None):
	global product_count1
	global product_count2
	global NULL_count
	global wrong_qids

	ALIAS = {}
	reverse_ALIAS= {}
	table2normal_predicates = {}
	table2text_predicates = {}
	table2reps = {}
	table2localreps = {}

	table_list_per_q = join_query['table_list']
	card = join_query['cardinality']

	if card < 1:
		return None

	if 'NULL' in join_query['where_preds'] or join_query['status'] != 'good':
		# if is_final_join:
		# 	print(join_query)
		# 	print(card)
		NULL_count += 1
		return None

	tables = []
	aliases = []
	joins = []
	join_preds = []
	mscn_joins = []
	key_groups = {}

	for t in table_list_per_q:
		table_name = t.split(' ')[0]
		alias = t.split(' ')[2]
		ALIAS[table_name] = alias
		reverse_ALIAS[alias] = table_name

		tables.append(table_name)
		aliases.append(alias)

	for alias_t in aliases:
		# table_id = get_table_id(full_t)
		table2normal_predicates[alias_t] = []
		table2reps[alias_t] = []
		table2localreps[alias_t] = []
		table2text_predicates[alias_t] = []

	### parse pred info
	for pred in join_query['where_preds']:
		parts = pred.split(" ")
		col_info = parts[0]
		col_info2 = parts[2:]
		col_info2 = ' '.join(col_info2)
		is_join_pred = False

		if col_info.split('.')[0] == 'tpce' and col_info2.split('.')[0] == 'tpce':
			is_join_pred = True

		if col_info == 'NULL':
			NULL_count += 1
			break

		op = parts[1]

		if not is_join_pred:
			alias_t = col_info.split('.')[1].strip("`")
			table_name = reverse_ALIAS[alias_t]
			col_name = col_info.split('.')[2].strip("`")

			col_id = FILTER_COLS[table_name].index(col_name)
			global_col_id =  PLAIN_FILTER_COLS.index(col_name)

			## get the col type and value
			val = col_info2.strip(')')

			if val.startswith('DATE'):
				col_type = 'date'

				val = val[4:]
			elif val.startswith('TIMESTAMP'):
				col_type = 'time'
				val = val[9:]
			elif is_int_or_float(val):
				col_type = 'number'
			else:
				col_type = 'string'

			min_val = col2minmax[col_name][0]
			max_val = col2minmax[col_name][1]

			if table_name not in table2ops:
				table2ops[table_name] = [op]
			else:
				if op not in table2ops[table_name]:
					table2ops[table_name].append(op)

			if op == '>=':
				val = process_val(val, col_name)
				is_visited = False
				for pred in table2normal_predicates[alias_t]:
					if pred[0] == col_id:
						pred[1][0][0] = normalize_val(val, min_val, max_val)
						is_visited = True
						break

				if not is_visited:
					table2normal_predicates[alias_t].append(
						[col_id, [[normalize_val(val, min_val, max_val), 1.]]])

				table2reps[alias_t].append(
						[global_col_id, op, [normalize_val(val, min_val, max_val)]])
				
				table2localreps[alias_t].append(
						[col_id, op, [normalize_val(val, min_val, max_val)]])
					
			elif op == '<=':
				val = process_val(val, col_name)
				is_visited = False
				for pred in table2normal_predicates[alias_t]:
					if pred[0] == col_id:
						pred[1][0][1] = normalize_val(val, min_val, max_val)
						is_visited = True
						break

				if not is_visited:
					table2normal_predicates[alias_t].append(
						[col_id, [[0., normalize_val(val, min_val, max_val)]]])
				
				table2reps[alias_t].append(
						[global_col_id, op, [normalize_val(val, min_val, max_val)]])
				
				table2localreps[alias_t].append(
						[col_id, op, [normalize_val(val, min_val, max_val)]])

			elif op == '=':
				val = process_val(val, col_name)
				if isinstance(val, str):
					if val.lower().startswith('_latin1'):
						val = val[7:]
						val = val.strip("'")
				if col_name in CATEGORICAL_COLS_VALS:
					val = CATEGORICAL_COLS_VALS[col_name].index(val)
				elif col_type == 'string':
					val = deterministic_hash(val) % IN_BUCKETS

				val = int(val)

				normal_val1 = normalize_val(val, min_val, max_val)
				normal_val2 = normalize_val(val + 1, min_val, max_val)

				table2normal_predicates[alias_t].append([col_id, [[normal_val1, normal_val2]]])

				table2reps[alias_t].append(
						[global_col_id, op, [normal_val1]])

				table2localreps[alias_t].append(
						[col_id, op, [normal_val1]])

			elif op == 'between':
				val1 = int(val.split('and')[0].strip())
				val2 = int(val.split('and')[1].strip())

				normal_val1 = normalize_val(val1, min_val, max_val)
				normal_val2 = normalize_val(val2 + 1, min_val, max_val)

				table2normal_predicates[alias_t].append([col_id, [[normal_val1, normal_val2]]])

				table2reps[alias_t].append(
						[global_col_id, op, [normal_val1, normal_val2]])
				
				table2localreps[alias_t].append(
						[col_id, op, [normal_val1, normal_val2]])

		else:
			### for join predicate

			alias_table_name1 = col_info.split('.')[1].strip("`")
			col_name1 = col_info.split('.')[2].strip("`")

			# for the right part
			alias_table_name2 = col_info2.split('.')[1].strip("`")
			col_name2 = col_info2.split('.')[2].strip("`")

			new_join = "{}.{} = {}.{}".format(alias_table_name1, col_name1, 
									alias_table_name2, col_name2)
			mscn_joins.append(new_join)

			if new_join not in all_joins:
				all_joins.append(new_join)

			if is_final_join:
				if pred not in join_preds:
					join_preds.append(pred)
					alias_table_name1 = col_info.split('.')[1].strip("`")
					col_name1 = col_info.split('.')[2].strip("`")

					# for the right part
					alias_table_name2 = col_info2.split('.')[1].strip("`")
					col_name2 = col_info2.split('.')[2].strip("`")

					joins.append((alias_table_name1 + '.' + col_name1,
								  alias_table_name2 + '.' + col_name2))

	### parse join info
	key_groups_tables = []
	if is_final_join:
		for join in joins:
			alias_table1 = join[0].split('.')[0]
			alias_table2 = join[1].split('.')[0]

			col1 = join[0].split('.')[1]
			full_key_name1 = reverse_ALIAS[alias_table1] + '.' + col1
			key_group = JOIN_MAP_TPCE[full_key_name1]

			if key_group not in key_groups:
				key_groups[key_group] = [[alias_table1, alias_table2]]
			else:
				key_groups[key_group].append([alias_table1, alias_table2])

		# merge same groups
		for key_group in key_groups:
			table_list = key_groups[key_group]
			key_groups[key_group] = find_connected_clusters(table_list)
	else:
		for key_group in full_key_groups:
			for t_set in full_key_groups[key_group]:
				intersect_ts = [t for t in t_set if t in aliases]
				if len(intersect_ts) >= 2:
					key_groups_tables.extend(intersect_ts)
					if key_group not in key_groups:
						key_groups[key_group] = [intersect_ts]
					else:
						key_groups[key_group].append(intersect_ts)

	query_info = [table2normal_predicates, table2text_predicates, table2reps, table2localreps, key_groups, mscn_joins, sorted(aliases)]

	if is_final_join:
		return (query_info, card, key_groups)
	else:
		### check if key groups are not connected. If so, tables are cross-product joined, not actually joined
		for t in aliases:
			if t not in key_groups_tables:
				# if join_query['table_len'] > 1:
				# 	print('=============')
				# 	print("original query fingerprint_md5: {}".format(qid))
				# 	print(join_query)
				# 	print("missing: {}".format(t))
				if len(tables) != len(set(tables)):
					product_count2 += 1
					if qid not in wrong_qids:
						wrong_qids.append(qid)
				elif join_query['table_len'] == 1:
					product_count2 += 1
				product_count1 += 1
				return None

		a,b = template_to_group_order(key_groups)
		if a is not None:
			# print(aliases)
			# print(key_groups)
			# print(card)
			return (query_info, card, key_groups)
		else:
			if qid not in wrong_qids:
				wrong_qids.append(qid)
			# product_count2 += 1
			return None


def read_query_file(col2minmax, num_q=400000, test_size=1000, file_path='./queries/tpce_queries_cleaned.json'):
	with open(file_path, 'r') as file:
		queries = json.load(file)

	# Filter out files that do not have a .csv extension

	random.seed(42)

	training_queries = []
	training_cards = []

	table2ops = {}

	test_queries = []
	test_cards = []

	query_str_set = []
	final_join_query = None
	single_table_queries = []
	single_table_queries_cards = []

	full_key_groups = {}
	single_t_count = 0

	invalid_count = 0
	all_table_alias, all_joins, max_num_tables, max_num_joins  = [], [], 0, 0

	sub_templates_in_training_ratio = 0.1
	tablelist2choice = {}

	for qid, query_set in enumerate(queries):
		# print(qid)
		if len(query_set["subset_sqls"]) == 0:
			single_t_count += 1
			continue

		final_join_query_id = 0
		current_table_len = query_set["subset_sqls"][0]['table_len']

		for qid, sub_query in enumerate(query_set["subset_sqls"]):
			# if sub_query['table_len'] == 1:
			# 	alias_table = sub_query['table_list'][0].split(' ')[-1]
			# 	table2cards[alias_table] = sub_query['cardinality']

			if sub_query['table_len'] > current_table_len:
				current_table_len = sub_query['table_len']
				final_join_query_id = qid

		is_valid = True

		parse_res = parse_a_query(col2minmax, query_set["subset_sqls"][final_join_query_id], all_joins, table2ops, 
								is_final_join=True)
	

		if parse_res is None:
			invalid_count += 1

		if parse_res is not None:
			query_info = parse_res[0]
			card = parse_res[1]
			full_key_groups = parse_res[2]

			final_join_query = query_set["subset_sqls"][final_join_query_id]

			ts = query_info[-1]

			if len(ts) == 1:
				continue

			mscn_joins = query_info[-2]
			for t in ts:
				if t not in all_table_alias:
					all_table_alias.append(t)
			if len(ts) > max_num_tables:
				max_num_tables = len(ts)
			if len(mscn_joins) > max_num_joins:
				max_num_joins = len(mscn_joins)

			if tuple(sorted(ts)) not in tablelist2choice:
				r_number = random.uniform(0, 1)
				if r_number < 1:
					tablelist2choice[tuple(sorted(ts))] = True  # training
				else:
					tablelist2choice[tuple(sorted(ts))] = False  # test

			if qid < num_q:
				if tablelist2choice[tuple(sorted(ts))]:
					training_queries.append(query_info)
					training_cards.append(card)
				else:
					test_queries.append(query_info)
					test_cards.append(card)
			else:
				is_valid = False
		else:
			continue

		if not is_valid:
			break

		for qid, sub_query in enumerate(query_set["subset_sqls"]):
			if qid != final_join_query_id:
				parse_res = parse_a_query(col2minmax, sub_query, all_joins, table2ops,
							  False, full_key_groups, qid=query_set['fingerprint_md5'])

				if parse_res is None:
					invalid_count += 1
					continue
				else:
					query_info = parse_res[0]
					card = parse_res[1]
					q_str = json.dumps(query_info[0])

					tables = query_info[-1]
					if tuple(sorted(tables)) not in tablelist2choice:
						r_number = random.uniform(0, 1)
						if r_number < sub_templates_in_training_ratio:
							tablelist2choice[tuple(sorted(tables))] = True  # training
						else:
							tablelist2choice[tuple(sorted(tables))] = False  # test

					if q_str not in query_str_set:
						query_str_set.append(q_str)

						if tablelist2choice[tuple(sorted(tables))]:
							training_queries.append(query_info)
							training_cards.append(card)
						else:
							test_queries.append(query_info)
							test_cards.append(card)

	print('number of training qs')
	print(len(training_queries))
	template2queries = {}
	template2cards = {}

	for query_info, card in zip(training_queries, training_cards):
		table_list = query_info[-1]
		table_list = tuple(table_list)

		if table_list not in template2queries:
			template2queries[table_list] = [query_info]
			template2cards[table_list] = [card]
		else:
			template2queries[table_list].append(query_info)
			template2cards[table_list].append(card)

	test_template2queries = {}
	test_template2cards = {}
	
	print('ss')
	print(len(test_queries))

	for query_info, card in zip(test_queries, test_cards):
		table_list = query_info[-1]
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

	print('single_t_count')
	print(single_t_count)

	print('invalid_count')
	print(invalid_count)

	print('NULL_count')
	print(NULL_count)

	print('product_count1')
	print(product_count1)

	print('product_count2')
	print(product_count2)

	print(tablelist2choice)

	print(wrong_qids)
	return template2queries, template2cards, test_template2queries, test_template2cards,\
			all_table_alias, all_joins, max_num_tables, max_num_joins, table2ops

import copy
import re
import itertools
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from CEB_utlities.join_utilits_CEB_job import IN_BUCKETS
from GRASP.utils import *
from GRASP.ce_models import NeuCDF, MSCNCE
from GRASP.ce_models import AutoregressiveCDF
from GRASP.lcs_models import UnifiedLCSPredictor
import torch.nn as nn
from CEB_utlities.NeuroCDF_helpers import *

ALIAS = {'title': 't',
		 'kind_type': 'kt',
		 'keyword': 'k',
		 'movie_info': 'mi',
		 'movie_info_idx': 'mii',
		 'info_type': 'it',
		 'cast_info': 'ci',
		 'role_type': 'rt',
		 'name': 'n',
		 'movie_companies': 'mc',
		 'company_name': 'cn',
		 'company_type': 'ct',
		 'movie_keyword': 'mk',
		 'movie_link':'ml',
		 'aka_name': 'an',
		 'person_info': 'pi'}

JOIN_MAP_IMDB_UPPER_BOUND = {
	'movie_id':2528312,
	'keyword':134170,
	'person_id':4167491,
	'kind_id':7,
	'role_id':12,
	'info_id':113,
	'company_type':4,
	'company_id':234997,
}

reverse_alias = {value: key for key, value in ALIAS.items()}

OPS = ['lt', 'eq', 'in', 'like']

class JoinHandler():
	def __init__(self, table_list, table_dim_list, table_key_groups,
				 table_size_list, cdf_model_choice='arcdf',
				  hidden_size=64, lcs_size=10, colid2featlen_per_table=None):
		super().__init__()

		"""
		Initializes the join handler with the given parameters.
		Args:
			table_list (list): List of table names.
			table_dim_list (list): List of dimensions for each table.
			table_key_groups (list): List of key groups for each table.
			table_size_list (list): List of sizes for each table.
			cdf_model_choice (str): Choice of CDF model ('neucdf' or 'arcdf').
			hidden_size (int, optional): Size of the hidden layer. Defaults to 64.
			lcs_size (int, optional): Size of the join bin in LCS model. Defaults to 10.
			colid2featlen_per_table (dict, optional): Dictionary mapping column IDs to feature lengths per table. Defaults to None.
		
		"""

		self.table_list = table_list
		self.table2id = {}
		self.tid2table = {}
		for tid, table_name in enumerate(table_list):
			self.table2id[table_name] = tid
			self.tid2table[tid] = table_name

		self.hidden_size = hidden_size
		self.table_dim_list = table_dim_list
		self.table_key_groups = table_key_groups
		self.table_size_list = table_size_list
		self.lcs_size = lcs_size
		self.cdf_model_choice = cdf_model_choice
		self.colid2featlen_per_table = colid2featlen_per_table

		self.table_cdf_models = []
		self.table_ce_models = []
		self.table_unified_join_predictors = {}
		self.table_join_predictor_types = {}

		self.join_keys_per_table = {}
		self.table2keygroups = {}

		if cdf_model_choice == 'neucdf':
			for num_dim in table_dim_list:
				self.table_cdf_models.append(NeuCDF(num_dim, hidden_size, lcs_size))
		elif cdf_model_choice == 'arcdf':
			for num_dim in table_dim_list:
				if num_dim > 0:
					self.table_cdf_models.append(AutoregressiveCDF(num_dim, hidden_size, lcs_size))
				else:
					self.table_cdf_models.append(None)
		else:
			raise Exception("Sorry, model not supported")


	def init_ce_predictors(self, colid2featlen_per_table):
		"""
        Initializes the cardinality estimation predictors for each table.

        Args:
            colid2featlen_per_table (dict): Mapping from table name to column feature lengths.
        """
		self.max_featlen_per_table = {}
		for table_name in self.table_size_list:
			self.max_featlen_per_table[table_name] = 0

		for key_group in self.table_key_groups:
			tid_list = []
			for table_name in self.table_key_groups[key_group]:
				if table_name not in self.table_size_list:
					continue

				if table_name not in self.table2keygroups:
					self.table2keygroups[table_name] = [key_group]
				else:
					self.table2keygroups[table_name].append(key_group)

				table_id = self.table2id[table_name]
				tid_list.append(table_id)

			for t_id, num_dim in enumerate(self.table_dim_list):
				if t_id in tid_list:
					if colid2featlen_per_table is not None:
						table_name = self.tid2table[t_id]
						col_size_dic = colid2featlen_per_table[table_name]
						feat_size = 0
						for col_id in col_size_dic:
							if col_size_dic[col_id] > feat_size:
								feat_size = col_size_dic[col_id]
						input_size = feat_size + len(OPS) + self.table_dim_list[t_id]
						self.max_featlen_per_table[table_name] = input_size
		
		for t_id, num_dim in enumerate(self.table_dim_list):
			### init mscn ce models
			if num_dim > 0:
				table_name = self.tid2table[t_id]
				self.table_ce_models.append(MSCNCE(self.max_featlen_per_table[table_name],
										self.hidden_size))
			else:
				self.table_ce_models.append(None)

	def init_unified_lcs_predictors(self):
		"""
        Initializes the LCS predictors for each table and key group.
        """
		self.bin_size_per_key_group = {}

		for key_group in self.table_key_groups:
			key_group_lcs_size = self.lcs_size
			if key_group in JOIN_MAP_IMDB_UPPER_BOUND:
				if key_group_lcs_size > JOIN_MAP_IMDB_UPPER_BOUND[key_group]:
					key_group_lcs_size =  JOIN_MAP_IMDB_UPPER_BOUND[key_group]
			self.bin_size_per_key_group[key_group] = key_group_lcs_size

			tid_list = []
			for table_name in self.table_key_groups[key_group]:

				if table_name not in self.table_size_list:
					continue

				if table_name not in self.table2keygroups:
					self.table2keygroups[table_name] = [key_group]
				else:
					self.table2keygroups[table_name].append(key_group)

				table_id = self.table2id[table_name]
				tid_list.append(table_id)

			for t_id, num_dim in enumerate(self.table_dim_list):
				if t_id in tid_list:
					table_name = self.tid2table[t_id]
					if table_name not in self.join_keys_per_table:
						self.join_keys_per_table[table_name] = {key_group: self.bin_size_per_key_group[key_group]}
					else:
						self.join_keys_per_table[table_name][key_group] = self.bin_size_per_key_group[key_group]

		for table_name in self.join_keys_per_table:
			input_size = self.max_featlen_per_table[table_name] 
			output_sizes = []
			for jk in self.join_keys_per_table[table_name]:
				output_sizes.append(self.join_keys_per_table[table_name][jk])
			self.table_unified_join_predictors[table_name] = UnifiedLCSPredictor(input_size, self.hidden_size, output_sizes)

	def get_parameters(self):
		trainable_paras = []
		for cdf_model in self.table_cdf_models:
			if cdf_model is not None:
				trainable_paras.extend(cdf_model.parameters())

		for cdf_model in self.table_ce_models:
			if cdf_model is not None:
				trainable_paras.extend(cdf_model.parameters())

		for jk in self.table_unified_join_predictors:
			join_model = self.table_unified_join_predictors[jk]
			if join_model is not None:
				trainable_paras.extend(join_model.parameters())

		return trainable_paras

	def generate_key_group_mask(self, table_list, key_groups):
		"""
        Generates masks for key groups and tables for join queries.

        Args:
            table_list (list): List of tables in the join.
            key_groups (dict): Key groups for the join.

        Returns:
            tuple: (per_t_mask, per_t_group_mask)
        """
		per_t_mask = []
		per_t_group_mask = []
		
		if len(list(table_list))> 1:
			for key_group in key_groups:
				key_group_mask  = []
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
				key_group_mask  = []
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

	def template_to_group_order(self, ori_key_groups):
		"""
        Orders key groups and tables for join traversal.

        Args:
            ori_key_groups (dict): Original key groups.

        Returns:
            tuple: (keys, tables) in traversal order.
        """

		if len(ori_key_groups) == 0:
			return None, None

		key_groups = copy.deepcopy(ori_key_groups)
		start_join_key = list(key_groups.keys())[0]
		current_connected_tables = copy.deepcopy(key_groups[start_join_key][0])

		keys = [start_join_key]
		tables = [copy.deepcopy(key_groups[start_join_key][0])]

		###### start traverse all key groups

		del key_groups[start_join_key][0]

		while not check_empty(key_groups):
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
		return keys, tables

	def template_to_group_order_w_mask(self, ori_key_groups, t_mask_list, t_group_mask_list):
		"""
        Orders key groups and tables for join traversal, with masks.

        Args:
            ori_key_groups (dict): Original key groups.
            t_mask_list (list): Table mask results.
            t_group_mask_list (list): Table group mask results.

        Returns:
            tuple: (keys, tables, final_t_mask_list, final_t_group_mask_list, use_table_sels_list)
        """

		if len(ori_key_groups) == 0:
			return None, None

		key_groups = copy.deepcopy(ori_key_groups)
		key_list = list(key_groups.keys())
		start_join_key = key_list[0]
		current_connected_tables = copy.deepcopy(key_groups[start_join_key][0])

		travesal_indicators = []
		for k in key_groups:
			key_indicator = []
			for _ in key_groups[k]:
				key_indicator.append(1)
			travesal_indicators.append(key_indicator)

		keys = [start_join_key]
		tables = [copy.deepcopy(key_groups[start_join_key][0])]
		travesal_indicators[0][0] = 0

		final_t_mask_list = []
		final_t_group_mask_list = []
		use_table_sels_list = []
		current_connected_tables_w_mask_list = []

		for t_mask, t_group_mask in zip(t_mask_list, t_group_mask_list):
			final_t_mask_list.append([t_mask[0][0]])
			final_t_group_mask_list.append([t_group_mask[0][0]])
			use_table_sels_list.append([t_mask[0][0]])

			current_connected_tables_w_mask = []
			for t, a_t_mask in zip(key_groups[start_join_key][0], t_mask[0][0]):
				if a_t_mask == 1:
					current_connected_tables_w_mask.append(t)
			current_connected_tables_w_mask_list.append(current_connected_tables_w_mask)

		###### start traverse all key groups
		while not check_empty_for_indicator(travesal_indicators):
			for join_key in key_groups:
				jk_id = key_list.index(join_key)
				for t_list_id, t_list in enumerate(key_groups[join_key]):
					if has_intersection(t_list, current_connected_tables) and travesal_indicators[jk_id][t_list_id] == 1:
						keys.append(join_key)
						tables_per_group = []

						use_table_sels_per_group_list = []
						for _ in range(len(final_t_mask_list)):
							use_table_sels_per_group_list.append([])

						for tid, t in enumerate(t_list):
							tables_per_group.append(t)

							if t not in current_connected_tables:
								current_connected_tables.append(t)
							
							for qid in range(len(final_t_mask_list)):
								if t not in current_connected_tables_w_mask_list[qid]:
									if t_mask_list[qid][jk_id][t_list_id][tid] == 1:
										use_table_sels_per_group_list[qid].append(1)
										current_connected_tables_w_mask_list[qid].append(t)
									else:
										use_table_sels_per_group_list[qid].append(0)
								else:
									use_table_sels_per_group_list[qid].append(0)

						travesal_indicators[jk_id][t_list_id] = 0
						tables.append(tables_per_group)

						for qid in range(len(final_t_mask_list)):
							use_table_sels_list[qid].append(use_table_sels_per_group_list[qid])
							final_t_mask_list[qid].append(t_mask_list[qid][jk_id][t_list_id])
							final_t_group_mask_list[qid].append(t_group_mask_list[qid][jk_id][t_list_id])

		return keys, tables, final_t_mask_list, final_t_group_mask_list, use_table_sels_list

	def separate_in_clase(self, table2predicates):
		"""
        Separates IN clauses in predicates for each table.

        Args:
            table2predicates (dict): Table to predicates.

        Returns:
            dict: Table to normal predicates.
        """
		table2normalpredicates = {}
		for t in table2predicates:
			table_res_list = []
			col_predicates = table2predicates[t]
			for col_id, col_ranges in col_predicates:
				if len(col_ranges) == 1:
					# only one range, not IN
					table_res_list.append([[col_id, col_ranges[0]]])
				else:
					# >=2 ranges, IN clause
					col_res = []
					for interval in col_ranges:
						col_res.append([col_id, interval])
					table_res_list.append(col_res)

			table_res_list = list(itertools.product(*table_res_list))
			table_res_list = [list(comb) for comb in table_res_list]  # Convert tuples to lists
			table2normalpredicates[t] = table_res_list

		return table2normalpredicates

	def load_training_queries(self, table2predicates_list, table2contexts_list, table2qreps_list, training_cards, bs=64, temp_table_list=None, is_cuda=True):
		"""
        Loads and processes training queries into batches for model training.

        Args:
            table2predicates_list (dict): table to list of query predicates.
            table2contexts_list (dict): table to list of query contexts.
            table2qreps_list (dict): table to list of query representations.
            training_cards (list): List of training cardinalities.
            bs (int): Batch size.
            temp_table_list (list, optional): table list for the template.
            is_cuda (bool): Whether to use CUDA.

        Returns:
            tuple: (dataloader, table_list)
        """

		if temp_table_list is None:
			table_list = list(table2predicates_list[0].keys())
		else:
			table_list = temp_table_list

		normal_qs_list = []
		qreps_list = []

		for join_query, query_contexts, qreps in zip(table2predicates_list, table2contexts_list, table2qreps_list):
			query_per_table = {}
			qrep_per_table = {}
			context_per_table = {}
			table2normalpredicates = self.separate_in_clase(join_query)
			for t in table_list:
				if t not in table2normalpredicates:
					is_found = False
					for original_t in table2normalpredicates:
						if (len(table2normalpredicates[original_t]) != 0 or len(qreps[original_t]) != 0) and drop_trailing_number(original_t) == t:
							look_t = original_t

							query_per_table[t] = table2normalpredicates[look_t]
							qrep_per_table[t] = qreps[look_t]
							context_per_table[t] = query_contexts[look_t]
							is_found = True
							break
					
					if not is_found:
						query_per_table[t] = []
						qrep_per_table[t] = []
						qrep_per_table[t] = []
				else:
					query_per_table[t] = table2normalpredicates[t]
					qrep_per_table[t] = qreps[t]
					context_per_table[t] = query_contexts[t]

			normal_qs_list.append(query_per_table)
			qreps_list.append(qrep_per_table)

		table2queries = {}
		table2reps = {}

		table2cdfs = {}
		table2signs = {}

		for table_name in table_list:
			table2queries[table_name] = []
			table2reps[table_name] = []
			table2cdfs[table_name] = []
			table2signs[table_name] = []

		for normal_qs, qreps in zip(normal_qs_list, qreps_list):
			for table_name in table_list:
				res_per_join_query = []

				for normal_q in normal_qs[table_name]:
					### all normal_q have the same context_on_table
					res_per_join_query.append(normal_q)

				table2queries[table_name].append(res_per_join_query)
				table2reps[table_name].append(qreps[table_name])

		dataloader_list = []
		training_cards = np.array(training_cards)
		training_cards = torch.from_numpy(training_cards)
		if is_cuda:
			training_cards = training_cards.cuda()

		for table_name in table_list:
			ori_table_alias = drop_trailing_number(table_name)
			ori_table_name = reverse_alias[ori_table_alias]
			table_id = self.table2id[ori_table_name]

			batch_queries = table2queries[table_name]
			batch_reps =  table2reps[table_name]

			num_cols = self.table_dim_list[table_id]

			batch_cdfs, batch_signs, batch_masks = multi_queries_batch_query2cdfs(batch_queries, num_cols)
			# batch_cdfs: [bs, num_normal_queries, num_cdfs, num_col]

			max_feat_len = self.max_featlen_per_table[ori_table_name]
			batch_reps, batch_rep_masks = multi_batch_query2reps(batch_reps, OPS, num_cols, max_feat_len)

			batch_cdfs = torch.from_numpy(batch_cdfs)
			batch_signs = torch.from_numpy(batch_signs)
			batch_masks = torch.from_numpy(batch_masks)

			if is_cuda:
				batch_cdfs = batch_cdfs.cuda()
				batch_signs = batch_signs.cuda()
				batch_masks = batch_masks.cuda()
				batch_reps = batch_reps.cuda()


				batch_rep_masks = batch_rep_masks.cuda()

			dataloader_list.extend([batch_cdfs, batch_signs, batch_reps,
									batch_masks, batch_rep_masks])

		# dataloader_list.append(batch_masks)
		dataloader_list.append(training_cards)

		dataloader_list = TensorDataset(*dataloader_list)
		dataloader = DataLoader(dataloader_list, batch_size=bs, shuffle=True)

		return dataloader, table_list

	def load_training_queries_w_masks(self, table2predicates_list, table2contexts_list, table2qreps_list, ori_key_groups, training_table_masks,
								   training_table_group_masks, is_stb_queries, training_cards, bs=64, is_cuda=True, is_shuffle=True):
		#### list of join queries
		#### batch_size * table2predicates_list

		normal_qs_list = []
		qreps_list = []

		table_list = list(table2predicates_list[0].keys())

		for join_query, qreps in zip(table2predicates_list, table2qreps_list):
			query_per_table = {}
			qrep_per_table = {}
			table2normalpredicates = self.separate_in_clase(join_query)
			for t in table_list:
				query_per_table[t] = table2normalpredicates[t]
				qrep_per_table[t] = qreps[t]

			normal_qs_list.append(query_per_table)
			qreps_list.append(qrep_per_table)

		table2queries = {}
		table2reps = {}

		table2cdfs = {}
		table2signs = {}

		for table_name in table_list:
			table2queries[table_name] = []
			table2reps[table_name] = []
			table2cdfs[table_name] = []
			table2signs[table_name] = []

		for normal_qs, qreps in zip(normal_qs_list, qreps_list):
			for table_name in table_list:
				res_per_join_query = []

				for normal_q in normal_qs[table_name]:
					### all normal_q have the same context_on_table
					res_per_join_query.append(normal_q)

				table2queries[table_name].append(res_per_join_query)
				table2reps[table_name].append(qreps[table_name])

		dataloader_list = []
		num_q = len(training_cards)
		training_cards = np.array(training_cards)
		training_cards = torch.from_numpy(training_cards)
		if is_cuda:
			training_cards = training_cards.cuda()

		for table_name in table_list:
			ori_table_alias = drop_trailing_number(table_name)
			ori_table_name = reverse_alias[ori_table_alias]
			table_id = self.table2id[ori_table_name]

			batch_queries = table2queries[table_name]
			batch_reps =  table2reps[table_name]

			num_cols = self.table_dim_list[table_id]

			batch_cdfs, batch_signs, batch_masks = multi_queries_batch_query2cdfs(batch_queries, num_cols)
			# batch_cdfs: [bs, num_normal_queries, num_cdfs, num_col]

			max_feat_len = self.max_featlen_per_table[ori_table_name]
			batch_reps, batch_rep_masks = multi_batch_query2reps(batch_reps, OPS, num_cols, max_feat_len)

			batch_cdfs = torch.from_numpy(batch_cdfs)
			batch_signs = torch.from_numpy(batch_signs)
			batch_masks = torch.from_numpy(batch_masks)

			if is_cuda:
				batch_cdfs = batch_cdfs.cuda()
				batch_signs = batch_signs.cuda()
				batch_masks = batch_masks.cuda()
				batch_reps = batch_reps.cuda()
				batch_rep_masks = batch_rep_masks.cuda()

			dataloader_list.extend([batch_cdfs, batch_signs, batch_reps,
									batch_masks, batch_rep_masks])

		keys, tables, t_masks, t_group_masks, use_table_sels = self.template_to_group_order_w_mask(ori_key_groups, training_table_masks, training_table_group_masks)

		for order_id in range(len(keys)):
			batch_t_masks = []
			batch_t_group_masks = [] 
			batch_use_table_sels = [] 

			for t_mask, t_group_mask, use_table_sel in zip(t_masks, t_group_masks, use_table_sels):
				batch_t_masks.append(t_mask[order_id])
				batch_t_group_masks.append(t_group_mask[order_id])
				batch_use_table_sels.append(use_table_sel[order_id])

			batch_t_masks = torch.from_numpy(np.array(batch_t_masks))
			batch_t_group_masks = torch.from_numpy(np.array(batch_t_group_masks))
			batch_use_table_sels = torch.from_numpy(np.array(batch_use_table_sels))

			if is_cuda:
				batch_t_masks = batch_t_masks.cuda()
				batch_t_group_masks = batch_t_group_masks.cuda()
				batch_use_table_sels = batch_use_table_sels.cuda()

			dataloader_list.append(batch_t_masks)
			dataloader_list.append(batch_t_group_masks)
			dataloader_list.append(batch_use_table_sels)

		is_stb_queries = torch.from_numpy(np.array(is_stb_queries))
		if is_cuda:
			is_stb_queries = is_stb_queries.cuda()
			
		dataloader_list.append(is_stb_queries)
		dataloader_list.append(training_cards)

		dataloader_list = TensorDataset(*dataloader_list)

		if bs > num_q:
			bs = num_q
		dataloader = DataLoader(dataloader_list, batch_size=bs, shuffle=is_shuffle)

		return dataloader, table_list, keys, tables


	def batch_estimate_join_queries_from_loader(self, databatch, table_list, keys_order, tables_order, is_cuda=True, print_details=False):

		"""
		Estimates the cardinality of join queries from a data batch, assuming that each batch is of the same join template
		uses mscn card est model as the per-table model for join queries that contain complex predictes, and arcdf for range queries.

		Parameters:
		-----------
		databatch : list
			A list containing data for each table in the join query. Each table's data is represented by 8 elements:
			- cdfs_on_table
			- signs_on_table
			- qreps_on_table
			- batch_masks
			- batch_qreps_masks
		table_list : list
			A list of table names involved in the join query.
		keys_order : list
			A list of keys used for ordering the join keys.
		tables_order : list
			A list of tables used for ordering the join tables.
		is_cuda : bool, optional
			If True, the computations will be performed on a CUDA-enabled GPU. Default is True.
		print_details : bool, optional
			If True, additional details will be printed during the execution. Default is False.
		Returns:
		--------
		torch.Tensor
			The estimated cardinality of the join query.
		"""
		
		if print_details:
			print(table_list)

		if len(table_list) == 1:
			tid = 0
			table_name = table_list[0]
			ori_table_alias = drop_trailing_number(table_name)
			ori_table_name = reverse_alias[ori_table_alias]
			table_id = self.table2id[ori_table_name]
			card_scaling = self.table_size_list[ori_table_name]

			cdfs_on_table = databatch[5 * tid]
			signs_on_table = databatch[5 * tid + 1]
			qreps_on_table = databatch[5 * tid + 2]

			batch_masks = databatch[5 * tid + 3]
			batch_qreps_masks = databatch[5 * tid + 4]
			

			# batch_cdfs: [bs, num_normal_queries, num_cdfs, num_col]
			# batch_contexts: [bs, num_contexts, context_size]

			bs = cdfs_on_table.shape[0]
			num_normal_qs = cdfs_on_table.shape[1]
			num_cdfs = cdfs_on_table.shape[2]
			batch_masks = batch_masks.view(bs, num_normal_qs)

			if ori_table_alias != 'mii': # for mscn ce models
				if self.table_ce_models[table_id] is not None:
					cdf_est = self.table_ce_models[table_id](qreps_on_table, batch_qreps_masks)
					q_sel_est = cdf_est
				else:
					q_sel_est = torch.ones(bs, 1)
					if is_cuda:
						q_sel_est = q_sel_est.cuda()
			else:
				if self.table_cdf_models[table_id] is not None:
					cdfs_on_table = cdfs_on_table.view(-1, cdfs_on_table.shape[-1])
					cdf_est = self.table_cdf_models[table_id](cdfs_on_table)

					cdf_est = cdf_est.view(bs, num_normal_qs, -1)
					q_sel_est = torch.sum(cdf_est * signs_on_table, dim=-1)  # [bs, num_normal_queries];
					q_sel_est = torch.sum(q_sel_est * batch_masks, dim=-1, keepdim=True)  # [bs, 1];

				else:
					q_sel_est = torch.ones(bs, 1)
					if is_cuda:
						q_sel_est = q_sel_est.cuda()

			sel_result = q_sel_est
			return torch.squeeze(sel_result * card_scaling)
		else:
			###
			sel_result = {}
			lcs_result = {}
			current_connected_tables = []
			bs = databatch[0].shape[0]

			for table_name in table_list:
				lcs_result[table_name] = {}

			### obtain base estimates
			for tid, table_name in enumerate(table_list):
				ori_table_alias = drop_trailing_number(table_name)
				ori_table_name = reverse_alias[ori_table_alias]
				table_id = self.table2id[ori_table_name]

				cdfs_on_table = databatch[5*tid]
				signs_on_table = databatch[5*tid+1]
				qreps_on_table = databatch[5*tid+2]

				batch_masks =  databatch[5*tid+3]
				batch_qreps_masks = databatch[5 * tid + 4]

				# batch_cdfs: [bs, num_normal_queries, num_cdfs, num_col]
				# batch_contexts: [bs, num_contexts, context_size]

				bs = cdfs_on_table.shape[0]
				num_normal_qs = cdfs_on_table.shape[1]

				batch_masks = batch_masks.view(bs, num_normal_qs)

				if ori_table_alias != 'mii': # for mscn ce models
					query_groups = list(set(keys_order) & set(self.table2keygroups[ori_table_name]))

					jk_idxs = []
					for jk in query_groups:
						jk_idxs.append(list(self.join_keys_per_table[ori_table_name].keys()).index(jk))

					jk_preds_per_group, _ = self.table_unified_join_predictors[ori_table_name](qreps_on_table,batch_qreps_masks,jk_idxs) 
					for jk in query_groups:
						jk_id = list(self.join_keys_per_table[ori_table_name].keys()).index(jk)
						lcs_result[table_name][jk]  = jk_preds_per_group[jk_id]

					# get ce estimates
					if self.table_ce_models[table_id] is not None:
						cdf_est = self.table_ce_models[table_id](qreps_on_table, batch_qreps_masks)
						q_sel_est = cdf_est
					else:
						q_sel_est = torch.ones(bs, 1)
						if is_cuda:
							q_sel_est = q_sel_est.cuda()

					sel_result[table_name] = q_sel_est
				else:
				
					query_groups = list(set(keys_order) & set(self.table2keygroups[ori_table_name]))

					jk_idxs = []
					for jk in query_groups:
						jk_idxs.append(list(self.join_keys_per_table[ori_table_name].keys()).index(jk))

					jk_preds_per_group, _ = self.table_unified_join_predictors[ori_table_name](qreps_on_table,batch_qreps_masks,jk_idxs) 
					for jk in query_groups:
						jk_id = list(self.join_keys_per_table[ori_table_name].keys()).index(jk)
						lcs_result[table_name][jk]  = jk_preds_per_group[jk_id]

					if self.table_cdf_models[table_id] is not None:
						input_cdfs_on_table = cdfs_on_table.view(-1, cdfs_on_table.shape[-1])
						cdf_est = self.table_cdf_models[table_id](input_cdfs_on_table)
						cdf_est = cdf_est.view(bs, num_normal_qs, -1)
						q_sel_est = torch.sum(cdf_est * signs_on_table, dim=-1)  # [bs, num_normal_queries];
						q_sel_est = torch.sum(q_sel_est * batch_masks, dim=-1, keepdim=True)  # [bs, 1];
					else:
						q_sel_est = torch.ones(bs, 1)
						if is_cuda:
							q_sel_est = q_sel_est.cuda()

					sel_result[table_name] = q_sel_est

			### obtain join estimates

			curr_card_pred = torch.ones(bs)
			if is_cuda:
				curr_card_pred = curr_card_pred.cuda()

			for key_id, (key_group, tables) in enumerate(zip(keys_order, tables_order)):
				if print_details:
					print(key_id)
				lcs_size = lcs_result[tables[0]][key_group].shape[-1]
				bin_sel_preds_prod = torch.ones(bs, lcs_size)
				sel_prod =  torch.ones(bs)

				if is_cuda:
					bin_sel_preds_prod = bin_sel_preds_prod.cuda()
					sel_prod = sel_prod.cuda()

				for t_id, table_name in enumerate(tables):
					ori_table_alias = drop_trailing_number(table_name)
					ori_table_name = reverse_alias[ori_table_alias]
					table_id = self.table2id[ori_table_name]

					table_sel = sel_result[table_name]
					table_sel = torch.squeeze(table_sel)
					bin_probs = lcs_result[table_name][key_group]

					if table_name not in current_connected_tables:
						card_scaling =  self.table_size_list[ori_table_name]
						sel_prod = sel_prod * table_sel * card_scaling
						current_connected_tables.append(table_name)

					bin_sel_preds_prod = bin_sel_preds_prod * bin_probs
			

				bin_sum =  torch.sum(bin_sel_preds_prod, dim=-1)
				curr_card_pred = curr_card_pred * bin_sum * sel_prod

		return curr_card_pred
	
	def batch_estimate_join_queries_from_loader_w_mask(self, databatch, table_list, keys_order, tables_order, 
														   is_cuda=True, print_details=False):

		"""
		Estimates the cardinality of join queries from a data batch, without assuming that each batch is of the same join template
		uses mscn card est model as the per-table model for join queries that contain complex predictes, and arcdf for range queries.
		"""

		if print_details:
			print(table_list)

		if len(table_list) == 1:
			tid = 0
			table_name = table_list[0]
			ori_table_alias = drop_trailing_number(table_name)
			ori_table_name = reverse_alias[ori_table_alias]
			table_id = self.table2id[ori_table_name]
			card_scaling = self.table_size_list[ori_table_name]

			cdfs_on_table = databatch[5 * tid]
			signs_on_table = databatch[5 * tid + 1]
			qreps_on_table = databatch[5 * tid + 2]

			batch_masks = databatch[5 * tid + 3]
			batch_qreps_masks = databatch[5 * tid + 4]

			# batch_cdfs: [bs, num_normal_queries, num_cdfs, num_col]

			bs = cdfs_on_table.shape[0]
			num_normal_qs = cdfs_on_table.shape[1]
			num_cdfs = cdfs_on_table.shape[2]

			batch_masks = batch_masks.view(bs, num_normal_qs)

			if ori_table_alias != 'mii': # for mscn ce models
				if self.table_ce_models[table_id] is not None:
					cdf_est = self.table_ce_models[table_id](qreps_on_table, batch_qreps_masks)
					q_sel_est = cdf_est
				else:
					q_sel_est = torch.ones(bs, 1)
					if is_cuda:
						q_sel_est = q_sel_est.cuda()
			else:
				if self.table_cdf_models[table_id] is not None:
					cdfs_on_table = cdfs_on_table.view(-1, cdfs_on_table.shape[-1])
					cdf_est = self.table_cdf_models[table_id](cdfs_on_table)

					cdf_est = cdf_est.view(bs, num_normal_qs, -1)
					q_sel_est = torch.sum(cdf_est * signs_on_table, dim=-1)  # [bs, num_normal_queries];
					q_sel_est = torch.sum(q_sel_est * batch_masks, dim=-1, keepdim=True)  # [bs, 1];

				else:
					q_sel_est = torch.ones(bs, 1)
					if is_cuda:
						q_sel_est = q_sel_est.cuda()

			sel_result = q_sel_est
			return torch.squeeze(sel_result * card_scaling)
		else:
			###
			sel_result = {}
			lcs_result = {}

			bs = databatch[0].shape[0]
			curr_idx = 0

			for table_name in table_list:
				lcs_result[table_name] = {}

			### obtain base estimates
			for tid, table_name in enumerate(table_list):
				ori_table_alias = drop_trailing_number(table_name)
				ori_table_name = reverse_alias[ori_table_alias]
				table_id = self.table2id[ori_table_name]

				cdfs_on_table = databatch[5*tid]
				signs_on_table = databatch[5*tid+1]
				qreps_on_table = databatch[5*tid+2]

				batch_masks =  databatch[5*tid+3]
				batch_qreps_masks = databatch[5 * tid + 4]

				curr_idx = 5 * tid + 5

				# batch_cdfs: [bs, num_normal_queries, num_cdfs, num_col]

				bs = cdfs_on_table.shape[0]
				num_normal_qs = cdfs_on_table.shape[1]
				batch_masks = batch_masks.view(bs, num_normal_qs)

				if ori_table_alias != 'mii': # for mscn ce models
					query_groups = list(set(keys_order) & set(self.table2keygroups[ori_table_name]))
					jk_idxs = []
					for jk in query_groups:
						jk_idxs.append(list(self.join_keys_per_table[ori_table_name].keys()).index(jk))

					jk_preds_per_group, _ = self.table_unified_join_predictors[ori_table_name](qreps_on_table,batch_qreps_masks,jk_idxs) 
					for jk in query_groups:
						jk_id = list(self.join_keys_per_table[ori_table_name].keys()).index(jk)
						lcs_result[table_name][jk] = jk_preds_per_group[jk_id]

					# get ce estimates
					if self.table_ce_models[table_id] is not None:
						cdf_est = self.table_ce_models[table_id](qreps_on_table, batch_qreps_masks)
						q_sel_est = cdf_est
					else:
						q_sel_est = torch.ones(bs, 1)
						if is_cuda:
							q_sel_est = q_sel_est.cuda()

					sel_result[table_name] = q_sel_est

				else:
					query_groups = list(set(keys_order) & set(self.table2keygroups[ori_table_name]))
					jk_idxs = []
					for jk in query_groups:
						jk_idxs.append(list(self.join_keys_per_table[ori_table_name].keys()).index(jk))

					jk_preds_per_group, _ = self.table_unified_join_predictors[ori_table_name](qreps_on_table,batch_qreps_masks,jk_idxs) 
					for jk in query_groups:
						jk_id = list(self.join_keys_per_table[ori_table_name].keys()).index(jk)
						lcs_result[table_name][jk]  = jk_preds_per_group[jk_id]

					if self.table_cdf_models[table_id] is not None:
						input_cdfs_on_table = cdfs_on_table.view(-1, cdfs_on_table.shape[-1])
						cdf_est = self.table_cdf_models[table_id](input_cdfs_on_table)
						cdf_est = cdf_est.view(bs, num_normal_qs, -1)
						q_sel_est = torch.sum(cdf_est * signs_on_table, dim=-1)  # [bs, num_normal_queries];
						q_sel_est = torch.sum(q_sel_est * batch_masks, dim=-1, keepdim=True)  # [bs, 1];
					else:
						q_sel_est = torch.ones(bs, 1)
						if is_cuda:
							q_sel_est = q_sel_est.cuda()

					sel_result[table_name] = q_sel_est

			### obtain join estimates

			curr_card_pred = torch.ones(bs)
			if is_cuda:
				curr_card_pred = curr_card_pred.cuda()
			stb_q_mask = databatch[-2]

			for key_id, (key_group, tables) in enumerate(zip(keys_order, tables_order)):
				lcs_size = lcs_result[tables[0]][key_group].shape[-1]
				bin_sel_preds_prod = torch.ones(bs, lcs_size)
				sel_prod =  torch.ones(bs)

				if is_cuda:
					bin_sel_preds_prod = bin_sel_preds_prod.cuda()
					sel_prod = sel_prod.cuda()

				a_t_mask = databatch[curr_idx + 3*key_id]
				a_t_group_mask = databatch[curr_idx + 3*key_id+1]
				a_use_sels = databatch[curr_idx + 3*key_id+2]
			
				for t_id, table_name in enumerate(tables):
					ori_table_alias = drop_trailing_number(table_name)
					ori_table_name = reverse_alias[ori_table_alias]
					table_id = self.table2id[ori_table_name]

					table_sel = sel_result[table_name]
					table_sel = torch.squeeze(table_sel)
					bin_probs = lcs_result[table_name][key_group]

					card_scaling = torch.ones(bs)
					if is_cuda:
						card_scaling = card_scaling.cuda()

					card_scaling = torch.where(a_use_sels[:,t_id] == 1, card_scaling * self.table_size_list[ori_table_name], card_scaling)

					bin_sel_preds_prod = torch.where(a_t_mask[:,t_id].unsqueeze(1).repeat(1, lcs_size) == 1, 
									  bin_sel_preds_prod * bin_probs, bin_sel_preds_prod)


					sel_prod = torch.where(a_use_sels[:,t_id] == 1, sel_prod * table_sel, sel_prod) * card_scaling
					if print_details:
						print(table_name)

				all_ones = torch.ones(bs) 
				if is_cuda:
					all_ones = all_ones.cuda()

				bin_sum = torch.where(stb_q_mask == 1, all_ones, torch.sum(bin_sel_preds_prod, dim=-1))

				if print_details:
					print('bin_sum')

				curr_card_pred = torch.where(a_t_group_mask == 1, curr_card_pred * bin_sum * sel_prod, curr_card_pred)				
				if print_details:
					print('curr_card_pred')
		return curr_card_pred
	
	def start_train(self):
		"""
        Sets all models to training mode.
        """

		for cdf_model in self.table_cdf_models:
			if cdf_model is not None:
				cdf_model.train()
		for cdf_model in self.table_ce_models:
			if cdf_model is not None:
				cdf_model.train()
		for jk in self.table_unified_join_predictors:
			join_model = self.table_unified_join_predictors[jk]
			if join_model is not None:
				join_model.train()

	def load_models_to_gpu(self):
		"""
		Loads all models to the GPU if they are not None.
		This method iterates over three different collections of models:
		- `self.table_cdf_models`
		- `self.table_ce_models`
		- `self.table_unified_join_predictors`
		For each model in these collections, if the model is not None, it is moved to the GPU using the `cuda()` method.
		"""
		for cdf_model in self.table_cdf_models:
			if cdf_model is not None:
				cdf_model.cuda()

		for ce_model in self.table_ce_models:
			if ce_model is not None:
				ce_model.cuda()

		for jk in self.table_unified_join_predictors:
			join_model = self.table_unified_join_predictors[jk]
			if join_model is not None:
				join_model.cuda()

	def save_models(self, epoch_id, bs, lr, save_directory='./saved_models/grasp/'):
		"""
        Saves all models to disk.

        Args:
            epoch_id (int): Epoch identifier.
            bs (int): Batch size.
            lr (float): Learning rate.
            save_directory (str): Directory to save models.
        """

		info = "{}+{}+{}".format(epoch_id, bs, lr)
		if not os.path.exists(save_directory + info):
			os.makedirs(save_directory + info)

		save_directory = save_directory + info

		for idx, cdf_model in enumerate(self.table_cdf_models):
			if cdf_model is not None:
				torch.save(cdf_model, f"{save_directory}/cdf_model_{idx}.pt")
		
		for idx, ce_model in enumerate(self.table_ce_models):
			if ce_model is not None:
				torch.save(ce_model, f"{save_directory}/ce_model_{idx}.pt")
		
		for table_name in self.table_unified_join_predictors:
			join_model = self.table_unified_join_predictors[table_name]
			if join_model is not None:
				torch.save(join_model, f"{save_directory}/unified_join_model_{table_name}_{idx}.pt")

	def load_models_from_disk(self, epoch_id, bs, lr, save_directory='./saved_models/grasp/'):
		"""
        Loads all models from disk.

        Args:
            epoch_id (int): Epoch identifier.
            bs (int): Batch size.
            lr (float): Learning rate.
            save_directory (str): Directory to load models from.
        """

		info = "{}+{}+{}".format(epoch_id, bs, lr)
		save_directory = save_directory + info

		for idx in range(len(self.table_cdf_models)):
			model_path = f"{save_directory}/cdf_model_{idx}.pt"
			if os.path.exists(model_path):
				print('table_cdf_models loaded')
				self.table_cdf_models[idx] = torch.load(model_path,map_location=torch.device('cpu') )

		for idx in range(len(self.table_ce_models)):
			model_path = f"{save_directory}/ce_model_{idx}.pt"
			if os.path.exists(model_path):
				self.table_ce_models[idx] = torch.load(model_path,map_location=torch.device('cpu') )
	
		for table_name in self.table_unified_join_predictors:
			model_path = f"{save_directory}/unified_join_model_{table_name}_{idx}.pt"
			if os.path.exists(model_path):
				self.table_unified_join_predictors[table_name] = torch.load(model_path,map_location=torch.device('cpu') )

	def models_to_double(self):
		"""
        Converts all models to double precision.
        """
		for cdf_model in self.table_cdf_models:
			if cdf_model is not None:
				cdf_model.double()
		for ce_model in self.table_ce_models:
			if ce_model is not None:
				ce_model.double()
		for jk in self.table_unified_join_predictors:
			join_model = self.table_unified_join_predictors[jk]
			if join_model is not None:
				join_model.double()

	def start_eval(self):
		"""
        Sets all models to evaluation mode.
        """
		for cdf_model in self.table_cdf_models:
			if cdf_model is not None:
				cdf_model.eval()
		for ce_model in self.table_ce_models:
			if ce_model is not None:
				ce_model.eval()
		for jk in self.table_unified_join_predictors:
			join_model = self.table_unified_join_predictors[jk]
			if join_model is not None:
				join_model.eval()

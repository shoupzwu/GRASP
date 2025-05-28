import copy
import re
import itertools
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from GRASP.utils import *
from GRASP.ce_models import AutoregressiveCDF, NeuCDF, MSCNCE
from GRASP.lcs_models import UnifiedLCSPredictor
import torch.nn as nn
from dsb_utlities.NeuroCDF_helpers import *

OPS = ['<', '<=', '>', '>=', '=']

class JoinHandler():
	def __init__(self, table_dim_list, table_size_list, cdf_model_choice,
				  hidden_size=64, lcs_size=10):
		super().__init__()

		
		self.key_group = 'd'
		self.table_dim_list = table_dim_list
		self.table_size_list = table_size_list
		self.lcs_size = lcs_size
		self.cdf_model_choice = cdf_model_choice

		self.table_cdf_models = []
		self.table_ce_models = []
		self.table_unified_join_predictors = {}

		self.join_keys_per_table = {}
		self.table2keygroups = {}
		
		# init MSCN CE models
		for num_dim in table_dim_list:
			if num_dim > 0:
				self.table_ce_models.append(MSCNCE(num_dim + len(OPS) + 1, hidden_size))
			else:
				self.table_ce_models.append(None)

		if cdf_model_choice == 'neucdf':
			for num_dim in table_dim_list:
				if num_dim > 0:
					self.table_cdf_models.append(NeuCDF(num_dim, hidden_size, lcs_size))
				else:
					self.table_cdf_models.append(None)
		elif cdf_model_choice == 'arcdf':
			for num_dim in table_dim_list:
				if num_dim > 0:
					self.table_cdf_models.append(AutoregressiveCDF(num_dim, hidden_size, lcs_size))
				else:
					self.table_cdf_models.append(None)
		else:
			raise Exception("Sorry, model not supported")

		key_group_lcs_size = lcs_size	
		for t_id, num_dim in enumerate(table_dim_list):
			if t_id not in self.join_keys_per_table:
				self.join_keys_per_table[t_id] = {self.key_group: key_group_lcs_size}
			else:
				self.join_keys_per_table[t_id][self.key_group] = key_group_lcs_size

		for t_id in self.join_keys_per_table:
			num_dim = table_dim_list[t_id]
			output_sizes = []
			for jk in self.join_keys_per_table[t_id]:
				output_sizes.append(self.join_keys_per_table[t_id][jk])
			self.table_unified_join_predictors[t_id] = UnifiedLCSPredictor(num_dim + len(OPS) + 1, hidden_size, output_sizes)


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

	def decompose_query_to_bins(self, query_on_table, table_id):
		join_col = self.table_join_dim_list[table_id]
		bin_queries = []
		bin_points = np.linspace(0, 1, self.lcs_size + 1)

		for bin_id in range(self.lcs_size):
			query_on_table_copy = copy.deepcopy(query_on_table)
			query_on_table_copy.append([join_col, ">=", bin_points[bin_id]])
			query_on_table_copy.append([join_col, "<=", bin_points[bin_id+1]])
			bin_queries.append(query_on_table_copy)

		return bin_queries

	def separate_in_clase(self, table2predicates):
		join_queries = []
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
			join_queries.append(table_res_list)

		join_queries = list(itertools.product(*join_queries))
		join_queries = [list(comb) for comb in join_queries]

		table2predicates_res = []
		table_list = list(table2predicates.keys())
		for q in join_queries:
			q_dic = {}
			for table, col_res in zip(table_list, q):
				q_dic[table] = col_res
			table2predicates_res.append(q_dic)

		return table2predicates_res
	
	def load_training_queries(self, table2predicates_list, table2qreps_list, training_cards, bs=64, is_cuda=True, is_shuffle=True):
		#### list of join queries 
		#### batch_size * table2predicates_list

		table_list = list(table2predicates_list[0].keys())
		normal_qs_list = []
		qreps_list = []
		max_num_normal_queries = 0

		for join_query, qreps in zip(table2predicates_list, table2qreps_list):
			normal_qs = self.separate_in_clase(join_query)
			normal_qs_list.append(normal_qs)
			qreps_list.append(qreps)
			if len(normal_qs) > max_num_normal_queries:
				max_num_normal_queries = len(normal_qs)

		table2queries = {}
		table2cdfs = {}
		table2signs = {}
		table2intervals = {}
		table2reps = {}

		for table_name in table_list:
			table2queries[table_name] = []
			table2cdfs[table_name] = []
			table2reps[table_name] = []
			table2signs[table_name] = []
			table2intervals[table_name] = []

		for normal_qs, qreps in zip(normal_qs_list, qreps_list):
			for table_name in table_list:
				res_per_join_query = []
				for normal_q in normal_qs:
					### all normal_q have the same context_on_table
					query_on_table = normal_q[table_name]
					res_per_join_query.append(query_on_table)

				table2queries[table_name].append(res_per_join_query)
				table2reps[table_name].append(qreps[table_name])

		dataloader_list = []
		training_cards = torch.from_numpy(np.array(training_cards))
		if is_cuda:
			training_cards = training_cards.cuda()

		for table_name in table_list:

			batch_queries = table2queries[table_name]
			batch_reps =  table2reps[table_name]

			num_cols = self.table_dim_list[table_name]

			batch_cdfs, batch_signs, batch_masks = multi_queries_batch_query2cdfs(batch_queries, num_cols)

			batch_intervals = multi_batch_query2interval(batch_queries, num_cols)

			batch_reps, batch_rep_masks = multi_batch_query2reps(batch_reps, OPS, num_cols, num_cols+len(OPS)+1)

			batch_cdfs = torch.from_numpy(batch_cdfs)
			batch_signs = torch.from_numpy(batch_signs)
			batch_masks = torch.from_numpy(batch_masks)

			if is_cuda:
				batch_cdfs = batch_cdfs.cuda()
				batch_signs = batch_signs.cuda()
				batch_masks = batch_masks.cuda()
				batch_intervals = batch_intervals.cuda()
				batch_reps = batch_reps.cuda()

				batch_rep_masks = batch_rep_masks.cuda()


			dataloader_list.extend([batch_cdfs, batch_signs, batch_intervals, batch_reps,
									batch_masks, batch_rep_masks])
			
		dataloader_list.append(training_cards)

		dataloader_list = TensorDataset(*dataloader_list)
		dataloader = DataLoader(dataloader_list, batch_size=bs, shuffle=is_shuffle)

		return dataloader, table_list

	def batch_estimate_join_queries_from_loader(self, databatch, table_list, is_cuda=True):
		#### list of join queries
		#### batch_size * table2predicates_list

		keys_order = ['d']
		tables_order = [table_list]

		batch_card_sum = self.batch_estimate_normal_join_queries_from_loader(databatch, table_list,
																			 keys_order, tables_order, is_cuda=is_cuda)

		return batch_card_sum


	def batch_estimate_normal_join_queries_from_loader(self, databatch, table_list, keys_order, tables_order, is_cuda=True):
		if len(table_list) == 1:
			# single table query
			table_name = table_list[0]
			card_scaling = self.table_size_list[table_name]
		
			cdfs_on_table = databatch[0]
			signs_on_table = databatch[1]
			intervals_on_table = databatch[2]
			reps_on_table = databatch[3]
		
			batch_masks = databatch[4]
			batch_rep_masks = databatch[5]

			bs = cdfs_on_table.shape[0]
			num_normal_qs = cdfs_on_table.shape[1]
			num_cdfs = cdfs_on_table.shape[2]

			if self.table_cdf_models[table_name] is not None:
				cdfs_on_table = cdfs_on_table.view(-1, cdfs_on_table.shape[-1])
				cdf_est = self.table_cdf_models[table_name](cdfs_on_table)

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
			for table_name in table_list:
				lcs_result[table_name] = {}
			### obtain base estimates
			for tid, table_name in enumerate(table_list):
				cdfs_on_table = databatch[6*tid]
				signs_on_table = databatch[6*tid+1]
				intervals_on_table = databatch[6*tid+2]
				batch_reps = databatch[6*tid+3]

				batch_masks =  databatch[6*tid+4]
				batch_rep_masks = databatch[6*tid+5]

				bs = cdfs_on_table.shape[0]
				num_normal_qs = cdfs_on_table.shape[1]

				batch_masks = batch_masks.view(bs, num_normal_qs)
				query_groups = ['d']
				jk_idxs = []
				for jk in query_groups:
					jk_idxs.append(list(self.join_keys_per_table[table_name].keys()).index(jk))

				intervals_on_table = torch.squeeze(intervals_on_table)
				jk_preds_per_group, _ = self.table_unified_join_predictors[table_name](batch_reps, batch_rep_masks, output_indices = jk_idxs) 
				for jk in query_groups:
					jk_id = list(self.join_keys_per_table[table_name].keys()).index(jk)
					lcs_result[table_name][jk] = jk_preds_per_group[jk_id]

				if self.table_ce_models[table_name] is not None:
					cdf_est = self.table_ce_models[table_name](batch_reps, batch_rep_masks)
					q_sel_est = cdf_est
				else:
					q_sel_est = torch.ones(bs, 1)
					if is_cuda:
						q_sel_est = q_sel_est.cuda()

				sel_result[table_name] = q_sel_est

			### obtain join estimates
			current_connected_tables = []
			curr_card_pred = torch.ones(bs)
			if is_cuda:
				curr_card_pred = curr_card_pred.cuda()

			for key_id, (key_group, tables) in enumerate(zip(keys_order, tables_order)):
				lcs_size = lcs_result[tables[0]][key_group].shape[-1]
				bin_sel_preds_prod = torch.ones(bs, lcs_size)
				sel_prod =  torch.ones(bs)
				
				if is_cuda:
					bin_sel_preds_prod = bin_sel_preds_prod.cuda()
					sel_prod = sel_prod.cuda()
				for t_id, table_name in enumerate(tables):
					table_id = table_name

					table_sel = sel_result[table_name]
					table_sel = torch.squeeze(table_sel)
					bin_probs = lcs_result[table_name][key_group]
					bin_probs = torch.squeeze(bin_probs)
				
					if table_name not in current_connected_tables:
						card_scaling =  self.table_size_list[table_id]
						sel_prod = sel_prod * table_sel * card_scaling
						current_connected_tables.append(table_name)

					bin_sel_preds_prod = bin_sel_preds_prod * bin_probs
		
				bin_sum =  torch.sum(bin_sel_preds_prod, dim=-1)
				curr_card_pred = curr_card_pred * bin_sum * sel_prod

		return curr_card_pred

	def start_train(self):
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
		for cdf_model in self.table_cdf_models:
			if cdf_model is not None:
				cdf_model.cuda()
		for cdf_model in self.table_ce_models:
			if cdf_model is not None:
				cdf_model.cuda()
		for jk in self.table_unified_join_predictors:
			join_model = self.table_unified_join_predictors[jk]
			if join_model is not None:
				join_model.cuda()

	def save_models(self, epoch_id, bs, lr, lcs_size, save_directory='./saved_models/grasp/'):
		# Ensure the save directory exists
		info = "{}+{}+{}+{}".format(epoch_id, bs, lr, lcs_size)
		if not os.path.exists(save_directory + info):
			os.makedirs(save_directory + info)

		save_directory = save_directory + info

		for idx, cdf_model in enumerate(self.table_cdf_models):
			if cdf_model is not None:
				torch.save(cdf_model, f"{save_directory}/cdf_model_{idx}.pt")
		
		for table_name in self.table_unified_join_predictors:
			join_model = self.table_unified_join_predictors[table_name]
			if join_model is not None:
				torch.save(join_model, f"{save_directory}/unified_join_model_{table_name}_{idx}.pt")

	def load_models_from_disk(self, epoch_id, load_directory='./saved_models/'):
		load_directory = load_directory + str(epoch_id)

		for idx in range(len(self.table_cdf_models)):
			model_path = f"{load_directory}/cdf_model_{idx}.pt"
			if os.path.exists(model_path):
				self.table_cdf_models[idx] = torch.load(model_path,map_location=torch.device('cpu') )
	
		for table_name in self.table_unified_join_predictors:
			model_path = f"{load_directory}/unified_join_model_{table_name}_{idx}.pt"
			if os.path.exists(model_path):
				self.table_unified_join_predictors[table_name] = torch.load(model_path,map_location=torch.device('cpu') )

	def models_to_double(self):
		for cdf_model in self.table_cdf_models:
			if cdf_model is not None:
				cdf_model.double()
		for cdf_model in self.table_ce_models:
			if cdf_model is not None:
				cdf_model.double()
		for jk in self.table_unified_join_predictors:
			join_model = self.table_unified_join_predictors[jk]
			if join_model is not None:
				join_model.double()

	def start_eval(self):
		for cdf_model in self.table_cdf_models:
			if cdf_model is not None:
				cdf_model.eval()
		for cdf_model in self.table_ce_models:
			if cdf_model is not None:
				cdf_model.eval()
		for jk in self.table_unified_join_predictors:
			join_model = self.table_unified_join_predictors[jk]
			if join_model is not None:
				join_model.eval()
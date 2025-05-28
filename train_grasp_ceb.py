import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import numpy as np
import torch

import random

import os

from CEB_utlities.join_utilits_CEB_job import *
from GRASP.join_handler_ceb import *
from utils import *
from torch.nn.utils import clip_grad_norm_

OPS = ['lt', 'eq', 'in', 'like']


random.seed(42)

is_cuda = torch.cuda.is_available()

IMDB_DIRECTORY = "/home/imdb/"	# directory to save the IMDB dataset
PROCESSED_WORKLOAD_DIRECTORY = "/home/processed_workloads/" # directory to save processed workloads

def train_grasp(epoch=100, feature_dim=256, lcs_dim=500, bs=128, lr=5e-4):
	num_q = 3000
	cdf_model_choice = 'arcdf'


	template_list =  ['1a', '2a', '2b', '2c', '3a', '3b', '4a', '5a', '6a', 
					'7a', '8a', '9a', '9b', '10a', '11a', '11b']


	directory_list = [IMDB_DIRECTORY + temp_name + "/" for temp_name in template_list]


	res_file = open("./grasp_{}_ratio_{}_lcsszie_{}_lr_{}_bs_{}.txt".format(",".join(template_list), sub_templates_in_training_ratio, lcs_dim, lr, bs), 'a')


	(table_list, table_dim_list, table_like_dim_list, table_sizes, table_key_groups,
	table_join_keys, table_text_cols, table_normal_cols, col_type, col2minmax) = get_table_info()

	template2queries, template2cards, _, test_template2queries, test_template2cards, _ , colid2featlen_per_table = read_query_file_batched(col2minmax, num_q=num_q, directory_list=directory_list
																																	,saved_ditectory=PROCESSED_WORKLOAD_DIRECTORY)
	t2ops = get_table_to_ops(template2queries, test_template2queries)

	print(t2ops)
	for t in t2ops:
		print(t)
		print(t2ops[t])

	join_handler = JoinHandler(table_list=table_list, table_dim_list=table_dim_list, table_key_groups=table_key_groups,
							table_size_list=table_sizes, cdf_model_choice=cdf_model_choice, hidden_size=feature_dim, lcs_size=lcs_dim, colid2featlen_per_table=colid2featlen_per_table)

	join_handler.init_ce_predictors(colid2featlen_per_table)
	join_handler.init_unified_lcs_predictors()
	join_handler.models_to_double()
	if is_cuda:
		join_handler.load_models_to_gpu()
	optimizer = torch.optim.Adam(join_handler.get_parameters(), lr=lr)

	training_loader_per_parent_template = {}
	in_test_loader_per_parent_template = {}
	ood_test_loader_per_parent_template = {}

	num_qs = 0
	test_num_qs = 0
	num_seen_templates = 0

	### load seen join templates
	for template in template2queries:
		num_seen_templates += 1
		training_queries = template2queries[template]
		all_training_cards = template2cards[template]

		if len(all_training_cards) > 100:
			num_train_per_temp = len(template2queries[template]) - 100

			num_qs += num_train_per_temp
			# for training
			for qid, (q, q_context, q_reps, parent_key_groups, key_groups_per_q, parent_table_tuple, table_tuple, table_mask, table_group_mask) in enumerate(training_queries[:num_train_per_temp]):
				parent_t_list = tuple(table_tuple)

				if parent_t_list not in training_loader_per_parent_template:
					training_loader_per_parent_template[parent_t_list] = {}
					training_loader_per_parent_template[parent_t_list]['qs'] = []
					training_loader_per_parent_template[parent_t_list]['q_contexts'] = []
					training_loader_per_parent_template[parent_t_list]['qreps'] = []
					training_loader_per_parent_template[parent_t_list]['table_masks'] = []
					training_loader_per_parent_template[parent_t_list]['table_group_masks'] = []
					training_loader_per_parent_template[parent_t_list]['is_stb'] = []
					training_loader_per_parent_template[parent_t_list]['parent_t_list'] = []
					training_loader_per_parent_template[parent_t_list]['key_groups'] = []
					training_loader_per_parent_template[parent_t_list]['cards'] = []
				
				training_loader_per_parent_template[parent_t_list]['qs'].append(q)
				training_loader_per_parent_template[parent_t_list]['q_contexts'].append(q_context)
				training_loader_per_parent_template[parent_t_list]['qreps'].append(q_reps)
				training_loader_per_parent_template[parent_t_list]['table_masks'].append(table_mask)
				training_loader_per_parent_template[parent_t_list]['table_group_masks'].append(table_group_mask)
				if len(list(table_tuple)) > 1:
					training_loader_per_parent_template[parent_t_list]['is_stb'].append(0)
				else:
					training_loader_per_parent_template[parent_t_list]['is_stb'].append(1)
				training_loader_per_parent_template[parent_t_list]['parent_t_list'].append(table_tuple)
				training_loader_per_parent_template[parent_t_list]['key_groups'].append(key_groups_per_q)
				training_loader_per_parent_template[parent_t_list]['cards'].append(all_training_cards[qid])


			for qid, (q, q_context, q_reps, parent_key_groups, key_groups_per_q, parent_table_tuple, table_tuple, table_mask, table_group_mask) in enumerate(training_queries[-100:]):
				parent_t_list = tuple(parent_table_tuple)
				if parent_t_list not in in_test_loader_per_parent_template:
					in_test_loader_per_parent_template[parent_t_list] = {}
					in_test_loader_per_parent_template[parent_t_list]['qs'] = []
					in_test_loader_per_parent_template[parent_t_list]['q_contexts'] = []
					in_test_loader_per_parent_template[parent_t_list]['qreps'] = []
					in_test_loader_per_parent_template[parent_t_list]['table_masks'] = []
					in_test_loader_per_parent_template[parent_t_list]['table_group_masks'] = []
					in_test_loader_per_parent_template[parent_t_list]['is_stb'] = []
					in_test_loader_per_parent_template[parent_t_list]['parent_t_list'] = []
					in_test_loader_per_parent_template[parent_t_list]['key_groups'] = []
					in_test_loader_per_parent_template[parent_t_list]['cards'] = []

				
				in_test_loader_per_parent_template[parent_t_list]['qs'].append(q)
				in_test_loader_per_parent_template[parent_t_list]['q_contexts'].append(q_context)
				in_test_loader_per_parent_template[parent_t_list]['qreps'].append(q_reps)
				in_test_loader_per_parent_template[parent_t_list]['table_masks'].append(table_mask)
				in_test_loader_per_parent_template[parent_t_list]['table_group_masks'].append(table_group_mask)
				if len(list(table_tuple)) > 1:
					in_test_loader_per_parent_template[parent_t_list]['is_stb'].append(0)
				else:
					in_test_loader_per_parent_template[parent_t_list]['is_stb'].append(1)
				in_test_loader_per_parent_template[parent_t_list]['parent_t_list'].append(parent_table_tuple)
				in_test_loader_per_parent_template[parent_t_list]['key_groups'].append(parent_key_groups)
				in_test_loader_per_parent_template[parent_t_list]['cards'].append(all_training_cards[num_train_per_temp+qid])

		else:
			# for training
			num_qs += len(training_queries)
			for qid, (q, q_context, q_reps, parent_key_groups, key_groups_per_q, parent_table_tuple, table_tuple, table_mask, table_group_mask) in enumerate(training_queries):
				
				parent_t_list = tuple(table_tuple)

				if parent_t_list not in training_loader_per_parent_template:
					training_loader_per_parent_template[parent_t_list] = {}
					training_loader_per_parent_template[parent_t_list]['qs'] = []
					training_loader_per_parent_template[parent_t_list]['q_contexts'] = []
					training_loader_per_parent_template[parent_t_list]['qreps'] = []
					training_loader_per_parent_template[parent_t_list]['table_masks'] = []
					training_loader_per_parent_template[parent_t_list]['table_group_masks'] = []
					training_loader_per_parent_template[parent_t_list]['is_stb'] = []
					training_loader_per_parent_template[parent_t_list]['parent_t_list'] = []
					training_loader_per_parent_template[parent_t_list]['key_groups'] = []
					training_loader_per_parent_template[parent_t_list]['cards'] = []
				
				training_loader_per_parent_template[parent_t_list]['qs'].append(q)
				training_loader_per_parent_template[parent_t_list]['q_contexts'].append(q_context)
				training_loader_per_parent_template[parent_t_list]['qreps'].append(q_reps)
				training_loader_per_parent_template[parent_t_list]['table_masks'].append(table_mask)
				training_loader_per_parent_template[parent_t_list]['table_group_masks'].append(table_group_mask)
				if len(list(table_tuple)) > 1:
					training_loader_per_parent_template[parent_t_list]['is_stb'].append(0)
				else:
					training_loader_per_parent_template[parent_t_list]['is_stb'].append(1)
				training_loader_per_parent_template[parent_t_list]['parent_t_list'].append(parent_table_tuple)
				training_loader_per_parent_template[parent_t_list]['key_groups'].append(key_groups_per_q)
				training_loader_per_parent_template[parent_t_list]['cards'].append(all_training_cards[qid])

	### load unseen queries
	for template in test_template2queries:
		if len(test_template2queries[template]) < 10:
			continue

		test_num_qs += len(test_template2queries[template])

		test_queries = test_template2queries[template]
		test_cards = test_template2cards[template]

		test_queries = list(test_queries)[:100]
		test_cards = list(test_cards)[:100]

		for qid, (q, q_context, q_reps, parent_key_groups, key_groups_per_q, parent_table_tuple, table_tuple, table_mask, table_group_mask) in enumerate(test_queries):
			parent_t_list = tuple(parent_table_tuple)

			if parent_t_list not in ood_test_loader_per_parent_template:
				ood_test_loader_per_parent_template[parent_t_list] = {}
				ood_test_loader_per_parent_template[parent_t_list]['qs'] = []
				ood_test_loader_per_parent_template[parent_t_list]['q_contexts'] = []
				ood_test_loader_per_parent_template[parent_t_list]['qreps'] = []
				ood_test_loader_per_parent_template[parent_t_list]['table_masks'] = []
				ood_test_loader_per_parent_template[parent_t_list]['table_group_masks'] = []
				ood_test_loader_per_parent_template[parent_t_list]['is_stb'] = []
				ood_test_loader_per_parent_template[parent_t_list]['parent_t_list'] = []
				ood_test_loader_per_parent_template[parent_t_list]['key_groups'] = []
				ood_test_loader_per_parent_template[parent_t_list]['cards'] = []
			
			ood_test_loader_per_parent_template[parent_t_list]['qs'].append(q)
			ood_test_loader_per_parent_template[parent_t_list]['q_contexts'].append(q_context)
			ood_test_loader_per_parent_template[parent_t_list]['qreps'].append(q_reps)
			ood_test_loader_per_parent_template[parent_t_list]['table_masks'].append(table_mask)
			ood_test_loader_per_parent_template[parent_t_list]['table_group_masks'].append(table_group_mask)
			if len(list(table_tuple)) > 1:
				ood_test_loader_per_parent_template[parent_t_list]['is_stb'].append(0)
			else:
				ood_test_loader_per_parent_template[parent_t_list]['is_stb'].append(1)
			ood_test_loader_per_parent_template[parent_t_list]['parent_t_list'].append(parent_table_tuple)
			ood_test_loader_per_parent_template[parent_t_list]['key_groups'].append(parent_key_groups)
			ood_test_loader_per_parent_template[parent_t_list]['cards'].append(test_cards[qid])

	num_batches = math.ceil(num_qs / bs)

	print("total number of queries: {}".format(num_qs))
	print("total number of seen join templates: {}".format(num_seen_templates))

	res_file.write("total number of queries: {} \n".format(num_qs))
	res_file.write("total number of seen join templates: {}".format(num_seen_templates))

	#### start loading data for pytorch training

	data_loader_list = []
	data_loader_iter_list = []
	train_key_groups_list = []
	train_tables_list = []

	in_test_data_loader_list = []
	in_test_parent_alias_list = []
	in_test_training_keys_list = []
	in_test_tables_list = []

	ood_test_data_loader_list = []
	ood_test_parent_alias_list = []
	ood_test_training_keys_list = []
	ood_test_tables_list = []

	for t_list in training_loader_per_parent_template:
		tmp_object = training_loader_per_parent_template[t_list]
		table_data_loader, table_list =  join_handler.load_training_queries(tmp_object['qs'], 
																			tmp_object['q_contexts'],
																			tmp_object['qreps'], 
																			tmp_object['cards'], 
																			bs, t_list,  is_cuda=is_cuda)
		
		data_loader_iterator = iter(table_data_loader)

		data_loader_list.append(table_data_loader)
		data_loader_iter_list.append(data_loader_iterator)
		train_key_groups_list.append(tmp_object['key_groups'][0])
		train_tables_list.append(t_list)

	for parent_t_list in in_test_loader_per_parent_template:
		tmp_object = in_test_loader_per_parent_template[parent_t_list]

		table_data_loader, parent_alias_list, training_keys, training_tables = join_handler.load_training_queries_w_masks(tmp_object['qs'], 
																													tmp_object['q_contexts'],
																													tmp_object['qreps'], 
																													tmp_object['key_groups'][0], 
																													tmp_object['table_masks'],
																													tmp_object['table_group_masks'], 
																													tmp_object['is_stb'], 
																													tmp_object['cards'], 
																													3000, is_cuda=is_cuda)
		

		in_test_data_loader_list.append(table_data_loader)
		in_test_parent_alias_list.append(parent_alias_list)
		in_test_training_keys_list.append(training_keys)
		in_test_tables_list.append(training_tables)

	for parent_t_list in ood_test_loader_per_parent_template:
		tmp_object = ood_test_loader_per_parent_template[parent_t_list]

		table_data_loader, parent_alias_list, training_keys, training_tables = join_handler.load_training_queries_w_masks(tmp_object['qs'], 
																													tmp_object['q_contexts'],
																													tmp_object['qreps'], 
																													tmp_object['key_groups'][0], 
																													tmp_object['table_masks'],
																													tmp_object['table_group_masks'], 
																													tmp_object['is_stb'], 
																													tmp_object['cards'], 
																													3000, is_cuda=is_cuda)
		

		ood_test_data_loader_list.append(table_data_loader)
		ood_test_parent_alias_list.append(parent_alias_list)
		ood_test_training_keys_list.append(training_keys)
		ood_test_tables_list.append(training_tables)

	############ finish data loading

	for epoch_id in range(epoch):
		join_handler.start_train()

		accu_loss_total = 0.
		
		current_data_loader_ids = list(range(len(data_loader_iter_list)))
		while len(current_data_loader_ids) > 0:
			d_loader_id = random.choice(current_data_loader_ids)

			table_data_loader = data_loader_iter_list[d_loader_id]
			table_list = train_tables_list[d_loader_id]
			key_groups = train_key_groups_list[d_loader_id]

			try:
				# Fetch the next batch
				batch = next(table_data_loader)
				keys_order, tables_order = join_handler.template_to_group_order(key_groups)
				est_cards = join_handler.batch_estimate_join_queries_from_loader(batch, table_list,
																				keys_order, tables_order, is_cuda=is_cuda)
				if not est_cards.requires_grad:
					continue
				batch_cards = batch[-1]
				batch_cards = batch_cards.to(torch.float64)
				optimizer.zero_grad()

				adj_est_cards = torch.where(est_cards > 0, est_cards, 1.)

				sle_loss = torch.square(torch.log(torch.squeeze(adj_est_cards)) - torch.log(batch_cards))
				se_loss = torch.square(torch.squeeze(est_cards) - batch_cards)

				ultimate_loss = torch.where(torch.squeeze(est_cards) > 0, sle_loss, se_loss)
				total_loss = torch.mean(ultimate_loss)

				accu_loss_total += total_loss.item()

				with torch.autograd.set_detect_anomaly(True):
					total_loss.backward()
					clip_grad_norm_(join_handler.get_parameters(), 10.)
					optimizer.step()	

			except StopIteration:
				# When the iterator is exhausted, we get here
				current_data_loader_ids.remove(d_loader_id)
				data_loader_iter_list[d_loader_id] = iter(data_loader_list[d_loader_id])

		print("epoch: {}; loss: {}".format(epoch_id, accu_loss_total / num_batches))

		if epoch_id > 1:
			join_handler.start_eval()
			with torch.no_grad():
				all_est_cards = []
				all_true_cards = []

				for table_data_loader, test_parent_alias, test_keys, test_tables in zip(in_test_data_loader_list, in_test_parent_alias_list, in_test_training_keys_list, in_test_tables_list):
					for batch in table_data_loader:
						est_cards = join_handler.batch_estimate_join_queries_from_loader_w_mask(batch, test_parent_alias, test_keys, test_tables,  is_cuda=is_cuda)
						if est_cards is not None:
							batch_cards = batch[-1]
							est_cards = torch.where(est_cards > 1, est_cards, 1.)
							all_est_cards.extend(est_cards.cpu().detach())
							all_true_cards.extend(batch_cards.cpu().detach())
				q_errors = get_join_qerror(all_est_cards, all_true_cards, "seen", res_file, epoch_id)

				### unseen join templates 
				all_est_cards = []
				all_true_cards = []
				for table_data_loader, test_parent_alias, test_keys, test_tables in zip(ood_test_data_loader_list, ood_test_parent_alias_list, ood_test_training_keys_list, ood_test_tables_list):
					for batch in table_data_loader:
						est_cards = join_handler.batch_estimate_join_queries_from_loader_w_mask(batch, test_parent_alias, test_keys, test_tables, is_cuda=is_cuda)
						if est_cards is not None:
							batch_cards = batch[-1]
							est_cards = torch.where(est_cards > 1, est_cards, 1.)
							all_est_cards.extend(est_cards.cpu().detach())
							all_true_cards.extend(batch_cards.cpu().detach())
				q_errors = get_join_qerror(all_est_cards, all_true_cards, "unseen", res_file, epoch_id)


def main():
	"""
	Main function to parse command-line arguments and initiate the training process.
	Arguments:
		--epochs (int): Number of epochs for training (default: 100).
		--feature_dim (int): Dimension of hidden layer of CE model (default: 256).
		--lcs_dim (int): Dimension of the latent space of LCS model (default: 500).
		--bs (int): Batch size for training (default: 128).
	The function checks if CUDA is available and then calls the train_grasp function
	with the parsed arguments.
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", help="number of epochs (default: 100)", type=int, default=100)
	parser.add_argument("--feature_dim", help="dimension of hidden layer of CE model (default: 256)", type=int, default=256)
	parser.add_argument("--lcs_dim", help="Dimension of the latent space of LCS model (default: 500)", type=int, default=500)
	parser.add_argument("--bs", help="batch size (default: 128)", type=int, default=128)
	args = parser.parse_args()

	is_cuda = torch.cuda.is_available()

	train_grasp(args.epochs, args.feature_dim, args.lcs_dim, args.bs)

if __name__ == "__main__":
	main()
import math
import os
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random

import torch
from tpce_utlities.join_utilits_tpce import *
from GRASP.join_handler_tpce import *
from utils import *
from torch.nn.utils import clip_grad_norm_
random.seed(0)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
is_cuda = torch.cuda.is_available()

def train_grasp(epoch=100, feature_dim=256, lcs_dim=200, bs=128, lr=5e-4):

	(table_list, table_dim_list, table_like_dim_list, table_sizes, table_key_groups,
	table_join_keys, table_text_cols, table_normal_cols, col_type, col2minmax) = get_table_info()

	template2queries, template2cards, test_template2queries, test_template2cards, _, _ , _, _, table2ops = read_query_file(col2minmax)

	cdf_model_choice = 'arcdf'


	join_handler = JoinHandler(table_list=table_list, table_dim_list=table_dim_list, table_key_groups=table_key_groups,
								table_size_list=table_sizes, cdf_model_choice=cdf_model_choice, hidden_size=feature_dim, lcs_dim=lcs_dim)

	join_handler.models_to_double()
	if is_cuda:
			join_handler.load_models_to_gpu()

	optimizer = torch.optim.Adam(join_handler.get_parameters(), lr=lr)

	data_loader_list = []
	table_list_list = []
	key_groups_list = []
	data_loader_iter_list = []
	train_key_groups_list = []
	train_tables_list = [] 

	in_test_data_loader_list = []
	in_test_table_list_list = []
	in_test_key_groups_list = []

	test_data_loader_list = []
	test_table_list_list = []
	test_key_groups_list = []

	num_qs = 0
	test_num_qs = 0

	training_templates_order = list(template2queries.keys())
	training_templates_order = sorted(training_templates_order, key=len)

	seen_temps = 0
	in_test_temps = 0

	for template in training_templates_order:
		seen_temps += 1
		if len(template2queries[template]) > 50:
			in_test_temps += 1
			num_train_per_temp = len(template2queries[template]) - 50
			num_qs += num_train_per_temp

			training_qs = []
			training_qreps = []
			key_groups = None

			training_queries = template2queries[template][:num_train_per_temp]
			training_cards = template2cards[template][:num_train_per_temp]

			for (q, q_context, q_reps, q_localreps, key_groups_per_q, _, table_tuple) in training_queries:
				training_qs.append(q)
				training_qreps.append(q_localreps)
				key_groups = key_groups_per_q

			table_data_loader, table_list = join_handler.load_training_queries(training_qs, training_qreps, training_cards, bs, is_cuda=is_cuda)

			data_loader_list.append(table_data_loader)
			table_list_list.append(table_list)
			key_groups_list.append(key_groups)

			data_loader_iterator = iter(table_data_loader)
			data_loader_iter_list.append(data_loader_iterator)
			train_key_groups_list.append(key_groups)
			train_tables_list.append(table_list)

			### for in-dist test
			training_qs = []
			training_qreps = []
			key_groups = None

			training_queries = template2queries[template][num_train_per_temp:]
			training_cards = template2cards[template][num_train_per_temp:]

			for (q, q_context, q_reps, q_localreps, key_groups_per_q, mscn_joins, table_tuple) in training_queries:
				training_qs.append(q)
				training_qreps.append(q_localreps)
				key_groups = key_groups_per_q

			table_data_loader, table_list = join_handler.load_training_queries(training_qs, training_qreps, training_cards, bs, is_cuda=is_cuda)

			in_test_data_loader_list.append(table_data_loader)
			in_test_table_list_list.append(table_list)
			in_test_key_groups_list.append(key_groups)
		else:
			num_train_per_temp = len(template2queries[template])
			num_qs += num_train_per_temp

			training_qs = []
			training_qreps = []
			key_groups = None

			training_queries = template2queries[template][:num_train_per_temp]
			training_cards = template2cards[template][:num_train_per_temp]

			for (q, q_context, q_reps, q_localreps, key_groups_per_q, mscn_joins, table_tuple) in training_queries:
				training_qs.append(q)
				training_qreps.append(q_localreps)
				key_groups = key_groups_per_q

			table_data_loader, table_list = join_handler.load_training_queries(training_qs, training_qreps, training_cards, bs, is_cuda=is_cuda)

			data_loader_list.append(table_data_loader)
			table_list_list.append(table_list)
			key_groups_list.append(key_groups)

			data_loader_iterator = iter(table_data_loader)
			data_loader_iter_list.append(data_loader_iterator)
			train_key_groups_list.append(key_groups)
			train_tables_list.append(table_list)

	random_templates = list(test_template2queries.keys())

	for template in random_templates:

		if len(test_template2queries[template]) < 10:
			continue

		test_num_qs += len(test_template2queries[template])

		test_qs = []
		test_qreps = []
		key_groups = None

		test_queries = test_template2queries[template]
		test_cards = test_template2cards[template]

		combined = list(zip(test_queries, test_cards))
		random.shuffle(combined)
		test_queries, test_cards = zip(*combined)

		test_queries = list(test_queries)[:100]
		test_cards = list(test_cards)[:100]

		for (q, q_context, q_reps, q_localreps, key_groups_per_q, mscn_joins, table_tuple) in test_queries:
			test_qs.append(q)
			test_qreps.append(q_localreps)
			key_groups = key_groups_per_q


		table_data_loader, table_list = join_handler.load_training_queries(test_qs, test_qreps, test_cards, bs, is_cuda=is_cuda)
		

		test_data_loader_list.append(table_data_loader)
		test_table_list_list.append(table_list)
		test_key_groups_list.append(key_groups)


	num_batches = math.ceil(num_qs / bs)

	print("num_qs: {}".format(num_qs))
	print("seen templates: {}".format(seen_temps))
	print("in test tempates: {}".format(in_test_temps))

	res_file = open("tpce-grasp-lcs_size_{}_lr_{}_bs_{}.txt".format(lcs_dim, lr, bs), 'a')

	for epoch_id in range(epoch):
		join_handler.start_train()

		accu_loss_total = 0.
		cdf_valid_loss = 0.

		current_data_loader_ids = list(range(len(data_loader_iter_list)))
		while len(current_data_loader_ids) > 0:
			d_loader_id = random.choice(current_data_loader_ids)

			table_data_loader = data_loader_iter_list[d_loader_id]
			table_list = train_tables_list[d_loader_id]
			key_groups = train_key_groups_list[d_loader_id]

			try:
				# Fetch the next batch
				batch = next(table_data_loader)
				est_cards = join_handler.batch_estimate_join_queries_from_loader(batch, table_list, key_groups, is_cuda=is_cuda)
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


		if epoch_id > 3:
			all_est_cards = []
			all_true_cards = []
			join_handler.start_eval()
			for table_data_loader, table_list, key_groups in zip(in_test_data_loader_list, in_test_table_list_list, in_test_key_groups_list):
				for batch in table_data_loader:
					est_cards = join_handler.batch_estimate_join_queries_from_loader(batch, table_list, key_groups, is_cuda=is_cuda)
					if est_cards is not None:
						batch_cards = batch[-1]
						batch_cards = batch_cards.to(torch.float64)

						est_cards = torch.where(est_cards > 1, est_cards, 1.)
						all_est_cards.extend(est_cards.cpu().detach())
						all_true_cards.extend(batch_cards.cpu().detach())
			q_errors = get_join_qerror(all_est_cards, all_true_cards, workload_type="in", res_file=res_file, epoch_id=epoch_id)

			#### unseen templates
			all_est_cards = []
			all_true_cards = []
			for table_data_loader, table_list, key_groups in zip(test_data_loader_list, test_table_list_list, test_key_groups_list):
				for batch in table_data_loader:
					est_cards = join_handler.batch_estimate_join_queries_from_loader(batch, table_list, key_groups, is_cuda=is_cuda)
					if est_cards is not None:
						batch_cards = batch[-1]
						batch_cards = batch_cards.to(torch.float64)

						est_cards = torch.where(est_cards > 1, est_cards, 1.)
						all_est_cards.extend(est_cards.cpu().detach())
						all_true_cards.extend(batch_cards.cpu().detach())
			q_errors = get_join_qerror(all_est_cards, all_true_cards, workload_type="ood", res_file=res_file, epoch_id=epoch_id)

		print("epoch: {}; loss: {}".format(epoch_id, accu_loss_total / num_batches))


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
	parser.add_argument("--lcs_dim", help="Dimension of the latent space of LCS model (default: 500)", type=int, default=200)
	parser.add_argument("--bs", help="batch size (default: 128)", type=int, default=128)
	args = parser.parse_args()

	train_grasp(args.epochs, args.feature_dim, args.lcs_dim, args.bs)

if __name__ == "__main__":
	main()
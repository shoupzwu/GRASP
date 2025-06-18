import copy
import random
import numpy as np
import torch
import time

random.seed(42)
candi_points = np.linspace(0, 1, 51, endpoint=False)[1:]


def batch_query2cdfs(batch_qs, num_cols):
	# only serve for only join query (of multiple separated normal queries)
	batch_cdfs = []
	batch_signs = []
	max_num_cdfs = 0
	for query in batch_qs:
		cdfs, signs = query2cdfs(query, num_cols)

		if len(cdfs) > max_num_cdfs:
			max_num_cdfs = len(cdfs)

		batch_cdfs.append(cdfs)
		batch_signs.append(signs)

	normal_batch_cdfs = []
	normal_batch_signs = []

	for cdfs, signs in zip(batch_cdfs, batch_signs):
		for _ in range(max_num_cdfs - len(cdfs)):
			cdfs.append(np.zeros(num_cols))
			signs.append(0.)
		normal_batch_cdfs.append(cdfs)
		normal_batch_signs.append(signs)

	normal_batch_cdfs = np.array(normal_batch_cdfs)
	normal_batch_signs = np.array(normal_batch_signs)

	return normal_batch_cdfs, normal_batch_signs

def multi_queries_batch_query2cdfs(batch_qs, num_cols):
	# serve for multiple join queries (each containing multiple separated normal queries) for each table
	# batch_qs: [num_bs, num_normal_queries, query_on_table],
	# num_normal_queries_per_table is varying
	batch_cdfs = []
	batch_signs = []
	max_num_normal_queries = 0
	max_num_cdfs = 0
	query_masks = []

	for join_query in batch_qs:
		num_normal_queries = len(join_query)
		max_num_normal_queries = max(max_num_normal_queries, num_normal_queries)

		zeros_maks = []
		for _ in range(num_normal_queries):
			zeros_maks.append(1.)

		cdfs_per_join = []
		signs_per_join = []

		for normal_query in join_query:
			cdfs, signs = query2cdfs(normal_query, num_cols)
			max_num_cdfs = max(max_num_cdfs, len(cdfs))
			cdfs_per_join.append(cdfs)
			signs_per_join.append(signs)

		batch_cdfs.append(cdfs_per_join)
		batch_signs.append(signs_per_join)
		query_masks.append([1.] * num_normal_queries)

	# Create numpy arrays with appropriate shapes
	normal_batch_cdfs = np.zeros((len(batch_qs), max_num_normal_queries, max_num_cdfs, num_cols), dtype=np.float64)
	normal_batch_signs = np.zeros((len(batch_qs), max_num_normal_queries, max_num_cdfs), dtype=np.float64)
	normal_batch_masks = np.zeros((len(batch_qs), max_num_normal_queries), dtype=np.float64)

	# Fill the numpy arrays with data
	for i, (cdfs_per_join, signs_per_join) in enumerate(zip(batch_cdfs, batch_signs)):
		for j, (cdfs, signs) in enumerate(zip(cdfs_per_join, signs_per_join)):
			normal_batch_cdfs[i, j, :len(cdfs), :] = np.array(cdfs)
			normal_batch_signs[i, j, :len(signs)] = np.array(signs)
			normal_batch_masks[i, j] = 1.

	return normal_batch_cdfs, normal_batch_signs, normal_batch_masks

def multi_queries_batch_contexts(batch_q_contexts, like_embed_size):
	# batch_q_contexts: [num_bs, context_on_table]

	max_num_contexts = max(len(q_contexts) for q_contexts in batch_q_contexts)
	num_batches = len(batch_q_contexts)

	# Initialize numpy arrays for contexts, masks, and indicators
	batch_contexts = np.zeros((num_batches, max_num_contexts, like_embed_size), dtype=np.float64)
	batch_masks = np.zeros((num_batches, max_num_contexts), dtype=np.float64)
	batch_indicators = np.zeros(num_batches, dtype=np.float64)

	for i, q_contexts in enumerate(batch_q_contexts):
		num_contexts = len(q_contexts)
		if num_contexts > 0:
			# Fill in the contexts and masks
			batch_contexts[i, :num_contexts] = np.array(q_contexts)
			batch_masks[i, :num_contexts] = 1
			batch_indicators[i] = 1

	return batch_contexts, batch_masks, batch_indicators

def query2cdfs(query, num_cols):
	# query: a list of [col, op, val], col should be the id
	# col: column index
	# val: normalized value
	lower_list = np.zeros(num_cols, dtype=np.float64)
	upper_list = np.ones(num_cols, dtype=np.float64)

	for col, col_range in query:
		lower_list[col] = col_range[0]
		upper_list[col] = col_range[1]

	cdfs = [[]]
	lower_counts = [0]

	for i in range(num_cols):
		if lower_list[i] == 0. and upper_list[i] == 1.:
			for cdf in cdfs:
				cdf.append(1.)
		else:
			if upper_list[i] != 1. and lower_list[i] == 0.:
				for cdf in cdfs:
					cdf.append(upper_list[i])
			else:
				cdfs_copy = [cdf[:] for cdf in cdfs]
				lower_counts_copy = lower_counts[:]

				for cdf in cdfs:
					cdf.append(lower_list[i])
				for j in range(len(lower_counts)):
					lower_counts[j] += 1

				for cdf in cdfs_copy:
					cdf.append(upper_list[i])

				cdfs.extend(cdfs_copy)
				lower_counts.extend(lower_counts_copy)

	signs = (-1.) ** np.array(lower_counts)
	return cdfs, signs

def query2interval(query, num_cols, flip_start=False):
	queried_cols = []
	query_range = []
	upper_list = []
	lower_list = []

	for _ in range(num_cols):
		query_range.append([0., 1.])
		upper_list.append(1.)
		lower_list.append(0.)

	for q in query:
		col = q[0]
		col_range = q[1]
		query_range[col][0] = col_range[0]
		query_range[col][1] = col_range[1]

		lower_list[col] = col_range[0]
		upper_list[col] = col_range[1]

		if col not in queried_cols:
			queried_cols.append(col)

	interval = []
	for col_id in range(len(upper_list)):
		if not flip_start:
			interval.append(lower_list[col_id])
		else:
			interval.append(1. - lower_list[col_id])
		interval.append(upper_list[col_id])

	interval = torch.from_numpy(np.array(interval))
	interval = torch.unsqueeze(interval, 0)
	return interval

def batch_query2interval(batch_queries, num_cols, flip_start=False):
	batch_intervals = []

	for query in batch_queries:
		queried_cols = []
		query_range = []
		upper_list = []
		lower_list = []

		for _ in range(num_cols):
			query_range.append([0., 1.])
			upper_list.append(1.)
			lower_list.append(0.)
		for q in query:
			col = q[0]
			col_range = q[1]
			query_range[col][0] = col_range[0]
			query_range[col][1] = col_range[1]

			lower_list[col] = col_range[0]
			upper_list[col] = col_range[1]

			if col not in queried_cols:
				queried_cols.append(col)

		interval = []
		for col_id in range(len(upper_list)):
			if not flip_start:
				interval.append(lower_list[col_id])
			else:
				interval.append(1. - lower_list[col_id])
			interval.append(upper_list[col_id])

		batch_intervals.append(interval)
	batch_intervals = torch.from_numpy(np.array(batch_intervals))
	return batch_intervals

def multi_batch_query2interval(batch_queries, num_cols, flip_start=False):
	# for multiple join queries

	t1 = time.time()
	interval_res = []
	max_num_normal_qs = 0
	for join_q in batch_queries:
		batch_intervals = np.tile([0.0, 1.0], (len(join_q), num_cols))
		for i, query in enumerate(join_q):
			for q in query:
				col = q[0]
				col_range = q[1]
				if not flip_start:
					batch_intervals[i, 2 * col] = col_range[0]
				else:
					batch_intervals[i, 2 * col] = 1. - col_range[0]
				batch_intervals[i, 2 * col + 1] = col_range[1]
		interval_res.append(batch_intervals)
		max_num_normal_qs = max(max_num_normal_qs, len(join_q))

	for i, intervals in enumerate(interval_res):
		if len(intervals) < max_num_normal_qs:
			pad_size = max_num_normal_qs - len(intervals)
			padding = np.zeros((pad_size, 2 * num_cols), dtype=np.float64)
			interval_res[i] = np.vstack((intervals, padding))

	interval_res = torch.from_numpy(np.array(interval_res))
	t2 = time.time()
	print("multi_batch_query2interval {}".format(t2 - t1))

	return interval_res

def multi_batch_query2reps(batch_reps, OPS, num_cols, max_feat_len):
	# for multiple join queries
	max_num_preds = 0
	qreps_res = []
	for reps in batch_reps:
		feat_set = []
		for per_rep in reps:
			col_id = per_rep[0]
			op = per_rep[1]
			pfeat = per_rep[2]

			col_vec = np.zeros(num_cols, dtype=np.float64)
			col_vec[col_id] = 1.

			op_id = OPS.index(op)
			op_vec = np.zeros(len(OPS), dtype=np.float64)
			op_vec[op_id] = 1.

			if isinstance(pfeat, float):
				pfeat = np.array([pfeat])

			pred_vec = np.concatenate((col_vec, op_vec, pfeat))

			pad_size = max_feat_len - len(pred_vec)
			padding = np.zeros(pad_size, dtype=np.float64)
			pred_vec = np.concatenate((pred_vec, padding))

			feat_set.append(pred_vec)

		qreps_res.append(feat_set)

		if len(feat_set) > max_num_preds:
			max_num_preds = len(feat_set)

	batch_masks = np.zeros((len(qreps_res), max_num_preds), dtype=np.float64)

	if max_num_preds > 0:
		for i, q_reps in enumerate(batch_reps):
			num_preds = len(q_reps)
			if num_preds > 0:
				pad_size = max_num_preds - num_preds
				padding = np.zeros((pad_size, max_feat_len), dtype=np.float64)
				qreps_res[i] = np.vstack((qreps_res[i], padding))
				batch_masks[i, :num_preds] = 1
			else:
				qreps_res[i] = np.zeros((max_num_preds, max_feat_len), dtype=np.float64)
				batch_masks[i, 0] = 1
	else:
		qreps_res = np.zeros((len(qreps_res), 1,  max_feat_len), dtype=np.float64)
		batch_masks = np.ones((len(qreps_res), 1), dtype=np.float64)

	qreps_res = torch.from_numpy(np.array(qreps_res))
	batch_masks = torch.from_numpy(batch_masks)
	batch_masks = torch.unsqueeze(batch_masks, dim=-1)
	return qreps_res, batch_masks

def prepare_batch(queries_batch, num_cols, encoding_size):
	max_num_cdfs = 0
	batch_cdfs = []
	batch_signs = []
	for query in queries_batch:
		cdfs, signs = query2cdfs(query, num_cols)
		batch_cdfs.append(cdfs)
		batch_signs.append(signs)

		if len(cdfs) > max_num_cdfs:
			max_num_cdfs = len(cdfs)

	for cdfs, signs in zip(batch_cdfs, batch_signs):
		if len(cdfs) < max_num_cdfs:
			for _ in range(max_num_cdfs - len(cdfs)):
				cdfs.append([0.] * (encoding_size))
				signs.append(0.)

	return batch_cdfs, batch_signs, max_num_cdfs

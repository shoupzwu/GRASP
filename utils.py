import json
import numpy as np
from scipy.stats import gmean

def get_queries(q_file_name, num_samples=50000, test_size=1000):
	queries = []
	intervals = []
	sels = []

	q_file = open(q_file_name, 'r')
	lines = q_file.readlines()
	for line in lines:
		q_obj = json.loads(line)
		queries.append(q_obj['query'])
		tmp = q_obj['interval']
		interval = []
		for i in range(len(tmp[0])):
			interval.append(tmp[0][i])
			interval.append(tmp[1][i])
		intervals.append(interval)
		sels.append(float(q_obj['card']) / num_samples)

	candi_size = 50000

	training_intervals_tmp = intervals[:candi_size]
	training_sels_tmp = sels[:candi_size]
	training_queries_tmp = queries[:candi_size]

	training_queries = []
	training_intervals = []
	training_sels = []

	ood_test_queries = []
	ood_test_intervals = []
	ood_test_sels = []

	low_count = 0
	median_count = 0
	high_count = 0

	q_id = 0
	for q, interval, sel in zip(training_queries_tmp, training_intervals_tmp, training_sels_tmp):
		card = sel * num_samples
		if q_id < 300:
			low_count += 1
			training_queries.append(q)
			training_intervals.append(interval)
			training_sels.append(sel)
		elif q_id < 35000:
			median_count += 1
		else:
			high_count += 1
			ood_test_intervals.append(interval)
			ood_test_sels.append(sel)
			ood_test_queries.append(q)
		q_id += 1

	ood_test_intervals = ood_test_intervals[:test_size]
	ood_test_sels = ood_test_sels[:test_size]
	ood_test_queries = ood_test_queries[:test_size]

	training_size = len(training_queries)
	actual_training_size = int(training_size * 0.9)

	test_queries = training_queries[actual_training_size:training_size]
	test_intervals = training_intervals[actual_training_size:training_size]
	test_sels = training_sels[actual_training_size:training_size]

	training_queries = training_queries[:actual_training_size]
	training_intervals = training_intervals[:actual_training_size]
	training_sels = training_sels[:actual_training_size]

	return training_queries, training_intervals, training_sels, test_queries, test_intervals, test_sels, ood_test_queries, ood_test_intervals, ood_test_sels

def get_qerror(preds, targets, workload_type='ID-Distribution', num_data=50000):
	qerror = []
	preds = preds.cpu().data.numpy()
	targets = targets.cpu().data.numpy()
	rmse = 0.
	for i in range(len(targets)):
		if preds[i] <= 1. / num_data:
			preds[i] = 1. / num_data

		rmse += np.square(preds[i] - targets[i])

		if (preds[i] > targets[i]):
			qerror.append(preds[i] / targets[i])
		else:
			qerror.append(targets[i] / preds[i])

	rmse = np.sqrt(rmse / len(qerror))

	print("Test workload:{}: RMSE:{}, Median Qerror: {}".format(workload_type, rmse, np.median(qerror)))


def get_join_qerror(preds, targets, workload_type='In-Distribution', res_file=None, epoch_id=None):
	qerror = []
	for i in range(len(targets)):
		if preds[i] <= 1.:
			preds[i] = 1.

		if (preds[i] > targets[i]):
			qerror.append(preds[i] / targets[i])
		else:
			qerror.append(targets[i] / preds[i])

	print("Test workload:{}: Mean: {}, GMean: {}, Median: {}, 90: {}; 95: {}; 99: {}; Max:{}".format(
		workload_type, np.mean(qerror), gmean(qerror), np.median(qerror), np.percentile(qerror,90), np.percentile(qerror,95), np.percentile(qerror,99), np.max(qerror)))
	
	if res_file is not None:
		res_file.write("epoch_id:{}; Test workload:{}: Mean: {}, GMean: {}, Median: {}, 90: {}; 95: {}; 99: {}; Max:{} \n".format(
		epoch_id, workload_type, np.mean(qerror), gmean(qerror), np.median(qerror), np.percentile(qerror,90), np.percentile(qerror,95), np.percentile(qerror,99), np.max(qerror)))
		res_file.flush()
	return qerror


def extract_sublist(original_list, indices):
	return [original_list[i] for i in indices]
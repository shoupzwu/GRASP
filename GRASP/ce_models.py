import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from arcdf.transforms import autoregressive

class MSCNCE(nn.Module):
	def __init__(self, predicate_feats, hid_units):
		super().__init__()

		self.feature_num = predicate_feats

		self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
		self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
		self.predicate_mlp3 = nn.Linear(hid_units, hid_units)
		self.out_mlp1 = nn.Linear(hid_units, 1)
	
	def forward(self, predicates, predicates_masks):
		if predicates.dtype != torch.float64:
			predicates = predicates.to(torch.float64)
		hid_predicate = torch.relu(self.predicate_mlp1(predicates))
		hid_predicate = torch.relu(self.predicate_mlp2(hid_predicate))
		hid_predicate = torch.relu(self.predicate_mlp3(hid_predicate))

		hid_predicate = hid_predicate * predicates_masks  # Mask
		hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
		predicates_masks = predicates_masks.sum(1, keepdim=False)
		# predicates_masks = torch.where(predicates_masks == 0, 1., predicates_masks)
		hid_predicate = hid_predicate / predicates_masks
		sel_pred = torch.sigmoid(self.out_mlp1(hid_predicate))

		return sel_pred

class MLP(nn.Module):
	def __init__(self, predicate_feats, hid_units, join_bin_size=20):
		super().__init__()
		self.predicate_mlp1 = nn.Linear(predicate_feats,  hid_units)
		self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
		self.predicate_mlp3= nn.Linear(hid_units, hid_units)

		self.out_mlp1 = nn.Linear(hid_units, 1)
		self.out_mlp2 = nn.Linear(hid_units, join_bin_size)

		self.feature_num = predicate_feats
		self.rs = np.random.RandomState(42)
		random.seed(42)


	def forward(self, predicates):
		hid_predicate = torch.relu(self.predicate_mlp1(predicates))
		hid_predicate = torch.relu(self.predicate_mlp2(hid_predicate))
		hid_predicate = torch.relu(self.predicate_mlp3(hid_predicate))
		out = torch.sigmoid(self.out_mlp1(hid_predicate))

		join_bin_distribution = torch.softmax(self.out_mlp2(hid_predicate), dim=-1)

		return out, join_bin_distribution

class NeuCDF(nn.Module):
	def __init__(self, predicate_feats, hid_units, join_bin_size=20, context_size=None):
		super().__init__()
		self.predicate_mlp1 = nn.Linear(predicate_feats,  hid_units)
		self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
		self.predicate_mlp3= nn.Linear(hid_units, hid_units)

		self.context_size = context_size
		if self.context_size is not None:
			self.context_layer = nn.Linear(context_size, hid_units)

		self.out_mlp1 = nn.Linear(hid_units, 1)

		self.feature_num = predicate_feats
		self.rs = np.random.RandomState(42)
		random.seed(42)


	def forward(self, predicates, context=None):
		hid_predicate = torch.relu(self.predicate_mlp1(predicates))
		hid_predicate = torch.relu(self.predicate_mlp2(hid_predicate))
		hid_predicate = torch.relu(self.predicate_mlp3(hid_predicate))

		### add the LIKE context
		if context is not None:
			hid_predicate += torch.relu(self.context_layer(context))

		out = torch.sigmoid(self.out_mlp1(hid_predicate))

		return out

class AutoregressiveCDF(nn.Module):
	def __init__(self, predicate_feats, hid_units, join_bin_size=20):
		super().__init__()

		self.transform = autoregressive.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
			num_bins=30,
			features=predicate_feats,
			hidden_features=hid_units,
			num_blocks=3,
			use_residual_blocks=True,
			context_features=hid_units,
		)

		self.feature_num = predicate_feats
		self.rs = np.random.RandomState(42)
		random.seed(42)


	def forward(self, predicates, contexts=None):
		if predicates.dtype != torch.float64:
			predicates = predicates.to(torch.float64)
		if contexts is None:
			cdfs, _ = self.transform(predicates)
		else:
			cdfs, _ = self.transform(predicates, contexts)
		cdfs = torch.where(predicates == 1., 1., cdfs)
		out = torch.prod(cdfs, dim=-1)

		return out



















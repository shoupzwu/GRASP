import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LearnableCategorical(nn.Module):
    def __init__(self, num_categories=10):
        super(LearnableCategorical, self).__init__()
        self.params = nn.Parameter(torch.randn(num_categories))

    def forward(self):
        return torch.softmax(self.params, dim=0)

class UnifiedLCSPredictor(nn.Module):
	### LCS model
	def __init__(self, predicate_feats, hid_units, output_sizes):
		super().__init__()

		self.feature_num = predicate_feats
		self.out_mlp1 = {}
		self.output_sizes = output_sizes
		

		if self.feature_num > 0:
			self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
			self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
			self.predicate_mlp3 = nn.Linear(hid_units, hid_units)
		else:
			self.join_dist_tensor = nn.Parameter(torch.zeros(hid_units))
		
		self.update_context_layer = nn.Linear(hid_units, hid_units)

		self.output_layers = nn.ModuleList([nn.Linear(hid_units, out_size) for out_size in output_sizes])

	def forward(self, predicates, predicates_masks, output_indices=None, context=None):
		if self.feature_num > 0:
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
		else:
			hid_predicate = self.join_dist_tensor

		if output_indices is None:
			output_indices = list(range(len(self.output_layers)))

		if context is not None:
			stacked_tensors = torch.stack(context)
			# Compute the mean along the new dimension
			mean_tensor = torch.mean(stacked_tensors, dim=0)
			hid_predicate = hid_predicate + mean_tensor  # Updating the node embeddings

		outputs = {}
		for idx in output_indices:
			output = self.output_layers[idx](hid_predicate)
			softmax_output = F.softmax(output, dim=-1)
			outputs[idx] = softmax_output

		return outputs, hid_predicate


















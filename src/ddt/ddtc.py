from __future__ import annotations

import math

import torch as th


class DDTC(th.nn.Module):
    # shape info
    n_covs: int
    n_labels: int
    # parameters
    weights: th.nn.Parameter
    comparators: th.nn.Parameter
    alpha: th.nn.Parameter
    path_masks: th.nn.Parameter
    action_probs: th.nn.Parameter
    # tree info
    n_leaves: int
    depth: int

    def __init__(
        self,
        n_covs: int,
        n_labels: int,
        weights: th.Tensor,
        comparators: th.Tensor,
        leaves: list[tuple[list[int], list[int], th.Tensor]],
        alpha: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # shape info
        self.n_covs = n_covs
        self.n_labels = n_labels
        # tree info
        self.n_leaves = len(leaves)
        self.depth = int(math.floor(math.log2(self.n_leaves)))
        # passed in initial parameters
        self.weights = th.nn.Parameter(weights)
        self.comparators = th.nn.Parameter(comparators)
        self.alpha = th.nn.Parameter(th.as_tensor(alpha))
        self.path_masks = self._make_path_masks(leaves)
        self.action_probs = th.nn.Parameter(th.stack([leaf[2] for leaf in leaves]))

    def _make_path_masks(
        self, leaves: list[tuple[list[int], list[int], th.Tensor]]
    ) -> th.nn.Parameter:
        assert self.n_leaves == len(leaves)
        # (n_non_leaf_nodes, n_leaves)
        left_branches: th.Tensor = th.zeros((self.weights.shape[0], self.n_leaves))
        right_branches: th.Tensor = th.zeros((self.weights.shape[0], self.n_leaves))
        for n, (left_path, right_path, _) in enumerate(leaves):
            for i in left_path:
                left_branches[i, n] = 1.0
            for j in right_path:
                right_branches[j, n] = 1.0
        path_masks = th.nn.Parameter(
            th.stack([left_branches, right_branches]), requires_grad=False
        )
        return path_masks

    def forward(self, inputs: th.Tensor, return_logits: bool = True):
        assert inputs.shape[1] == self.n_covs
        # prepare inputs
        # (n_samps, 1, n_covs)
        inputs_: th.Tensor = inputs.T[None, :, :].permute(2, 0, 1)
        # propagate data through each node
        # mus are membership functions for each node
        # (n_samps, n_non_leaf_nodes, 1)
        mus: th.Tensor = (self.weights * inputs_).sum(dim=2)[:, :, None]
        mus = th.sigmoid((mus - self.comparators) * self.alpha)
        # (n_samps, n_non_leaf_nodes)
        mus = mus[:, :, 0]
        # (n_samps, n_leaves, n_non_leaf_nodes)
        left_path_probs: th.Tensor = self.path_masks[0].T[None, :, :]
        right_path_probs: th.Tensor = self.path_masks[1].T[None, :, :]
        left_path_probs = left_path_probs * mus[:, None, :]
        right_path_probs = right_path_probs * (1 - mus[:, None, :])
        # (n_samps, n_non_leaf_nodes, n_leaves)
        left_path_probs = left_path_probs.permute(0, 2, 1)
        right_path_probs = right_path_probs.permute(0, 2, 1)
        # We don't want 0s to ruin leaf probabilities, so replace them with 1s so they don't affect the product
        fillers: th.Tensor = th.where(self.path_masks == 0, 1.0, 0.0)
        left_path_probs = left_path_probs + fillers[0][None, :, :]
        right_path_probs = right_path_probs + fillers[1][None, :, :]
        # (n_samps, 2*n_non_leaf_nodes, n_leaves)
        path_probs: th.Tensor = th.cat((left_path_probs, right_path_probs), dim=1)
        # (n_samps, n_leaves)
        path_probs = path_probs.prod(dim=1)
        # (n_samps, n_labels)
        labels_logits: th.Tensor = path_probs @ self.action_probs
        if return_logits:
            return labels_logits
        labels_probs: th.Tensor = th.softmax(labels_logits, dim=1)
        return labels_probs

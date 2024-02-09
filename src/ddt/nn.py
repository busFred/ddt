from __future__ import annotations

import math
from abc import ABC

import torch as th


class DDT(th.nn.Module):
    # shape info
    n_covs: int
    n_responses: int

    # paramters
    weights: th.nn.Parameter
    comparators: th.nn.Parameter
    alpha: th.nn.Parameter
    path_masks: th.nn.Parameter
    values: th.nn.Parameter
    # tree info
    n_leaves: int
    depth: int

    def __init__(
        self,
        n_covs: int,
        n_responses: int,
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
        self.n_responses = n_responses
        # tree info
        self.n_leaves = len(leaves)
        self.depth = int(math.floor(math.log2(self.n_leaves)))
        # passed in initial parameters
        self.weights = th.nn.Parameter(weights)
        self.comparators = th.nn.Parameter(comparators)
        self.alpha = th.nn.Parameter(th.as_tensor(alpha))
        self.path_masks = self._make_path_masks(leaves)
        self.values = th.nn.Parameter(th.stack([leaf[2] for leaf in leaves]))

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

    def forward(self, inputs: th.Tensor):
        assert inputs.shape[1] == self.n_covs
        # prepare inputs
        # (n_samps, 1, n_covs)
        inputs_: th.Tensor = inputs.T[None, :, :].permute(2, 0, 1)
        # propagate data through each node
        # mus are relative membership functions for each node
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
        # (n_samps, n_responses)
        outputs: th.Tensor = path_probs @ self.values
        return outputs


class DDTWrapper(th.nn.Module, ABC):
    ddt: DDT

    @property
    def n_covs(self) -> int:
        return self.ddt.n_covs

    @property
    def n_responses(self) -> int:
        return self.ddt.n_responses

    # paramters
    @property
    def weights(self) -> th.nn.Parameter:
        return self.ddt.weights

    @property
    def comparators(self) -> th.nn.Parameter:
        return self.ddt.comparators

    @property
    def alpha(self) -> th.nn.Parameter:
        return self.ddt.alpha

    @property
    def path_masks(self) -> th.nn.Parameter:
        return self.ddt.path_masks

    @property
    def values(self) -> th.nn.Parameter:
        return self.ddt.values

    # tree info
    @property
    def n_leaves(self) -> int:
        return self.ddt.n_leaves

    @property
    def depth(self) -> int:
        return self.ddt.depth

    def __init__(self, ddt: DDT, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ddt = ddt

    def forward(self, inputs: th.Tensor, *args, **kwargs) -> th.Tensor:
        return self.ddt.forward(inputs)


class DDTC(DDTWrapper):
    @property
    def n_labels(self) -> int:
        return self.ddt.n_responses

    @staticmethod
    def make_ddtc(
        n_covs: int,
        n_labels: int,
        weights: th.Tensor,
        comparators: th.Tensor,
        leaves: list[tuple[list[int], list[int], th.Tensor]],
        alpha: float,
        *args,
        **kwargs,
    ) -> DDTC:
        ddtc = DDTC(
            ddt=DDT(
                n_covs=n_covs,
                n_responses=n_labels,
                weights=weights,
                comparators=comparators,
                leaves=leaves,
                alpha=alpha,
                *args,
                **kwargs,
            )
        )
        return ddtc

    def forward(self, inputs: th.Tensor, return_logits: bool = True):
        logits: th.Tensor = super().forward(inputs)
        if return_logits:
            return logits
        probs: th.Tensor = th.softmax(logits, dim=-1)
        return probs
